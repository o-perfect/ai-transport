"""Public API: data-based risk state of a strait, for the agent and any other code.

    from strait_risk import assess, assess_many, weekly_table
    assess("hormuz")                 # JSON-serialisable dict for the latest week
    assess_many(["suez", "red sea"]) # several at once
    weekly_table("Suez Canal")       # full weekly history as a DataFrame

The model is fitted on first use (downloads about 9 x 2,900 daily rows, cached a
day under .cache/strait_risk/) and kept in memory; it refits once the data is
more than a day old. Safe to call from several threads at once.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import data, features
from .hmm import STATES, DiagGaussianHMM

SEVERITY = np.array([0, 50, 100])  # risk index = probability-weighted severity, 0-100
MAX_AGE_HOURS = 24
HORIZONS = (4, 8, 13)
SOURCE = "IMF PortWatch Daily_Chokepoints_Data (AIS); 3-state HMM fitted on 8 busy straits since 2019"
METHOD_NOTES = [
    "Data-based nowcast: weekly averages, so it lags real events by about 1-2 weeks "
    "(PortWatch delay plus weekly aggregation). Recent news can be ahead of it.",
    "States: Normal; Reduced (about 60% of normal traffic, tankers holding up better than "
    "containers: selective avoidance); Closed (about 6% of normal, every ship type gone).",
    "Durations and outlooks assume memoryless regimes and rest on few past episodes: rough.",
]

_lock = threading.Lock()
_cache = {}  # model, weeks (by strait name), fitted_at


# --- pipeline ------------------------------------------------------------------
def build(names, refresh=False):
    """Weekly feature tables for the given chokepoint names."""
    age = 0 if refresh else MAX_AGE_HOURS
    with ThreadPoolExecutor(max_workers=8) as pool:
        dfs = list(pool.map(lambda n: data.load(data.CHOKEPOINTS[n], max_age_hours=age), names))
    return {n: features.weekly(features.daily(df)) for n, df in zip(names, dfs)}


def annotate(model, w):
    """Add filtered state probabilities, state, risk index and run length to a weekly table."""
    post, prior = model.filter(w[features.FEATURES].to_numpy())
    for k, s in enumerate(STATES):
        w["P_" + s] = post[:, k]
        w["prior_" + s] = prior[:, k]
    w["hmm_state"] = [STATES[k] for k in post.argmax(1)]
    w["risk"] = post @ SEVERITY
    run = (w["hmm_state"] != w["hmm_state"].shift()).cumsum()
    w["weeks_in_state"] = w.groupby(run).cumcount() + 1
    return w


def fit(weeks):
    """Fit the pooled HMM on the training straits (seeded from the Stage 1 bands)."""
    names = [n for n in data.POOL if n in weeks]
    seqs = [weeks[n][features.FEATURES].to_numpy() for n in names]
    labels = [weeks[n]["rule_state"].map({s: i for i, s in enumerate(STATES)}).to_numpy() for n in names]
    model = DiagGaussianHMM().fit(seqs, labels)
    for n in weeks:
        annotate(model, weeks[n])
    return model


def get_model(refresh=False):
    """(model, weekly tables) — fitted once, refitted when older than MAX_AGE_HOURS."""
    with _lock:
        stale = not _cache or time.time() - _cache["fitted_at"] > MAX_AGE_HOURS * 3600
        if refresh or stale:
            weeks = build(list(data.POOL) + list(data.DIVERSION), refresh)
            model = fit(weeks)
            _cache.update(model=model, weeks=weeks, fitted_at=time.time())
        return _cache["model"], _cache["weeks"]


def weekly_table(chokepoint, refresh=False):
    """Annotated weekly history for any PortWatch chokepoint. Straits outside the
    training pool are scored with the pooled model's states."""
    name = data.resolve(chokepoint)
    if name is None:
        raise KeyError(f"Unknown chokepoint '{chokepoint}'. Known: {list(data.CHOKEPOINTS)}")
    model, weeks = get_model(refresh)
    with _lock:
        if name not in weeks:
            weeks[name] = annotate(model, build([name])[name])
        return weeks[name]


# --- assessment ----------------------------------------------------------------
def _gap_phrase(gap):
    if gap > 0.2:
        return "tankers holding up better than containers (selective avoidance)"
    if gap < -0.2:
        return "tankers down as much as or more than containers (broad stop)"
    return "containers and tankers affected about equally"


def assess(chokepoint, history_weeks=52):
    """Risk state of one chokepoint for its latest complete week, as a JSON-serialisable dict."""
    name = data.resolve(chokepoint)
    if name is None:
        return {"error": f"Unknown chokepoint '{chokepoint}'. Known: {list(data.CHOKEPOINTS)}"}
    try:
        w = weekly_table(name)
    except Exception as e:  # network or data problem: report it, don't crash the caller
        return {"error": f"Could not assess {name}: {type(e).__name__}: {e}"}
    if w.empty:
        return {"error": f"No usable PortWatch data for {name}"}
    model, _ = get_model()
    last = w.iloc[-1]
    p = np.array([last["P_" + s] for s in STATES])
    k = int(p.argmax())
    state = STATES[k]

    # Explanation: log-odds of the current state (or Reduced, if Normal) against Normal.
    other = state if state != "Normal" else "Reduced"
    prior = np.array([last["prior_" + s] for s in STATES])
    memory, parts = model.explain(w[features.FEATURES].to_numpy()[-1], prior, a=other)
    terms = dict(zip(features.FEATURES, parts))
    share = 100 * last.obs_total / last.exp_total if last.exp_total > 0 else None
    summary = (f"{state}: traffic at {share:.0f}% of normal for this time of year, "
               f"{_gap_phrase(last.gap)}; in this state for {int(last.weeks_in_state)} week(s).")

    notes = list(METHOD_NOTES)
    if name not in data.POOL:
        notes.append("This strait is not in the model's training pool; its state uses the pooled "
                     "states and is indicative only.")
    if name in data.DIVERSION:
        notes.append("The Cape of Good Hope gains traffic when the Red Sea is avoided: above-normal "
                     "traffic here signals diversion, not risk.")
    if last.exp_total < 20:
        notes.append("Low normal traffic at this strait: weekly figures are noisy.")

    tail = w.iloc[-history_weeks:]
    return {
        "chokepoint": name,
        "week_ending": str(last.name.date()),
        "data_through": w.attrs.get("data_through"),
        "state": state,
        "probabilities": {s: round(float(x), 3) for s, x in zip(STATES, p)},
        "risk_index": round(float(last.risk)),
        "risk_index_scale": "0-100: probability-weighted, Normal = 0, Reduced = 50, Closed = 100",
        "weeks_in_state": int(last.weeks_in_state),
        "summary": summary,
        "traffic_per_day": {
            "total": round(float(last.obs_total), 1), "total_expected": round(float(last.exp_total), 1),
            "total_share_of_normal_pct": None if share is None else round(share),
            "container": round(float(last.obs_container), 1),
            "container_expected": round(float(last.exp_container), 1),
            "tanker": round(float(last.obs_tanker), 1), "tanker_expected": round(float(last.exp_tanker), 1),
        },
        "container_tanker_gap": {"value": round(float(last.gap), 2), "meaning": _gap_phrase(last.gap)},
        "outlook": {
            f"P_{state.lower()}_in_{h}_weeks": round(float((p @ np.linalg.matrix_power(model.A, h))[k]), 2)
            for h in HORIZONS},
        "typical_state_length_weeks": round(float(model.durations()[k])),
        "explanation": {
            "compares": f"{other} vs Normal (log-odds; positive favours {other})",
            "total": round(float(memory + parts.sum()), 1),
            "memory_from_previous_weeks": round(float(memory), 1),
            "traffic_level": round(float(terms["level"]), 1),
            "container_tanker_gap": round(float(terms["gap"]), 1),
        },
        "simple_rule": {"missing_traffic_pct": round(100 * float(last.D)), "band": last.rule_state,
                        "bands": "Normal < 15% missing, Reduced 15-40%, Closed > 40%"},
        "in_training_pool": name in data.POOL,
        "notes": notes,
        "source": SOURCE,
        "_series": {
            "week": [str(d.date()) for d in tail.index],
            "risk_index": [round(float(x)) for x in tail.risk],
            "total_per_day": [round(float(x), 1) for x in tail.obs_total],
            "expected_per_day": [round(float(x), 1) for x in tail.exp_total],
            "state": list(tail.hmm_state),
        },
    }


def assess_many(chokepoints, history_weeks=52):
    """{name: assessment} for several chokepoints (unknown names give an error entry)."""
    get_model()  # fit once before fanning out
    return {c: assess(c, history_weeks) for c in chokepoints}


def assess_all(history_weeks=52):
    """Assessments for every strait in the training pool plus the Cape diversion route."""
    return assess_many(list(data.POOL) + list(data.DIVERSION), history_weeks)
