"""Fit the MVP and print: the fitted states, the current state of every strait,
why, and how the model behaves on the validation episodes.

    python -m strait_risk            # uses cached data if younger than a day
    python -m strait_risk --refresh  # re-download PortWatch history
"""

import sys

import numpy as np
import pandas as pd

from . import data, features
from .api import build, fit
from .hmm import STATES

# Validation episodes. Never used for fitting, only compared against afterwards.
# The Hormuz 2026 start is approximate (first month of collapsed traffic).
EVENTS = [
    ("Ever Given blockage", "Suez Canal", "2021-03-23", "2021-03-29", "6-day closure: brief Closed, then Normal"),
    ("Panama drought limits", "Panama Canal", "2023-07-01", "2024-08-31", "Reduced"),
    ("Red Sea / Houthi attacks", "Bab el-Mandeb Strait", "2023-11-19", None, "Reduced or Closed, persistent"),
    ("Red Sea / Houthi attacks", "Suez Canal", "2023-11-19", None, "Reduced, persistent"),
    ("Israel-Iran war", "Strait of Hormuz", "2025-06-13", "2025-06-24", "no traffic effect: stay Normal"),
    ("Winter dip", "Strait of Hormuz", "2025-11-01", "2026-01-31", "open question"),
    ("Closure", "Strait of Hormuz", "2026-03-01", None, "Closed, persistent"),
]
def pct(log_ratio):
    return f"{100 * np.exp(log_ratio):.0f}%"


def print_model(model, weeks):
    n = sum(len(weeks[s]) for s in data.POOL)
    print(f"Fitted on {n} strait-weeks ({len(data.POOL)} straits, 2019 to now), "
          f"features {features.FEATURES}, {model.iterations} EM iterations\n")
    pooled = pd.concat([weeks[s] for s in data.POOL])
    print("STATES  (traffic = all transits as share of normal; gap = tanker minus container log-anomaly;")
    print("         container/size/volatility are not model inputs, shown as averages over each state's weeks)")
    print(f"{'state':<9} {'traffic':>8} {'gap':>6} {'containers':>11} {'ship size':>10} {'volatility':>11} "
          f"{'avg length':>11} {'weeks':>6}")
    for k, s in enumerate(STATES):
        m = model.means[k]
        sub = pooled[pooled.hmm_state == s]
        print(f"{s:<9} {pct(m[0]):>8} {m[1]:>+6.2f} {pct(sub.container.mean()):>11} {pct(sub['size'].mean()):>10} "
              f"{sub.vol.mean():>11.2f} {model.durations()[k]:>8.0f} wk {len(sub):>6}")
    print("\nTRANSITIONS per week (row = this week, column = next week)")
    print(" " * 9 + "".join(f"{s:>9}" for s in STATES))
    for k, s in enumerate(STATES):
        print(f"{s:<9}" + "".join(f"{p:>9.3f}" for p in model.A[k]))


def print_current(model, weeks):
    print("\nCURRENT STATE (latest complete week)")
    print(f"{'strait':<21} {'week':<11} {'transits/day':>15} {'rule D':>7} {'rule':>8} "
          f"{'P(N)':>5} {'P(R)':>5} {'P(C)':>5} {'risk':>5} {'HMM state':>10} {'same in 13 wk':>14}")
    for name in data.POOL:
        w = weeks[name].iloc[-1]
        p = np.array([w["P_" + s] for s in STATES])
        k = int(p.argmax())
        same = (p @ np.linalg.matrix_power(model.A, 13))[k]
        print(f"{name:<21} {w.name.date()!s:<11} {w.obs_total:>6.1f} vs {w.exp_total:>5.1f} "
              f"{w.D:>7.0%} {w.rule_state:>8} {p[0]:>5.2f} {p[1]:>5.2f} {p[2]:>5.2f} "
              f"{w.risk:>5.0f} {STATES[k]:>10} {same:>14.0%}")
    c = weeks["Cape of Good Hope"].iloc[-1]
    print(f"{'Cape of Good Hope':<21} {c.name.date()!s:<11} {c.obs_total:>6.1f} vs {c.exp_total:>5.1f} "
          f"{c.D:>7.0%}  diversion signal (not in the HMM): negative = more traffic than normal")


def print_explanations(model, weeks):
    print("\nWHY (latest week): log-odds of the HMM state vs Normal = memory + one term per feature")
    for name in data.POOL:
        w = weeks[name]
        state = w.hmm_state.iloc[-1]
        if state == "Normal":
            continue
        x = w[features.FEATURES].to_numpy()[-1]
        prior = np.array([w["prior_" + s].iloc[-1] for s in STATES])
        memory, parts = model.explain(x, prior, a=state)
        terms = "  ".join(f"{f} {v:+.1f}" for f, v in zip(features.FEATURES, parts))
        last = w.iloc[-1]
        print(f"  {name:<21} {state} vs Normal: {memory + parts.sum():+.1f} = memory {memory:+.1f}  {terms}")
        print(f"  {'':<21} total {last.obs_total:.1f}/day vs {last.exp_total:.1f} expected; "
              f"containers {last.obs_container:.1f} vs {last.exp_container:.1f}; "
              f"tankers {last.obs_tanker:.1f} vs {last.exp_tanker:.1f}")


def print_events(weeks):
    print("\nVALIDATION EPISODES (weeks overlapping the episode)")
    print(f"{'episode':<25} {'strait':<21} {'expected':<41} {'max D':>6} {'max P(R+C)':>10} "
          f"{'max P(C)':>8} {'1st flag':>9} {'weeks N/R/C':>12}")
    for label, strait, start, end, expect in EVENTS:
        w = weeks[strait]
        a, b = pd.Timestamp(start), pd.Timestamp(end) if end else w.index.max()
        seg = w[(w.index >= a) & (w.index <= b + pd.Timedelta(days=6))]
        if seg.empty:
            continue
        flagged = seg.index[seg.hmm_state != "Normal"]
        first = f"+{(flagged[0] - a).days} d" if len(flagged) else "never"
        counts = "/".join(str((seg.hmm_state == s).sum()) for s in STATES)
        print(f"{label:<25} {strait:<21} {expect:<41} {seg.D.max():>6.0%} "
              f"{(seg.P_Reduced + seg.P_Closed).max():>10.2f} {seg.P_Closed.max():>8.2f} "
              f"{first:>9} {counts:>12}")
    calm = pd.concat([weeks[n][features.calm_mask(weeks[n].index)] for n in data.POOL])
    print(f"\nFalse alarms: {np.mean(calm.P_Normal < 0.5):.1%} of {len(calm)} calm-period strait-weeks "
          f"not called Normal by the HMM (rule index: {np.mean(calm.rule_state != 'Normal'):.1%})")


def main():
    weeks = build(list(data.POOL) + list(data.DIVERSION), refresh="--refresh" in sys.argv)
    model = fit(weeks)
    print_model(model, weeks)
    print_current(model, weeks)
    print_explanations(model, weeks)
    print_events(weeks)
    out = data.CACHE / "weekly_results.csv"
    pd.concat({n: weeks[n] for n in data.POOL}, names=["strait", "week"]).to_csv(out)
    print(f"\nWeekly table: {out}")


if __name__ == "__main__":
    main()
