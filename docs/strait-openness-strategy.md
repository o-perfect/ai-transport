# Strategy for determining strait openness

How this project decides, from data, whether a maritime chokepoint is **open**, **partly avoided**
or **closed**, and how that judgement is combined with news. It covers the method, what the model
learned, how it was validated, the assumptions behind it and the limitations found while building it.

Code: `strait_risk/` (library and validation report), `agent/tools.py::assess_chokepoint_risk`
(agent tool). The original plan is in `docs/strait-risk-plan.md`. This document records what was
actually built and why it differs from that plan.

Figures below are from the run on PortWatch data through **15 Nov 2026**.

---

## 1. The question and the approach

**Question:** is ship traffic through a strait behaving normally, reduced, or stopped, and how long is
that state likely to last?

**Approach:** a **nowcast** from observed transits, not a forecast of geopolitical events. Traffic
reacts *after* something happens, so this model tells us what ships are actually doing; news tells us
what is declared or threatened. The two are combined by explicit rules (section 7).

Pipeline:

```
PortWatch daily transits (2019 to now, per strait)
  -> Stage 1: seasonal baseline per strait -> daily anomalies -> weekly features
  -> Stage 2: 3-state hidden Markov model (HMM), fitted unsupervised over 8 straits
  -> per strait and week: state, probabilities, risk index, outlook, explanation
```

---

## 2. Data

- **Source:** IMF PortWatch `Daily_Chokepoints_Data`, based on AIS ship positions. One row per
  chokepoint per day since 1 Jan 2019, with transit counts by ship type (`n_total`, `n_container`,
  `n_tanker`, `n_dry_bulk`, …) and deadweight `capacity`.
- **Training straits (8):** Suez, Panama, Bosporus, Bab el-Mandeb, Malacca, Hormuz, Gibraltar, Taiwan.
  These are busy enough for stable weekly figures.
- **Kept out of training:** the Cape of Good Hope (it *gains* traffic when the Red Sea is avoided,
  see section 8.3) and low-traffic straits. Any of the 28 PortWatch chokepoints can still be scored
  with the fitted model; those outside the training set are flagged "indicative only".
- **Caching:** full histories are cached for a day in `.cache/strait_risk/`.

---

## 3. Stage 1: what is "normal", and how far off is this week?

### Seasonal baseline
For each strait and series (total, container, tanker transits), a regression is fitted on **calm
periods only** (Jan 2019 – Feb 2020 and Jul 2021 – May 2023, i.e. before COVID, after the COVID
recovery, and before the Panama drought and the Red Sea crisis):

```
log(n + 0.5) = intercept + 3 yearly sine/cosine pairs + day-of-week effects
```

The fit is **robust**: days with residuals above 4 robust standard deviations (e.g. AIS outages) are
dropped and the model refitted, three times. There is **no trend term**, to avoid extrapolating
2019–2023 growth into 2026.

### Daily anomaly
`y = log(observed + 0.5) − log(expected)`. A value of −0.69 means half of normal; −2.3 means a tenth.
The `+0.5` keeps zero-count days defined (important at Hormuz in 2026).

### Weekly features (model inputs)
| Feature | Definition | Meaning |
|---|---|---|
| `level` | weekly mean anomaly of **total** transits, capped at +0.3 | How much traffic is missing |
| `gap` | weekly mean tanker anomaly − container anomaly | > 0: tankers hold up better than containers (selective avoidance); ≈ 0 or < 0: everyone stops |

Weeks with fewer than 5 days of data are dropped.

### Simple rule (Stage 1 result and benchmark)
`D = 1 − exp(level)` = share of expected traffic missing. Bands: **Normal** < 15%, **Reduced**
15–40%, **Closed** > 40%. These bands also provide the HMM's starting values.

---

## 4. Stage 2: the hidden Markov model

### Model
- Three hidden states. Each week the strait is in one of them, and the state persists or switches
  week to week according to a transition matrix `A`.
- In each state, each feature follows a normal distribution with its own mean and spread
  (diagonal covariance: features are independent given the state).
- Fitted with Baum–Welch (EM) over all 8 straits at once. Each strait is a separate sequence, but
  they **share** the state definitions and the transition matrix.
- A **sticky prior** (Dirichlet, +20 pseudo-counts on the diagonal of `A`) favours persistence,
  because there are very few real switches in the data.
- Starting values come from the Stage 1 bands. **No event dates or labels are used in fitting.**
- Implemented in plain numpy (`strait_risk/hmm.py`, ~100 lines); no HMM library needed.

### What the model reports
- **Filtered probabilities** P(state this week | data up to this week). These use no future data,
  so backtests are honest. (A library's default "smoothed" probabilities would look ahead.)
- **Risk index 0–100:** probability-weighted severity, Normal = 0, Reduced = 50, Closed = 100.
- **Outlook:** P(same state in 4/8/13 weeks) = current probabilities × `A^h`.
- **Typical state length:** 1 / (1 − A_kk) weeks.
- **Weeks in current state** (run length).
- **Explanation** (section 6).

### What it learned (unsupervised)

| State | Traffic vs normal | Gap | Typical length | Weeks in data | Interpretation |
|---|---|---|---|---|---|
| **Normal** | 100% | −0.04 | ~5 years | 2,887 | — |
| **Reduced** | 59% | **+0.39** | ~65 weeks | 363 | Selective avoidance: containers leave, tankers keep going more (Red Sea 2024–26) |
| **Closed** | 5% | −0.59 | ~57 weeks | 38 | Everything stops (Hormuz 2026, Ever Given week) |

The model was not told about "selective avoidance" versus "full stop". It separated them by itself
through the container–tanker gap, which supports assumption A1 (section 9).

Weekly transition matrix (row = this week, column = next week):

|  | Normal | Reduced | Closed |
|---|---|---|---|
| **Normal** | 0.996 | 0.003 | 0.001 |
| **Reduced** | 0.015 | 0.985 | 0.000 |
| **Closed** | 0.018 | 0.000 | 0.982 |

---

## 5. Validation

Historical episodes are used **only after fitting**, to compare the model's states with real
history (external validation).

| Episode | Strait | Expected | Result |
|---|---|---|---|
| Ever Given blockage, Mar 2021 | Suez | Brief Closed, then Normal | Closed exactly 1 week (20 transits/day vs 61), Normal the next week ✓ |
| Drought limits, 2023–24 | Panama | Reduced | Reduced 25 of 62 weeks, from about Nov 2023 when caps were tightest; partial ✓ |
| Houthi attacks, Nov 2023 → | Bab el-Mandeb | Reduced/Closed, persistent | Reduced 152 weeks, never Closed; flagged 35 days after the first attacks ✓ |
| Houthi attacks, Nov 2023 → | Suez | Reduced, persistent | Reduced 150 weeks; flagged after 49 days ✓ |
| Israel–Iran war, Jun 2025 | Hormuz | No traffic effect: Normal | Normal ✓ |
| Winter dip, Nov 2025 – Jan 2026 | Hormuz | Open question | Normal (the simple rule says Reduced) |
| Closure, Mar 2026 → | Hormuz | Closed, persistent | Closed within 1 week, 37 of 38 weeks ✓ |

**False alarms:** in calm periods, 0.6% of 1,280 strait-weeks were not called Normal by the HMM,
against 9.5% for the simple rule. Persistence filters out one-week blips.

### Current state (week ending 15 Nov 2026)

| Strait | Transits/day vs normal | State | Risk |
|---|---|---|---|
| Hormuz | 2.3 vs 75.2 (3%) | Closed | 100 |
| Bab el-Mandeb | 25.6 vs 61.5 (42%) | Reduced | 50 |
| Bosporus | 48.6 vs 102.3 (47%) | Reduced | 50 |
| Suez | 46.1 vs 61.9 (75%) | Reduced (borderline, held mostly by memory) | 49 |
| Panama, Malacca, Gibraltar, Taiwan | — | Normal | 0 |
| Cape of Good Hope | 75.0 vs 42.5 (+75%) | Normal (diversion surge, see 8.3) | 0 |

**Bosporus** was found by the model without prompting: traffic has declined since June 2026 and held
at about half of normal since September, across all ship types, with dry bulk down most (~60%). The
engine-failure suspension in October 2026 shows only as a one-day dip (16 and 20 transits on 14–15 Oct)
and does not explain it. The cause is not established; a Black Sea-related cause is plausible but
unconfirmed.

---

## 6. Explainability

The model is small enough to be read directly, so it is not a black box.

1. **States are readable:** each is a mean traffic level and gap that convert to "% of normal"
   (table in section 4).
2. **Each week is explainable term by term.** With diagonal covariance the log-odds of a state against
   Normal split exactly into:

   `log-odds = memory term + level term + gap term`

   - **Memory term:** what last week's state and `A` imply (the model's inertia).
   - **Feature terms:** how strongly each feature favours the state this week.

   Example, Bab el-Mandeb, latest week: Reduced vs Normal = +33.1 = memory +4.2, level +18.6, gap +10.3.
   Example, Suez: +3.6 = memory +3.7, level +1.3, gap −1.4. Its "Reduced" call rests almost entirely
   on the previous weeks.
3. **One headline number:** the risk index, always reported next to the raw counts it rests on.
4. **The model explains the state, not the cause.** Why ships stopped is not in transit data; news
   provides that.

---

## 7. How it is used

### As a library
```python
from strait_risk import assess, assess_many, weekly_table
assess("hormuz")            # dict: state, probabilities, risk index, traffic vs normal,
                            # outlook, explanation, notes, 52-week series for charts
weekly_table("Suez Canal")  # full annotated weekly history (DataFrame)
```
Fitted on first use (~9 s including download), then cached in memory and refitted daily. Thread-safe.

### As an agent tool
`assess_chokepoint_risk(chokepoints=[...])`, up to 6 per call. The system prompt combines it with news
using these rules:

| News | Model | Reading |
|---|---|---|
| "Closed" | Closed | Confirmed closure |
| "Closed" | Normal/Reduced | Too recent for the data, or only partial: treat as closed for decisions now, and say the data does not show it yet |
| Nothing | Reduced/Closed | Unexplained change: flag it (possible data issue) |
| "Reopened" | Still Reduced/Closed | Formally open, ships not back yet |

Plus: a one-day incident in the news does not explain a change lasting weeks. Every number the agent
quotes still comes from a tool result.

### Validation report
`python -m strait_risk` prints the fitted states, current states, explanations and the episode table.

---

## 8. Design decisions made from the data (deviations from the plan)

### 8.1 Features reduced to level and gap
The plan had four inputs (container level, gap, ship size, volatility). Variant runs showed:
- **Volatility** turned the worst state into "bad-weather week": typhoon weeks in the Taiwan Strait and
  storm or one-way-traffic weeks at the Bosporus were called the worst state.
- **Ship size** used up a state to separate slightly-above-normal weeks from normal ones.
- **Container-only level** missed the Panama drought and the Bosporus decline. **Total** traffic
  catches both while still separating the Red Sea from Hormuz.

Size and volatility are still computed and shown for each state, but not used as inputs.

### 8.2 States renamed
The plan's Normal / Stressed / Disrupted became **Normal / Reduced / Closed**, named after what the
fitted states actually contain.

### 8.3 Above-normal traffic capped (the Cape of Good Hope problem)
Without a cap, the Cape of Good Hope, whose traffic rose to +75% of normal when ships diverted from the
Red Sea, was scored **Closed** (P = 0.97) from January 2024 onward. Why:

- A Gaussian state scores a week by how many spreads it lies from the state's centre. Normal is narrow
  (sd 0.14) and Closed is wide (sd 0.91).
- The Cape's level of +0.56 was 4.0 spreads from Normal and 3.7 from Closed. The two states found it
  about equally implausible (log-likelihood −7.5 against −7.6).
- The model has no notion of direction: "far above normal" and "far below normal" both just look far.
- The gap (−0.34) was slightly closer to Closed, which tipped the near-tie.
- Persistence then held it: Closed for 147 of the following 150 weeks.

**Fix:** the level input is capped at +0.3 (about 35% above normal), so above-normal traffic counts as
normal. The Cape now scores clearly Normal (−2.2 against −12.6), and the validation results did not
change.

### 8.4 Robust baseline instead of event-based exclusions
The baseline excludes outlier days automatically instead of using a hand-made list of event dates,
which keeps event knowledge out of the model.

---

## 9. Assumptions

| # | Assumption | Status |
|---|---|---|
| A1 | **Ship types decide differently.** Container lines reroute as whole fleets (liner schedules, alliance decisions, high cargo value); tankers and bulkers decide ship by ship on the spot market. | Supported: the model found a Reduced state with tankers holding up better (gap +0.39), and a Closed state where all types stop. |
| A2 | Calm periods (2019, mid-2021 to mid-2023) represent normal seasonal traffic for each strait. | Reasonable, but normal levels may have drifted since (section 10). |
| A3 | Regimes are persistent and switch as a Markov chain (memoryless). | Persistence fits well; memorylessness does not fit closures (section 10). |
| A4 | After expressing traffic relative to each strait's own normal, all straits share the same three states. | Works for the training straits; Hormuz and the Red Sea separate cleanly. Not tested by leaving one strait out. |
| A5 | AIS-based transit counts are a usable proxy for real traffic. | Generally yes; weaker in the Gulf (section 10). |
| A6 | Above-normal traffic is not a risk signal at that strait. | Design choice (section 8.3). Diversion surges are risk evidence for *another* strait. |
| A7 | Weekly averages are the right resolution. | Trades 1–2 weeks of lag for much less noise and fewer false alarms. |

---

## 10. Limitations

**Method**
- **Design chosen while looking at the test cases.** Features and the level cap were chosen after
  seeing results on the validation episodes. This amounts to tuning on the test set, so the
  validation table is optimistic. A clean test would freeze the design and evaluate on a held-out
  period or strait.
- **Few episodes.** The transition probabilities rest on a handful of switches. Off-diagonal values
  (e.g. Normal → Closed = 0.001) are rough.
- **Memoryless durations.** The Closed state mixes a 6-day blockage with a 9-month closure, so its
  single typical length (~57 weeks) fits neither. The outlook figures (e.g. 79% that Hormuz is still
  Closed in 13 weeks) are rough indications.
- **A state is always assigned.** A Gaussian HMM gives some state to every input, even an unusual one.
  Outside the training range the widest state tends to win, regardless of meaning (section 8.3). There
  is currently no "no state fits well" warning.
- **Probabilities are not calibrated.** With so few episodes they cannot be checked against real
  frequencies. They are model-based confidence, and are often near 0 or 1.
- **Extreme explanation values.** Because Normal is narrow, a 97% traffic drop gives log-odds of +360.
  The ranking of evidence terms is meaningful; the magnitudes are not.

**Data**
- **Lag:** PortWatch publishes with a few days' delay, and weekly aggregation adds up to a week. The
  model cannot see the first 1–2 weeks of an event.
- **AIS limits:** ships sailing "dark" (AIS off) are not counted, and GPS jamming in the Gulf distorts
  positions. Real Hormuz traffic may be somewhat above the recorded ~3/day. Ship-to-ship transfers at
  anchorages outside the strait are not transits and are not counted.
- **Aggregate counts only:** no flag, owner or cargo origin, so selective avoidance by ownership
  (e.g. Israel- or US-linked ships avoiding the Red Sea) can only be inferred indirectly through the
  type mix.
- **Baseline age:** normal levels are fitted on 2019–2023 without a trend. Structural change since
  then (fleet growth, new trade patterns) is read as anomaly.
- **Low-traffic straits:** few daily transits make weekly figures noisy. Hormuz containers in 2026
  are often 0 or 1 a day.

**Scope**
- **Not a forecast of events.** The model says what traffic is doing and how persistent that has
  historically been, not whether a conflict will start or end.
- **Physical vs security causes look similar.** Panama's drought and the Red Sea's attacks both appear
  as Reduced. Ship size could separate them but is not used (section 8.1).
- **Unexplained detections need checking.** The Bosporus decline is real in the data, but its cause
  is unknown.

---

## 11. Possible next steps

- Freeze the design and evaluate on a held-out period or strait (leave one strait out).
- Add a "no state fits well" flag when even the best state's likelihood is very low.
- Report diversion (Cape of Good Hope surge) as evidence of Red Sea risk in a combined score (Stage 4).
- Add changepoint detection (CUSUM or Bayesian online) for faster onset alerts (Stage 3).
- Test a state-length model that is not memoryless (hidden semi-Markov) once more episodes exist.
- Measure how much each ship type leads or lags at onset, as a direct test of A1.
