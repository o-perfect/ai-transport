# Strait risk model: plan for Stage 1 (anomaly features) and Stage 2 (HMM regimes)

Goal: from IMF PortWatch daily transit data alone, estimate for each major strait and week
the probability that it is **Normal**, **Stressed** (shaky, selectively avoided) or
**Disrupted** (largely closed or abandoned), plus how long the current state is likely to last.

This is a **nowcast + persistence forecast**, not a prediction of when a conflict starts.
Traffic reacts after events. Early warning from news is Stage 4, out of scope here.

---

## 0. Modelling assumptions (state these in the report)

| # | Assumption | Why it is plausible | How we check it |
|---|---|---|---|
| A1 | **Ship types decide differently.** Container lines reroute as whole fleets (liner schedules, alliance decisions, high cargo value). Tankers and bulkers decide ship by ship on the spot market. | Industry structure; Red Sea 2023–24, where container transits fell far more than tanker transits. In a true closure the difference disappears: at Hormuz since Mar 2026, containers and tankers are both down ~98%. | The fitted Stressed state should show a large container–tanker gap, and a full closure (Hormuz 2026) a gap near 0. Onset dates per type (Stage 3/PELT) should differ. If the gap never moves, the model falls back to level only and still works. |
| A2 | A calm-period baseline represents "normal" traffic for each strait. | Seasonality and weekday patterns are stable year to year. | Residuals in calm years should be centred on 0 with no seasonal shape left. |
| A3 | Regimes are **persistent** and switch as a Markov chain. | Rerouting decisions are sticky: contracts, insurance, schedules. | Fitted diagonal of A close to 1. Regime lengths in the backtest look roughly geometric. |
| A4 | After per-strait standardisation, **all straits share the same three states**. | Features are unitless anomalies, so "−2σ containers" means the same thing everywhere. | Leave-one-strait-out test (section 2.6). |
| A5 | AIS-based counts are a usable proxy for real traffic. | PortWatch is built on AIS. | Flag known AIS gaps. Treat Hormuz with care (spoofing, dark transits). |

---

## 1. Stage 1: Baseline and weekly anomaly features

### 1.1 Data pull
- Source: PortWatch `Daily_Chokepoints_Data` (already used in `agent/tools.py::_portwatch_query`).
- Range: 2019-01-01 to latest. PortWatch lags by a few days, so the last week may be partial and is dropped.
- Fields per day: `n_total`, `n_container`, `n_tanker`, `n_dry_bulk`, `n_general_cargo`,
  `n_roro`, `capacity` and the per-type capacity fields (verify exact names against the layer schema).
- **Strait pool** (high-traffic only): Suez, Bab el-Mandeb, Hormuz, Panama, Malacca, Bosporus,
  Gibraltar, Cape of Good Hope, Taiwan Strait. Optionally Dover, Lombok.
- Store one parquet/CSV per strait under `.cache/strait_risk/`, refreshed at most daily.
  The full history is about 2,500 rows per strait, so the volume is small.

### 1.2 Cleaning
- Reindex to a complete daily calendar and mark missing days.
- Flag suspected AIS outages: a day with `n_total` far below both neighbours on a calm date, or a whole
  strait dropping to 0 for a single day. Exclude them from fitting and from volatility.
- Keep an `events.yaml` with known episodes (dates, strait, type). It is used to exclude disturbed
  periods from the baseline and later for validation. Never used as model input.

### 1.3 Expected-traffic baseline (per strait, per series)
Fit on a **calm window** (by default 2019-01 to 2020-02 plus 2021-07 to 2023-10, minus that strait's events):

```
log(n_d + 0.5) = a + b·t + Σ_{k=1..3} [c_k sin(2πk·doy/365.25) + d_k cos(2πk·doy/365.25)] + DOW_d + ε
```

- Ordinary least squares (`numpy.linalg.lstsq`), no extra dependency.
- `t` is a linear trend. **Risk:** extrapolating a trend fitted in 2019–2023 into 2026 can drift.
  Fit with and without the trend and report the difference. Use no trend if they disagree badly.
- Fit separately for `n_total`, `n_container`, `n_tanker` and capacity-per-ship.
- Output: `expected_d` per day and the calm-window residual standard deviation `s` per series.

### 1.4 Daily anomalies
```
y_d = log(n_d + 0.5) − log(expected_d)
```
The +0.5 keeps zero-count days defined. That matters for container counts at Hormuz.

### 1.5 Weekly feature vector (per strait, ISO week)
| Feature | Definition | Captures |
|---|---|---|
| `f_level` | weekly mean of `y_d` for **containers** | How far the fleet-level movers have left |
| `f_gap` | weekly mean of `y_tanker − y_container` | **Selective avoidance**: high when shaky, ~0 when normal *and* when fully closed |
| `f_size` | weekly mean anomaly of capacity / transits (all types) | Physical constraint (draft limits) vs security threat |
| `f_vol` | excess volatility: `sqrt(max(0, var(y_d,total) − mean(1/n_d)))` over the 7 days, using MAD-based variance | Day-to-day instability beyond counting noise |

Then **z-score each feature per strait** using the calm-window weeks (mean and std of that feature).
That gives a common scale across straits (assumption A4) and down-weights noisy series
(e.g. containers at Hormuz).

Notes:
- `f_gap` replaces a separate tanker feature. Keeping container, tanker *and* gap would make the
  full covariance matrix singular.
- `f_vol` is optional. Keep it only if it improves BIC or backtest lag (section 2.4).
- Diversion (Cape up when the Red Sea is down) is deliberately **not** in the pooled vector,
  because the substitute route differs per strait and Hormuz has none. It goes into Stage 4.

### 1.6 Rule-based disruption index (Stage 1 deliverable and Stage 2 benchmark)
```
D_w = 1 − exp(weekly mean y_d for n_total)       # fraction of expected traffic missing
Normal: D < 0.15   Stressed: 0.15–0.40   Disrupted: > 0.40
```
Simple, explainable, and the benchmark the HMM has to beat.

### 1.7 Stage 1 checks
- Plot observed vs expected for each strait, with events shaded.
- Calm-period residuals: mean ≈ 0, no leftover seasonality (autocorrelation at lag 52 weeks small).
- Red Sea 2023–24: `D` crosses 0.40 at Bab el-Mandeb and Suez, `f_gap` clearly positive
  (selective avoidance).
- Hormuz from Mar 2026: `D` above 0.9 and `f_gap` near 0 (everyone stopped). Daily counts are
  0–5, so check that the +0.5 in the log keeps the anomalies finite and that `f_vol`'s
  counting-noise correction stops tiny counts from showing up as volatility.

**Stage 1 output:** table `strait, week, f_level, f_gap, f_size, f_vol, D, rule_state`.

---

## 2. Stage 2: Pooled hidden Markov model

### 2.1 Model
- Hidden state s_w ∈ {Normal, Stressed, Disrupted}, K = 3.
- Emission: x_w | s_w = k ~ N(μ_k, Σ_k), x_w = [f_level, f_gap, f_size, (f_vol)].
- Transition matrix A (K×K), shared by all straits. Initial distribution π.
- Library: `hmmlearn.hmm.GaussianHMM`, `covariance_type="full"` (try `"diag"` too).

### 2.2 Fitting
```python
X = np.vstack([features[s] for s in straits])
lengths = [len(features[s]) for s in straits]   # straits stay separate sequences
model.fit(X, lengths)
```
- Baum–Welch (EM). Run 20 random restarts and keep the best log-likelihood.
- **Seed the means** from Stage 1 so the states mean what we intend:
  Normal μ ≈ [0, 0, 0, 0], Stressed ≈ [−1.5, +1.5, 0, +1], Disrupted ≈ [−4, +1, ?, 0] (z units).
  The Disrupted gap is a compromise: about +2 for the Red Sea (tankers stayed) but about 0 for
  Hormuz 2026 (all types stopped). See the K = 4 test in 2.4.
- **Sticky prior:** `transmat_prior` (Dirichlet) with extra weight on the diagonal, because
  only a few switches exist in the data and the off-diagonal terms are otherwise unstable.
- If seeding is not enough, fix the Normal mean at 0 (`init_params`/`params` without `"m"`
  for that state). That makes the model semi-supervised, and the report should say so.

### 2.3 Naming the states
After fitting, order the states by μ_k[f_level]: highest = Normal, lowest = Disrupted.
Check that the middle state has the largest `f_gap` (assumption A1). If it does not, report it.

### 2.4 Model selection
Compare on BIC and on the backtest (section 2.6):
- K = 2, 3, 4. With two very different disrupted episodes in the data, K = 4 may split Disrupted
  into **Avoided** (Red Sea: level down, gap high) and **Closed** (Hormuz 2026: level near zero,
  gap ≈ 0). Accept the split only if BIC prefers it, not because it matches the story.
- full vs diagonal covariance
- with vs without `f_vol`, with vs without `f_size`
- trend vs no trend in the Stage 1 baseline

### 2.5 Inference (what the tool returns)
- **Filtered** probabilities α_w(k) = P(s_w = k | x_1..x_w) by a forward pass.
  `hmmlearn.predict_proba` returns *smoothed* probabilities, which use future weeks and would leak
  into a backtest. Write the forward recursion (about 15 lines) using `model._compute_log_likelihood`
  or `scipy.stats.multivariate_normal`.
- Forecast: p_{w+h} = α_w · A^h for h = 4, 8, 13 weeks.
- Expected duration of state k: 1 / (1 − A_kk). Survival P(still Disrupted after d weeks) ≈ A_kk^d.
- Caveat to state: geometric durations are memoryless.

### 2.6 Validation
The model is fitted **without labels**. Episodes (from `events.yaml`) are used only afterwards,
to compare the model's states with real history:

| Episode | Strait | Kind | What PortWatch shows | What should happen |
|---|---|---|---|---|
| Ever Given, Mar 2021 | Suez | 1-week accident | Short, sharp dip | Stays Normal or brief Stressed, **no** Disrupted |
| Drought draft limits, 2023–24 | Panama | Physical, gradual | Fewer and lighter-loaded ships | Stressed, driven by `f_size` rather than `f_gap` |
| Houthi attacks, Nov 2023 → | Bab el-Mandeb, Suez | Geopolitical, selective avoidance | ~65% below 2023, containers hit hardest; a further ~25% drop from Aug 2026 | Stressed → Disrupted within weeks, persistent; gap clearly positive |
| Israel–Iran war, Jun 2025 | Hormuz | Tension, no traffic effect | ~102/day in Apr–Jun 2025, no dip | Stays **Normal** (tests over-reaction) |
| Winter dip, Nov 2025 – Jan 2026 | Hormuz | Unclear | 55–67/day vs ~90–100 earlier in 2025; Dec 2024 was also low (~71) | Open question: seasonal, AIS/GPS jamming, or a lead-up to closure. Compare with the same months a year earlier before calling it Stressed |
| **Closure, Mar 2026 →** | **Hormuz** | **Near-total closure** | **~3/day vs ~86 a year earlier (−96%); containers and tankers both −98%** | **Disrupted (or Closed if K = 4) within 1–2 weeks, persistent; gap ≈ 0.** The main closure test case |

Notes on Hormuz 2026:
- PortWatch counts AIS-tracked **transits**. Ship-to-ship transfers at anchorages outside the strait
  are not counted, and ships sailing dark (AIS off) are missed, so true traffic may be somewhat above
  the recorded ~3/day. The model sees recorded traffic either way.
- The episode is still running, so it also tests the duration output: the expected remaining
  duration from 1/(1 − A_kk) should be reported with its geometric-duration caveat.
- It is the strongest test of A1: the Red Sea should look like selective avoidance (gap high) and
  Hormuz 2026 like a full stop (gap ≈ 0).

Metrics:
- **Detection lag:** weeks from the episode start to the first week with P(Disrupted or Stressed) > 0.5.
- **False alarms:** share of calm weeks with P(Normal) < 0.5.
- **Duration realism:** fitted 1/(1 − A_kk) vs actual episode lengths.
- **HMM vs rule-based index (1.6):** same metrics for both. The HMM should match lag with fewer
  false alarms (stickiness filters one-week blips).
- **Leave-one-strait-out:** refit without strait s, infer on s. If states still line up, pooling works (A4).
- **Assumption A1 finding:** container vs tanker onset dates per episode (simple: first week each
  series' z < −2; better: PELT in Stage 3).

---

## 3. Code layout and app integration

```
strait_risk/
  data.py       # fetch + cache PortWatch history, cleaning, AIS-gap flags
  baseline.py   # calm-window fit, expected_d, residual std
  features.py   # daily anomalies → weekly z-scored features, rule index D
  hmm.py        # fit, state naming, filtered forward pass, forecasts, save/load (joblib)
  evaluate.py   # backtest metrics, leave-one-strait-out, figures for the report
  events.yaml   # known episodes (validation only)
notebooks/strait_risk.ipynb   # figures for the report
```

- New dependency: `hmmlearn` (plus `joblib`, already pulled in by scikit-learn if present).
- Fit offline and save the model. The app only runs the forward pass on fresh data, which is cheap.
- Agent tool `assess_chokepoint_risk(chokepoint)` in `agent/tools.py`: returns `D`, state
  probabilities, expected remaining duration, 4/8/13-week forecast and the feature values, with
  units and source. One call, no extra LLM requests (keeps within `MAX_STEPS = 5`). Every number
  the agent quotes still comes from a tool result.
- Name mapping: `agent/network.py` has its own 10-chokepoint set and PortWatch uses other names.
  Reuse `PORTWATCH_CHOKEPOINTS` in `agent/tools.py`.

---

## 4. Milestones

1. **Data + baseline (Stage 1.1–1.3):** history cached for all pool straits; observed vs expected plots.
2. **Features + rule index (1.4–1.7):** weekly table; Red Sea sanity check passes.
3. **HMM fit (2.1–2.3):** three interpretable states; check of A1 on the Stressed state.
4. **Selection + validation (2.4–2.6):** backtest table, HMM vs rule index, leave-one-strait-out.
5. **Integration (3):** agent tool returning the latest state per strait.

## 5. Open risks
- Few episodes, so transition probabilities are rough. Report them with that caveat.
- Long-running disruptions (Bab el-Mandeb since late 2023) make the baseline depend on extrapolation from years ago.
- Hormuz has few containers, so `f_level`/`f_gap` are noisy there. z-scoring helps but does not fix it.
  If needed, use a tanker-based level for tanker-dominated straits.
- Hormuz since Mar 2026 has 0–5 transits a day. Its z-scores will be extreme (far outside anything
  in the calm window), which can pull a Gaussian state towards that one strait. Check whether the
  fitted Disrupted state is dominated by Hormuz weeks; if so, clip features (e.g. at −6σ) before fitting.
- AIS spoofing and dark transits, especially in the Gulf.
