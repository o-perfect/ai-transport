# TODO

Known bugs, hardcoded assumptions, and data that should be fetched or checked
instead of typed in. Found in a review of the codebase in November 2026. File
references are to `main` as of commit 829d3f6.

Already handled in separate PRs:

- **Taiwan Strait closure cut off 8+ coastal ports.** Closing a custom
  chokepoint also leaked. Branch `fix/taiwan-strait-gate`.
- **Runtime switch flips were written into the tracked
  `data/chokepoints.json`.** Branch `fix/chokepoint-runtime-state`.
- *(Already fixed on `main` by the move to `searoute`.)* The old hand-drawn agent
  graph sent New York → Los Angeles via Asia when Panama was closed.

## 1. Bugs

### 1.1 Malacca allows a fully laden VLCC
`data/chokepoints.json:61` sets Malacca's `max_draft_m` to **25.0**. The
commonly cited "Malaccamax" draft is about 20.5–21 m. The VLCC in
`data/ships.json` has a 22.5 m draft, so `core/` routes it through Malacca. In
reality, deep-laden VLCCs usually go via Lombok/Makassar, and neither is in
either chokepoint set. Check the real limit (Singapore Strait under-keel
clearance rules), set it, and consider adding Lombok.

### 1.2 The agent's "Risk assessment" rating comes from no tool
`agent/core.py:40` asks the LLM for a Low/Medium/High/Critical rating, but no
tool returns a risk score. The `risk_score` values in `core/` aren't
exposed to the agent. This goes against the system prompt's rule that numbers
must come from tools. Either add a tool that computes a rating from evidence
(see 2.1), or label the rating as the model's qualitative judgement.

### 1.3 The "2023 pre-crisis" baseline can include the crisis
`agent/tools.py:102` compares against the same window in 2023. Houthi attacks on
shipping began around 19 Nov 2023, and Red Sea traffic fell sharply from
December. A window that reaches into late Nov/Dec therefore compares against
crisis-hit data while calling it "pre-crisis". Use a fixed pre-crisis period
(e.g. the same window in 2022, or "up to 2023-11-15"), and make the cutoff a
named constant.

### 1.4 No upper limit on speed, so fuel figures blow up
`compare_routes` / `find_affected_routes` accept any `speed_knots`. Fuel scales with
(speed/ref)³, so a Capesize at 25 kn gives **4,464 t of fuel for a 14-day
Singapore → Rotterdam voyage** (about 320 t/day). Give each vessel a maximum
speed and either clamp to it or return an error.

### 1.5 `core/`: closing the Cape of Good Hope does nothing
In `core/routing.py`, the `south_africa` passage can be bypassed at 40°S, so
Shanghai → Rotterdam with Suez *and* Cape closed still goes around Africa
(13,517 nm). `agent/network.py:143` (`_network`) already works around this, but
`core/` uses `searoute`'s stock network. Move `core/` onto the same fixed
network (see 3.1).

### 1.6 Detours around South America are not named
With Panama closed, New York → Los Angeles correctly goes around South America
(12,873 nm), but its `chokepoints` list is empty. Magellan / Cape Horn isn't
a named chokepoint (`agent/network.py:52` notes the `chili` passage is Cape Horn
/ Drake). As a result the route shows as "open sea", and `alternatives()` cannot
branch off it. Add a "Cape Horn / Drake Passage" entry for `chili`.

## 2. Values that should be fetched, derived, or at least dated

None of these record a source or an "as of" date. Suggested fix: store every
reference value as `{value, unit, source, as_of}`, and have the agent cite
`source` and `as_of` in its "Assumptions" section.

### 2.1 Chokepoint status and risk (`data/chokepoints.json`)
| Field | Now | Better |
|---|---|---|
| `status` | Set by hand; only open/closed | Derive from PortWatch transits vs. baseline (the agent already fetches this) plus PortWatch disruption events; allow "reduced" / "convoy" states |
| `risk_score` | 0.05–0.7, guessed | Compute from transit drop, ACLED conflict events within X km, Lloyd's Joint War Committee listed areas, and active UKMTO / MARAD advisories |
| Risk combination | `core/costs.py:42` treats chokepoint risks as independent; Low/Med/High cut-offs at 0.3 / 0.6 (`costs.py:23-25`) | Document the method or replace it |
| `war_risk_premium_pct` | Bab 0.5, Hormuz 0.2, Bosporus 0.1; agent default 0.7 | Market rates change weekly: dated input, with JWC listing deciding whether a premium applies at all |
| `toll_usd` | Suez 400k, Panama 300k, × made-up `toll_factor` | Implement the Suez (SCNT) and Panama (PC/UMS / TEU) tariff formulas from ship GT / TEU |
| `max_draft_m` etc. | Single values | Suez draft depends on beam (a table); Panama draft is cut in droughts (~13.4 m in 2023), so read Panama Canal Authority advisories |
| `lat` / `lon` / `radius_km` | Hand-placed | Take positions from PortWatch's chokepoint layer; Malacca differs between `core/` (2.5, 101.2) and the agent (3.2, 100.6) |

### 2.2 Ships: two separate cost models
- `data/ships.json` (8 types, used by `core/`) and `agent/tools.py:34`
  `VESSELS` (4 types, used by the agent) have different numbers for similar
  ships. For example, the 20k TEU ULCV burns 160 t/day at 18 kn, while
  `container_large` (14k TEU) burns 80 t/day at 16 kn. Merge them into one source.
- `daily_cost_usd` / `charter_usd_day`: charter rates are very volatile. Use a
  dated input or an index (Harpex for containers, Baltic Exchange for bulk /
  tankers).
- `hull_value_usd`: guesses; e.g. the LNG carrier's 200M is below current
  newbuild prices. Valuation sources (VesselsValue, Clarksons) are paid, so label
  these as assumptions.
- The agent's `no_go` (Suezmax / Capesize can't use Panama) is hardcoded. Derive
  it from ship dimensions vs. canal limits, as `ShipType.fits()` already does in
  `core/`.
- Fuel model: the cube law also scales auxiliary engine burn, and ignores
  laden vs. ballast and weather. Fuel type isn't modelled: the LNG carrier is
  costed on bunker fuel.

### 2.3 Market and emissions inputs
- Bunker price default $600/t (`agent/tools.py:153`): one global number. Use a
  dated VLSFO / HSFO price, ideally per bunkering port.
- `CO2_PER_T_FUEL = 3.151` (`agent/tools.py:48`) is IMO's factor for LFO. HFO is
  3.114, MGO 3.206, LNG 2.750. Make it depend on fuel type.
- Default speed of 14 kn for every vessel. Default to each vessel's reference
  speed instead.
- War-risk insurance: the agent charges it **once** per voyage when any risky
  chokepoint is crossed (`agent/tools.py:130`); `core/` charges **per**
  chokepoint (`core/costs.py:37`). Pick one.
- Not modelled: canal transit and queue time (Suez ~12–16 h; Panama queues of
  days in 2023), port time, and the cost of cargo sitting in transit.

### 2.4 External-source constants
- PortWatch layer IDs `chokepoint1..28` and the ArcGIS org URL
  (`agent/tools.py:15-30`): read them from PortWatch's chokepoint metadata
  layer at startup.
- `MODELS` (`agent/core.py:15`): could be read from the endpoint's `/models`
  list.

## 3. Consistency and modelling

### 3.1 Two routing setups on the same network
`core/routing.py` and `agent/network.py` both wrap `searoute`, each with its own
fixes. `agent/` copies passage tags to both edge directions and fixes the
Cape. `core/` now also copies tags to both directions, but only for custom
chokepoints (once `fix/taiwan-strait-gate` is merged). Build the fixed network
in one place and use it from both.

### 3.2 Different chokepoint sets
`core/` has 10 (including Bosporus and a custom Taiwan Strait), the agent has 10
(including Bering, with no Taiwan), and PortWatch has 28. The agent can't block
Taiwan, Dover, Lombok, etc. Its names also differ from `core/`'s ("Bab el-Mandeb
Strait" vs "Bab-el-Mandeb (Red Sea / Yemen)"). Key a single registry by PortWatch
ID, with an optional `searoute` passage and an optional gate.

### 3.3 Ports
- The ports are `searoute`'s bundled list, with no depth or size limits, so a VLCC
  can be routed into a shallow port. The NGA World Port Index (free) has channel
  and berth depths.
- `core/ports.py:1` says ~3,700 ports and the README says ~3,900; the actual count is 3,917.
- `find_affected_routes` gives every hub pair equal weight. "Which trade lanes
  are hit" should be weighted by trade volume (PortWatch daily port data, UN
  Comtrade).
- `agent/network.py` aliases and substring matching can quietly replace the place
  the user asked about. For example, "red sea" resolves to Bab el-Mandeb Strait.
  Report the substitution back in the tool result.

### 3.4 Smaller items
- Northwest Passage is always blocked (`core/routing.py:91`,
  `agent/network.py:69`). That's reasonable, but the season isn't considered.
  The Northern Sea Route isn't considered at all.
- Custom chokepoints without a `gate` still block every edge within their
  radius, which cuts off ports inside the circle (`registry.add()` creates these).
  Consider requiring a gate, or warning when a circle contains ports.
- Tests: there are now unit tests under `tests/` (`python -m unittest discover -s
  tests -t .`). Worth adding: route distances vs. published port-to-port
  tables, chokepoint positions vs. PortWatch, every vessel × canal limit.
