# AI-transport project

Intelligent agent for risk assessment and safeguard decision-making for international
transport corridors facing geopolitical shocks (TE8315, Topic 5).

One Streamlit page (`app.py`), natural-language input only:

- **Chat** (right, `app_chat.py`; also `chat_cli.py`): an LLM agent that uses tools (live
  IMF PortWatch traffic, recent news, route + cost comparison, affected-link analysis) to write a risk
  assessment and contingency plan. Describe the disruption in words, e.g. "Suez is
  closed, options from Shanghai to Rotterdam?".
- **Map** (left): the routes from the agent's latest `compare_routes` call, with the
  chokepoints it treated as closed or high-risk, and a table of distance, days and cost.

The chat hands the routes to the map through `st.session_state` (keys in `shared_state.py`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip ...
echo "SJTU_API_KEY=sk-..." > .env             # never commit this
```

## Run

```bash
.venv/bin/streamlit run app.py                      # map + agent chat
.venv/bin/streamlit run app_chat.py                 # chat UI only (no map)
.venv/bin/python chat_cli.py                        # terminal chat
.venv/bin/python chat_cli.py --no-tools -m qwen     # baseline: plain LLM, no tools
```

## Project layout

```
app.py                 Streamlit page: map of the agent's routes (left), chat (right)
app_chat.py            Agent chat panel: tool-call trace, charts, export (chat_panel())
shared_state.py        Session-state keys shared by map and chat
core/ports.py          ~3,900 world ports (bundled with the searoute package)
core/ships.py          Ship types + attributes; ShipType.to_features() for ML
core/chokepoints.py    Chokepoint registry: open/close, risk, add custom ones
core/routing.py        Sea routing (searoute network) avoiding closed chokepoints
core/costs.py          Voyage cost model used by the agent (fuel, charter time, tolls,
                       war-risk insurance, EU ETS)
data/chokepoints.json  Chokepoint state (the "switches")
data/ships.json        The 8 ship types the agent can choose from
data/cost_assumptions.json  Fuel price, canal tolls, war-risk premiums, EU ETS, with sources
                       (core/routing.py and core/chokepoints.py are the earlier route planner's
                       logic; the agent uses core/costs.py, core/ships.py and core/ports.py)

agent/core.py          Agent loop, system prompt, task understanding, report format
agent/tools.py         Tools: live PortWatch traffic, news, route + cost comparison, affected links
agent/network.py       Agent routing on searoute: port table, PortWatch -> passage map
chat_cli.py            Terminal chat for quick testing
```

## Notes / assumptions

- Routing uses the open-source `searoute` maritime network. Built-in passages
  (Suez, Panama, Malacca, …) are blocked natively. A custom chokepoint with a
  `gate` (a line across the strait, e.g. the Taiwan Strait) blocks only the
  network edges crossing that line, so ports along the strait stay reachable.
  One without a gate blocks every edge inside its radius, which also cuts off
  any port inside that circle.
- Tests: `python -m unittest discover -s tests -t .`
- Ship size limits apply automatically. For example, a VLCC is too deep for
  Suez and a 400 m container ship is too long for Panama.
- Costs are hard-coded in `core/costs.py`; the LLM never calculates them. Per voyage:
  fuel (∝ speed³) + charter time (sea days + canal waiting days) + canal tolls
  (base toll × ship's toll factor) + war-risk insurance (% of hull value per risky
  chokepoint) + EU ETS (CO2 × 100% / 50% / 0% EU share × carbon price). Port fees are
  left out because they are equal on every route between the same two ports.
- The agent routes on the same `searoute` network. Its chokepoints are the IMF PortWatch
  chokepoints that have a searoute passage (Suez, Panama, Bosporus, Bab el-Mandeb,
  Malacca, Hormuz, Cape of Good Hope, Gibraltar, Sunda, Bering). Others (e.g. Taiwan,
  Dover) can't be blocked by the agent and come back as `unrecognised_blocked`.
  Alternatives are found by re-routing with each passage the route uses blocked.
- LLM API: `https://models.sjtu.edu.cn/api/v1`, models `qwen` (default, fastest),
  `deepseek-chat`, `deepseek-reasoner`. Limit 10 requests/min; one agent turn uses about 3-5 requests (capped at 5).
- PortWatch responses are cached for 6 h in `.cache/`.
- Voice input: the mic button in the chat records in the browser; `ui/speech.py` transcribes it
  locally with Whisper (`faster-whisper`, model "small", ~460 MB, downloaded from Hugging Face on
  first start, then offline). Browsers only allow the mic on `localhost` or HTTPS. Without
  `faster-whisper` installed the mic button is simply hidden.
- News comes from Google News RSS search (no key): gCaptain for maritime trade press
  (`site:gcaptain.com`) plus general news. Cached for 30 min in `.cache/`.
- Cost figures are planning assumptions in `data/ships.json` and
  `data/cost_assumptions.json` (with sources and date, Nov 2026); the agent cites them.
  Ship size limits come from `data/chokepoints.json` (e.g. a VLCC is too deep for Suez).
