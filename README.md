# AI-transport project

Intelligent agent for risk assessment and safeguard decision-making for international
transport corridors facing geopolitical shocks (TE8315, Topic 5).

One Streamlit app (`app.py`) with two pages:

- **Route planner** (`app_planner.py`): plan a sea route between any two ports, close
  chokepoints (Suez, Bab-el-Mandeb, Hormuz, …) and compare the detour's distance, time,
  cost and risk on a map.
- **Corridor risk agent** (`app_chat.py`, also `chat_cli.py`): an LLM agent that uses tools
  (live IMF PortWatch traffic, route + cost comparison, affected-link analysis) to write
  a risk assessment and contingency plan.

The pages talk through `st.session_state` (keys in `shared_state.py`): chokepoints closed
on the map are added to the chat prompt as the scenario, and the routes from the agent's
last `compare_routes` call are drawn on the map.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip ...
echo "SJTU_API_KEY=sk-..." > .env             # never commit this
```

## Run

```bash
.venv/bin/streamlit run app.py                      # route planner + agent chat
.venv/bin/streamlit run app_chat.py                 # chat UI only (no map)
.venv/bin/python chat_cli.py                        # terminal chat
.venv/bin/python chat_cli.py --no-tools -m qwen     # baseline: plain LLM, no tools
```

## Project layout

```
app.py                 Streamlit entry: page navigation, map scenario -> session state
app_planner.py         Route planner page + map (UI only); draws agent routes
app_chat.py            Agent chat page: tool-call trace, charts, export (chat_page())
shared_state.py        Session-state keys shared by the two pages
core/ports.py          ~3,900 world ports (bundled with the searoute package)
core/ships.py          Ship types + attributes; ShipType.to_features() for ML
core/chokepoints.py    Chokepoint registry: open/close, risk, add custom ones
core/routing.py        Sea routing (searoute network) avoiding closed chokepoints
core/costs.py          Fuel, time, tolls, war-risk insurance, risk score
data/chokepoints.json  Chokepoint state (the "switches")
data/ships.json        Ship type definitions

agent/core.py          Agent loop, system prompt, task understanding, report format
agent/tools.py         Tools: live PortWatch traffic, route + cost comparison, affected links
agent/network.py       Agent routing on searoute: port table, PortWatch -> passage map
chat_cli.py            Terminal chat for quick testing
```

## For the AI agent: flipping a chokepoint switch

Chokepoint state is stored in `data/chokepoints.json`. Change it, and the map
updates on the next refresh.

```python
from core.chokepoints import ChokepointRegistry
reg = ChokepointRegistry()
reg.set_status("babalmandab", "closed", note="Attacks on shipping")
reg.set_risk("hormuz", risk_score=0.9, war_risk_premium_pct=1.0)
reg.add("Danish Straits", lat=55.6, lon=10.9, radius_km=80, risk_score=0.1)
```

Or from the command line:

```bash
python -m core.chokepoints list
python -m core.chokepoints close hormuz "Tanker seizures"
python -m core.chokepoints open hormuz
```

## Notes / assumptions

- Routing uses the open-source `searoute` maritime network. Built-in passages
  (Suez, Panama, Malacca, …) are blocked natively. Custom chokepoints block
  every network edge inside their radius.
- Ship size limits apply automatically. For example, a VLCC is too deep for
  Suez and a 400 m container ship is too long for Panama.
- Costs are a simple, transparent model: fuel ∝ speed³, tolls × ship size
  factor, war-risk premium = % of hull value per risky chokepoint. All numbers
  are rough placeholders for a school project, not market data.
- The agent routes on the same `searoute` network. Its chokepoints are the IMF PortWatch
  chokepoints that have a searoute passage (Suez, Panama, Bosporus, Bab el-Mandeb,
  Malacca, Hormuz, Cape of Good Hope, Gibraltar, Sunda, Bering). Others (e.g. Taiwan,
  Dover) can't be blocked by the agent and come back as `unrecognised_blocked`.
  Alternatives are found by re-routing with each passage the route uses blocked.
- LLM API: `https://models.sjtu.edu.cn/api/v1`, models `qwen` (default, fastest),
  `deepseek-chat`, `deepseek-reasoner`. Limit 10 requests/min; one agent turn uses about 2-4 requests.
- PortWatch responses are cached for 6 h in `.cache/`.
- Agent cost figures (fuel burn, charter rates, canal tolls, war-risk premium) are
  planning assumptions in `agent/tools.py::VESSELS`; cite them as such in the report.
