# AI-transport project

Intelligent agent for risk assessment and safeguard decision-making for international
transport corridors facing geopolitical shocks (TE8315, Topic 5).

Two parts:

- **Shipping Route Planner** (`app.py`): plan a sea route between any two ports, close
  chokepoints (Suez, Bab-el-Mandeb, Hormuz, …) and compare the detour's distance, time,
  cost and risk on a map.
- **Corridor Risk Agent** (`app_chat.py`, `chat_cli.py`): an LLM agent that uses tools
  (live IMF PortWatch traffic, route + cost comparison, affected-link analysis) to write
  a risk assessment and contingency plan.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip ...
echo "SJTU_API_KEY=sk-..." > .env             # never commit this
```

## Run

```bash
.venv/bin/streamlit run app.py                      # route planner + map
.venv/bin/streamlit run app_chat.py                 # chat UI
.venv/bin/python chat_cli.py                        # terminal chat
.venv/bin/python chat_cli.py --no-tools -m qwen     # baseline: plain LLM, no tools
```

## Project layout

```
app.py                 Streamlit page + map (UI only)
core/ports.py          ~3,900 world ports (bundled with the searoute package)
core/ships.py          Ship types + attributes; ShipType.to_features() for ML
core/chokepoints.py    Chokepoint registry: open/close, risk, add custom ones
core/routing.py        Sea routing (searoute network) avoiding closed chokepoints
core/costs.py          Fuel, time, tolls, war-risk insurance, risk score
data/chokepoints.json  Chokepoint state (the "switches")
data/ships.json        Ship type definitions

agent/core.py          Agent loop, system prompt, task understanding, report format
agent/tools.py         Tools: live PortWatch traffic, route + cost comparison, affected links
agent/network.py       Coarse sea-lane graph + Dijkstra (to be replaced by searoute)
app_chat.py            Streamlit chat page: tool-call trace, charts, export (chat_page())
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
- LLM API: `https://models.sjtu.edu.cn/api/v1`, models `deepseek-chat`, `qwen`,
  `deepseek-reasoner`. Limit 10 requests/min; one agent turn uses about 2-4 requests.
- PortWatch responses are cached for 6 h in `.cache/`.
- Agent cost figures (fuel burn, charter rates, canal tolls, war-risk premium) are
  planning assumptions in `agent/tools.py::VESSELS`; cite them as such in the report.
