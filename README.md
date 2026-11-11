# AI-transport project: Shipping Route Planner

Plan a sea route between any two ports, close chokepoints (Suez, Bab-el-Mandeb,
Hormuz, …) and compare the detour's distance, time, cost and risk.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
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
