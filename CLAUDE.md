# shipping-route-planner

## What this is

A course project (TE8315, Topic 5) building an intelligent agent for **risk assessment and
safeguard decision-making on international transport corridors facing geopolitical shocks**
(canal/strait closures, conflicts, sanctions, port disruptions).

Concretely, this means: when a chokepoint like Suez, Bab-el-Mandeb, or Hormuz closes or gets
risky, figure out which trade lanes are hit, what the rerouting costs in distance/time/money,
and recommend mitigations.

## Two halves of the app

1. **Route map** (`app.py`, `core/`) — deterministic routing/costing engine. Plans a sea route
   between two ports on the real `searoute` network, lets you flip chokepoints open/closed, and
   computes the detour's distance, time, fuel/toll/insurance cost, and a risk score. No LLM
   involved; works without an API key.
2. **Agent chat** (`app_chat.py`, `agent/`, `chat_cli.py`) — an LLM tool-calling agent (via the
   SJTU OpenAI-compatible endpoint) that answers free-form questions by calling tools: live
   PortWatch traffic data, route/cost comparison, and affected-trade-lane analysis. Designed so
   every quantitative claim in its answer traces back to a tool call, not model invention.

`streamlit_app.py` is the combined entry point (sidebar to switch between the two pages).

## Key modules

| File | Role |
|---|---|
| `core/ports.py` | ~3,900 world ports (bundled with `searoute`) |
| `core/ships.py` | Ship types/attributes, `ShipType.to_features()` for ML |
| `core/chokepoints.py` | Chokepoint registry — open/close, risk, custom chokepoints; backed by `data/chokepoints.json` |
| `core/routing.py` | Sea routing on the `searoute` network, avoiding closed chokepoints |
| `core/costs.py` | Fuel, time, tolls, war-risk insurance, risk score |
| `agent/core.py` | Agent loop (`run_agent`), system prompt, final-report format, tool dispatch |
| `agent/tools.py` | Tool implementations + OpenAI function schemas the agent can call |
| `agent/network.py` | Coarse sea-lane graph + Dijkstra — flagged in README as "to be replaced/merged with the route planner" |

The two cost models (map's `data/ships.json`/`data/chokepoints.json` vs. agent's
`agent/tools.py::VESSELS`) are intentionally separate for now — not yet unified.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
echo "SJTU_API_KEY=sk-..." > .env       # only needed for chat; never commit this

streamlit run streamlit_app.py          # full app
streamlit run app.py                    # map only, no API key needed
streamlit run app_chat.py               # chat only
python chat_cli.py                      # terminal chat
python chat_cli.py --no-tools -m qwen   # baseline comparison: plain LLM, no tools
```

Flipping a chokepoint:
```bash
python -m core.chokepoints close hormuz "Tanker seizures"
python -m core.chokepoints open hormuz
```

## Things to know before touching code

- Agent endpoint is `https://models.sjtu.edu.cn/api/v1`, rate-limited to 10 req/min; a single
  agent turn burns ~2-4 requests. Keep that in mind when testing changes to `agent/core.py`.
- PortWatch tool responses are cached 6h in `.cache/`.
- `MAX_STEPS = 8` caps the agent's tool-calling loop (`agent/core.py`).
- The agent's system prompt explicitly forbids inventing traffic/distance/cost/CO2 numbers —
  they must come from a tool result. Preserve this constraint in any prompt edits.
- `node_modules/`, `package.json`, `package-lock.json` at the repo root are stray (this is a
  Python/Streamlit project) — likely accidental `npm init`, not part of the actual app.
