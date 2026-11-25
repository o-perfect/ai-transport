# shipping-route-planner

## What this is

A course project (TE8315, Topic 5) building an intelligent agent for **risk assessment and
safeguard decision-making on international transport corridors facing geopolitical shocks**
(canal/strait closures, conflicts, sanctions, port disruptions).

Concretely, this means: when a chokepoint like Suez, Bab-el-Mandeb, or Hormuz closes or gets
risky, figure out which trade lanes are hit, what the rerouting costs in distance/time/money,
and recommend mitigations.

## How the app is laid out

`app.py` is the single Streamlit page: the agent chat (`app_chat.py::chat_panel`) sits beside a
folium map. The app is natural-language only. The user describes a disruption, the agent calls
tools, and the routes and scenario it computes are written to `st.session_state`
(`shared_state.py`) and drawn on the map. Every quantitative claim must come from a tool call.

## Key modules

| File | Role | Used by the app? |
|---|---|---|
| `agent/core.py` | Agent loop (`run_agent`), system prompt, report format, max 5 LLM calls | yes |
| `agent/tools.py` | Tools: PortWatch traffic, news, `compare_routes`, `find_affected_routes` | yes |
| `agent/network.py` | Routing on the `searoute` network (fixed Cape, both-direction tags), port lookup, its own 10-chokepoint set | yes (the live router) |
| `core/costs.py` | Voyage cost model (fuel, charter, tolls, war risk, EU ETS) for every agent route | yes |
| `core/ships.py`, `data/ships.json` | 8 ship types; `fits()` checks canal limits | yes |
| `core/ports.py` | 3,917 world ports bundled with `searoute` | yes |
| `data/cost_assumptions.json` | Dated prices and sources behind `core/costs.py` | yes |
| `data/chokepoints.json` | Only size limits (`max_draft_m` etc.) are read by the app, via `core/costs.py` | partly |
| `core/routing.py`, `core/chokepoints.py` | Earlier route planner's router and registry | no: CLI/library only |

There is one cost model now (`core/costs.py`), but still two chokepoint sets and two routers
(`agent/network.py` vs `core/routing.py` + `core/chokepoints.py`). See `TODO.md` on branch
`docs/todo` for the known-issue list.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
echo "SJTU_API_KEY=sk-..." > .env       # only needed for chat; never commit this

streamlit run app.py                    # the app (map + chat; needs the API key)
python chat_cli.py                      # terminal chat
python chat_cli.py --no-tools -m qwen   # baseline comparison: plain LLM, no tools
```

Flipping a chokepoint in the `core/` registry (CLI only; the live app ignores it, since the agent sets the scenario):
```bash
python -m core.chokepoints close hormuz "Tanker seizures"
python -m core.chokepoints open hormuz
```

## Things to know before touching code

- Agent endpoint is `https://models.sjtu.edu.cn/api/v1`, rate-limited to 10 req/min; a single
  agent turn burns up to 5 requests. Keep that in mind when testing changes to `agent/core.py`.
- PortWatch tool responses are cached 6h in `.cache/`.
- `MAX_STEPS = 5` caps LLM calls per question (`agent/core.py`); the last step is forced to answer without tools.
- The agent's system prompt explicitly forbids inventing traffic/distance/cost/CO2 numbers —
  they must come from a tool result. Preserve this constraint in any prompt edits.
- `node_modules/`, `package.json`, `package-lock.json` at the repo root are stray (this is a
  Python/Streamlit project) — likely accidental `npm init`, not part of the actual app.
