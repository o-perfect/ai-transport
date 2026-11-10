# AI-transport project

Intelligent agent for risk assessment and safeguard decision-making for international
transport corridors facing geopolitical shocks (TE8315, Topic 5).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "SJTU_API_KEY=sk-..." > .env      # never commit this
```

## Run

```bash
.venv/bin/streamlit run app_chat.py                 # chat UI
.venv/bin/python chat_cli.py                        # terminal chat
.venv/bin/python chat_cli.py --no-tools -m qwen     # baseline: plain LLM, no tools
```

## Structure

| File | Role (course module) |
|---|---|
| `agent/core.py` | Agent loop, system prompt, task understanding, report format |
| `agent/tools.py` | Tools: live PortWatch traffic, route + cost comparison, affected-link analysis |
| `agent/network.py` | Coarse sea-lane graph + Dijkstra (to be replaced/merged with the frontend route planner) |
| `app_chat.py` | Streamlit chat page: tool-call trace, charts, export (`chat_page()` is importable) |
| `chat_cli.py` | Terminal chat for quick testing |

## Notes

- API: `https://models.sjtu.edu.cn/api/v1`, models `deepseek-chat`, `qwen`, `deepseek-reasoner`.
  Limit 10 requests/min; one agent turn uses about 2-4 requests.
- PortWatch responses are cached for 6 h in `.cache/`.
- Cost figures (fuel burn, charter rates, canal tolls, war-risk premium) are planning
  assumptions in `agent/tools.py::VESSELS`; cite them as such in the report.
