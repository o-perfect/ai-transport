"""Tool-calling agent loop over the SJTU OpenAI-compatible API."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .tools import FUNCTIONS, SCHEMAS

load_dotenv()

BASE_URL = "https://models.sjtu.edu.cn/api/v1"
MODELS = ["deepseek-chat", "qwen", "deepseek-reasoner"]
MAX_STEPS = 8

SYSTEM_PROMPT = """You are a maritime transport risk analyst agent. You help shipping, port and \
logistics planners assess geopolitical shocks to international transport corridors (canal or strait \
closures, conflicts, sanctions, port disruptions) and produce contingency plans.

Rules:
- Every number about traffic, distance, time, cost or CO2 must come from a tool result. Never invent \
or estimate such figures yourself. If a tool cannot answer, say so.
- Plan first: work out which tools you need, call them (several in one step if independent), then reason.
- For a closure scenario, typically: check live traffic at the chokepoint, find the affected links, \
compare routes for the specific trade lane(s) the user cares about.
- If the user's port or chokepoint is not supported, pick the nearest supported one and say so.
- Reply in the user's language.

Final answer format (markdown):
1. **Situation** - what is disrupted and current traffic evidence.
2. **Affected links** - which trade lanes are hit and how badly.
3. **Route options** - a table: route (via), distance nm, sea days, total cost USD, extra vs baseline.
4. **Risk assessment** - a qualitative rating (Low/Medium/High/Critical) with reasons.
5. **Recommendations** - concrete short-term and medium-term mitigation actions.
6. **Assumptions & sources** - key parameters used and data sources (IMF PortWatch, network model).
"""

BASELINE_PROMPT = """You are a maritime transport risk analyst. Answer the user's question about \
geopolitical shocks to shipping routes as well as you can, with concrete figures where possible."""


def make_client():
    key = os.environ.get("SJTU_API_KEY")
    if not key:
        raise RuntimeError("SJTU_API_KEY not set (put it in .env)")
    return OpenAI(base_url=BASE_URL, api_key=key, max_retries=5, timeout=120)


def _strip_private(result):
    if isinstance(result, dict):
        return {k: _strip_private(v) for k, v in result.items() if not k.startswith("_")}
    return result


def run_tool(name, args):
    fn = FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}
    except Exception as e:  # network errors etc. go back to the model
        return {"error": f"{type(e).__name__}: {e}"}


def run_agent(history, model="deepseek-chat", use_tools=True, on_event=None, client=None):
    """Run one user turn.

    history: list of {"role": "user"|"assistant", "content": str} (no system message).
    on_event: optional callback(event_dict) for live UI updates.
    Returns {"answer": str, "trace": [event, ...], "usage": {...}}.
    """
    client = client or make_client()
    emit = on_event or (lambda e: None)
    messages = [{"role": "system", "content": SYSTEM_PROMPT if use_tools else BASELINE_PROMPT}, *history]
    trace = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}

    for step in range(MAX_STEPS):
        kwargs = {"model": model, "messages": messages, "temperature": 0.2}
        if use_tools:
            kwargs["tools"] = SCHEMAS
        t0 = time.time()
        resp = client.chat.completions.create(**kwargs)
        usage["llm_calls"] += 1
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens
        msg = resp.choices[0].message

        if not msg.tool_calls:
            ev = {"type": "answer", "step": step, "seconds": round(time.time() - t0, 1)}
            trace.append(ev)
            emit(ev)
            return {"answer": msg.content or "", "trace": trace, "usage": usage}

        if msg.content:
            ev = {"type": "thought", "step": step, "text": msg.content}
            trace.append(ev)
            emit(ev)
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            t1 = time.time()
            result = run_tool(tc.function.name, args)
            ev = {"type": "tool", "step": step, "name": tc.function.name, "args": args,
                  "result": result, "seconds": round(time.time() - t1, 2)}
            trace.append(ev)
            emit(ev)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(_strip_private(result), ensure_ascii=False)})

    return {"answer": "Stopped: reached the maximum number of tool steps.", "trace": trace, "usage": usage}
