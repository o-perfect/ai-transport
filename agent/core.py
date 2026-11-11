"""Tool-calling agent loop over the SJTU OpenAI-compatible API."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from .tools import FUNCTIONS, SCHEMAS

load_dotenv()

BASE_URL = "https://models.sjtu.edu.cn/api/v1"
MODELS = ["qwen", "deepseek-chat", "deepseek-reasoner"]  # first is the default; qwen writes ~2x faster
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
- compare_routes takes any world port. If a port name is ambiguous or unknown, use the closest match \
the tool suggests (or 'Name, Country' / UN/LOCODE) and say which port you used. If a chokepoint is not \
supported, say so.
- A message may start with "[Map scenario: closed chokepoints: ...]". Those are the chokepoints the \
user closed on the map: treat them as blocked (pass them as blocked_chokepoints) unless the question \
says otherwise. If one is not in the routing network, say it could not be modelled.
- Reply in the user's language.

Final answer format (markdown), at most about 350 words in total. Be terse: short bullets, no \
introduction or closing summary, and don't repeat numbers from the table in prose.
1. **Situation** - 1-2 bullets: what is disrupted and the current traffic evidence.
2. **Affected links** - 1-3 bullets: which trade lanes are hit and how badly.
3. **Route options** - a table: route (via), distance nm, sea days, total cost USD, extra vs baseline.
4. **Risk assessment** - a rating (Low/Medium/High/Critical) and one sentence why.
5. **Recommendations** - up to 3 short-term and 2 medium-term actions, one line each.
6. **Assumptions & sources** - one line: key parameters and sources (IMF PortWatch, searoute \
maritime network).
For a simple question that is not a disruption scenario (e.g. only current traffic), skip the \
template and answer in a few lines.
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


def _stream_completion(client, kwargs, emit):
    """Streamed chat completion. Emits {"type": "delta", "text"} for each piece of
    content so UIs can show the answer as it is written.
    Returns (content, tool_calls as [{"id", "name", "arguments"}], usage or None)."""
    stream = client.chat.completions.create(**kwargs, stream=True,
                                            stream_options={"include_usage": True})
    content, calls, usage = [], {}, None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content.append(delta.content)
            emit({"type": "delta", "text": delta.content})
        for tc in delta.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            c["id"] = tc.id or c["id"]
            if tc.function and tc.function.name:
                c["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                c["arguments"] += tc.function.arguments
    return "".join(content), [calls[i] for i in sorted(calls)], usage


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


def run_agent(history, model=MODELS[0], use_tools=True, on_event=None, client=None):
    """Run one user turn.

    history: list of {"role": "user"|"assistant", "content": str} (no system message).
    on_event: optional callback(event_dict) for live UI updates. "delta" events
        stream content as it is generated; if the step then calls tools, a
        "thought" event follows with the same text, so drop the streamed text.
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
        content, tool_calls, u = _stream_completion(client, kwargs, emit)
        usage["llm_calls"] += 1
        if u:
            usage["prompt_tokens"] += u.prompt_tokens
            usage["completion_tokens"] += u.completion_tokens

        if not tool_calls:
            ev = {"type": "answer", "step": step, "seconds": round(time.time() - t0, 1)}
            trace.append(ev)
            emit(ev)
            return {"answer": content, "trace": trace, "usage": usage}

        if content:
            ev = {"type": "thought", "step": step, "text": content}
            trace.append(ev)
            emit(ev)
        messages.append({
            "role": "assistant", "content": content,
            "tool_calls": [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                           for tc in tool_calls],
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            t1 = time.time()
            result = run_tool(tc["name"], args)
            ev = {"type": "tool", "step": step, "name": tc["name"], "args": args,
                  "result": result, "seconds": round(time.time() - t1, 2)}
            trace.append(ev)
            emit(ev)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(_strip_private(result), ensure_ascii=False)})

    return {"answer": "Stopped: reached the maximum number of tool steps.", "trace": trace, "usage": usage}
