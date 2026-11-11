"""Streamlit chat page for the agent:  streamlit run app_chat.py

Can be merged into the main app as a page; everything lives in chat_page().
"""

import datetime as dt
import json

import pandas as pd
import streamlit as st

from agent.core import MODELS, run_agent

EXAMPLES = [
    "The Suez Canal is blocked. We ship containers Shanghai to Rotterdam on a 14,000 TEU ship. Options and cost?",
    "Houthi attacks make Bab el-Mandeb high-risk but open. Is it cheaper to pay war-risk insurance or go via the Cape for Singapore to Rotterdam?",
    "If the Strait of Hormuz closes, which trade links are affected and how badly?",
    "What is the current traffic situation at the Panama Canal compared with last year?",
]


def render_trace(trace):
    tools = [e for e in trace if e["type"] == "tool"]
    if not tools:
        return
    with st.expander(f"Tool calls ({len(tools)})", expanded=False):
        for e in trace:
            if e["type"] == "thought":
                st.caption(f"💭 {e['text']}")
            elif e["type"] == "tool":
                ok = "error" not in e["result"]
                st.markdown(f"{'✅' if ok else '❌'} **{e['name']}** · step {e['step']} · {e['seconds']}s")
                st.code(json.dumps(e["args"], ensure_ascii=False, indent=1), language="json")
                with st.popover("Result"):
                    st.json(e["result"], expanded=False)
    render_charts(tools)


def render_charts(tools):
    for e in tools:
        r = e["result"]
        if "error" in r:
            continue
        if e["name"] == "get_chokepoint_traffic" and "_series" in r:
            s = r["_series"]
            df = pd.DataFrame({"date": pd.to_datetime(s["date"]), "transits/day": s["n_total"]})
            ya = s.get("year_ago_n_total") or []
            if len(ya) == len(df):
                df["same days, year before"] = ya
            st.markdown(f"**{r['chokepoint']}: daily transits** ({r['window']})")
            st.line_chart(df.set_index("date"))
        elif e["name"] == "compare_routes" and isinstance(r.get("alternatives"), list):
            rows = [("Baseline: " + " / ".join(r["baseline_route"]["chokepoints"]), r["baseline_route"])]
            rows += [("Alt: " + " / ".join(a["chokepoints"]), a) for a in r["alternatives"]]
            df = pd.DataFrame([{"route": name, "distance_nm": x["distance_nm"], "sea_days": x["sea_days"],
                                **{f"{k}_usd": v for k, v in x["cost_usd"].items()}, "co2_t": x["co2_t"]}
                               for name, x in rows])
            st.markdown(f"**{r['origin']} → {r['destination']}**, blocked: {', '.join(r['blocked']) or 'none'}")
            st.dataframe(df, hide_index=True)
            st.bar_chart(df.set_index("route")[["fuel_usd", "charter_time_usd", "canal_tolls_usd",
                                                "war_risk_insurance_usd"]], horizontal=True)
        elif e["name"] == "find_affected_routes" and r.get("affected"):
            df = pd.DataFrame(r["affected"])
            st.markdown(f"**Links through {r['chokepoint']}**: {r['links_affected']} of {r['links_total']}")
            st.dataframe(df, hide_index=True)


def report_markdown(messages):
    lines = [f"# Corridor risk report\n\n_Generated {dt.datetime.now():%Y-%m-%d %H:%M}_\n"]
    for m in messages:
        if m["role"] == "user":
            lines.append(f"## Question\n\n{m['content']}\n")
        else:
            lines.append(f"## Answer\n\n{m['content']}\n")
            tools = [e for e in m.get("trace", []) if e["type"] == "tool"]
            if tools:
                lines.append("### Tool calls\n")
                lines += [f"- `{e['name']}({json.dumps(e['args'], ensure_ascii=False)})`" for e in tools]
                lines.append("")
    return "\n".join(lines)


def chat_page():
    st.title("🚢 Corridor Risk Agent")
    ss = st.session_state
    ss.setdefault("messages", [])

    with st.sidebar:
        model = st.selectbox("Model", MODELS)
        use_tools = st.toggle("Use tools (off = baseline LLM)", value=True)
        st.caption("Example scenarios")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                ss.pending = ex
        if ss.messages:
            st.download_button("Export report (.md)", report_markdown(ss.messages),
                               file_name="corridor_report.md", use_container_width=True)
            st.download_button("Export trace (.json)",
                               json.dumps(ss.messages, ensure_ascii=False, indent=1, default=str),
                               file_name="corridor_trace.json", use_container_width=True)
            if st.button("Clear chat", use_container_width=True):
                ss.messages = []
                st.rerun()

    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_trace(m.get("trace", []))
                if m.get("usage"):
                    st.caption(f"{m['model']} · tools {'on' if m['tools'] else 'off'} · {m['usage']}")

    prompt = st.chat_input("Describe a disruption scenario or ask a question…") or ss.pop("pending", None)
    if not prompt:
        return
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)

        def on_event(e):
            if e["type"] == "tool":
                status.write(f"🔧 `{e['name']}` {json.dumps(e['args'], ensure_ascii=False)}")
            elif e["type"] == "thought":
                status.write(f"💭 {e['text'][:300]}")

        history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
        try:
            out = run_agent(history, model=model, use_tools=use_tools, on_event=on_event)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            ss.messages.pop()
            return
        status.update(label=f"Done · {out['usage']['llm_calls']} LLM calls", state="complete", expanded=False)
        st.markdown(out["answer"])
        render_trace(out["trace"])
    ss.messages.append({"role": "assistant", "content": out["answer"], "trace": out["trace"],
                        "usage": out["usage"], "model": model, "tools": use_tools})


if __name__ == "__main__":
    st.set_page_config(page_title="Corridor Risk Agent", page_icon="🚢", layout="wide")
    chat_page()
