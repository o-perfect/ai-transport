"""Streamlit chat for the agent. app.py shows it beside the map (chat_panel());
it also runs on its own without the map: streamlit run app_chat.py

The routes from the agent's latest compare_routes call go to the map through
st.session_state (see shared_state.py).
"""

import datetime as dt
import json
import time

import pandas as pd
import streamlit as st

from agent.core import MODELS, run_agent
from shared_state import AGENT_BUSY, AGENT_ROUTES
from ui import components as ui
from ui import speech

TYPING = '<div class="cra-typing"><span></span><span></span><span></span></div>'

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
    with st.expander(f"Tool calls ({len(tools)})", expanded=False, icon=":material/build:"):
        for e in trace:
            if e["type"] == "thought":
                st.caption(f":material/psychology: {e['text']}")
            elif e["type"] == "tool":
                ok = "error" not in e["result"]
                icon = ":green[:material/check_circle:]" if ok else ":red[:material/error:]"
                st.markdown(f"{icon} **{e['name']}** (step {e['step']}, {e['seconds']} s)")
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
                                                "war_risk_insurance_usd", "eu_ets_usd"]], horizontal=True)
        elif e["name"] == "find_affected_routes" and r.get("affected"):
            df = pd.DataFrame(r.get("_all_affected", r["affected"]))
            st.markdown(f"**Links through {r['chokepoint']}**: {r['links_affected']} of {r['links_total']}")
            st.dataframe(df, hide_index=True)
        elif e["name"] == "get_news" and r.get("_items"):
            st.markdown(f"**News: {r['query']}** (last {r['days']} days)")
            st.dataframe(pd.DataFrame(r["_items"]), hide_index=True)


def latest_geometry(trace):
    """GeoJSON of the last compare_routes call in an agent turn, if any."""
    for e in reversed(trace):
        if e["type"] == "tool" and e["name"] == "compare_routes" and "_geometry" in e["result"]:
            return e["result"]["_geometry"]
    return None


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


def render_message(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("voice"):
            v = m["voice"]
            st.caption(f":material/mic: Spoken, transcribed locally ({v.get('language', '?')}, {v.get('seconds', '?')} s)")
        if m["role"] == "assistant":
            render_trace(m.get("trace", []))
            geometry = latest_geometry(m.get("trace", []))
            if geometry is not None and geometry is st.session_state.get(AGENT_ROUTES):
                st.caption(f":material/map: {len(geometry['features'])} route(s) drawn on the map.")
            if m.get("usage"):
                u = m["usage"]
                tokens = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                st.caption(f"{m['model']}, tools {'on' if m['tools'] else 'off'}, "
                           f"{u.get('llm_calls', 0)} LLM calls, {tokens:,} tokens")


def _start_turn(example=None):
    """Submit callback: runs before the page redraws, so controls can be
    disabled for the whole turn (see AGENT_BUSY)."""
    st.session_state[AGENT_BUSY] = True
    if example:
        st.session_state.pending = example


def chat_panel(height=640):
    """The chat inside the current container: settings, messages and input.

    Returns a function that runs the submitted turn, or None. Call it after the
    rest of the page is drawn so the page stays visible while the agent works;
    it reruns the app when done so the map picks up the agent's routes."""
    ss = st.session_state
    ss.setdefault("messages", [])
    busy = ss.get(AGENT_BUSY, False)

    card = st.container(key="chat_card")
    with card:
        with st.container(horizontal=True, vertical_alignment="center", key="chat_head"):
            ui.chat_title()
            with st.popover("Settings", icon=":material/tune:", help="Agent settings, export, clear"):
                model = st.selectbox("Model", MODELS, key="agent_model", disabled=busy)
                use_tools = st.toggle("Use tools (off = baseline LLM)", value=True, key="agent_tools",
                                      disabled=busy)
                if ss.messages:
                    st.download_button("Export report (.md)", report_markdown(ss.messages),
                                       file_name="corridor_report.md", width="stretch", disabled=busy,
                                       icon=":material/description:")
                    st.download_button("Export trace (.json)",
                                       json.dumps(ss.messages, ensure_ascii=False, indent=1, default=str),
                                       file_name="corridor_trace.json", width="stretch", disabled=busy,
                                       icon=":material/data_object:")
                    if st.button("Clear chat", width="stretch", disabled=busy, icon=":material/delete_sweep:"):
                        ss.messages = []
                        st.rerun()

        box = st.container(height=height, key="chat_messages")
        with box:
            intro = st.empty()
            if not ss.messages:
                with intro.container(key="examples"):
                    st.html('<div class="cra-intro"><b>Ask about any corridor disruption</b> '
                            "(a closed canal, a high-risk strait, a sanction) or start from an example:</div>")
                    cols = st.columns(2, gap="small")
                    for i, ex in enumerate(EXAMPLES):
                        cols[i % 2].button(ui.example_label(i), key=f"agent_example_{i}", width="stretch",
                                           icon=ui.example_icon(i), help=ex, disabled=busy,
                                           on_click=_start_turn, args=(ex,))
            for m in ss.messages:
                render_message(m)
            if "agent_error" in ss:  # from a failed turn, kept across the rerun that unfreezes the page
                st.error(ss.pop("agent_error"), icon=":material/error:")

        speech.warm_up()  # load Whisper in the background (no-op if not installed)
        raw = st.chat_input("Describe a disruption or ask a question…", submit_mode="disable",
                            on_submit=_start_turn, accept_audio=speech.AVAILABLE)
    # str without audio support; ChatInputValue (.text, .audio) with it
    prompt = (raw if isinstance(raw, str) else getattr(raw, "text", "")) or ss.pop("pending", None)
    audio = getattr(raw, "audio", None)
    if not prompt and audio is None:
        if busy:  # a turn was cancelled (e.g. page reload mid-run): unfreeze
            ss[AGENT_BUSY] = False
            st.rerun()
        return None
    intro.empty()
    if audio is not None:
        return lambda: _run_voice_turn(box, audio.getvalue(), prompt, model, use_tools)
    ss.messages.append({"role": "user", "content": prompt})
    return lambda: _run_turn(box, model, use_tools)


def _run_voice_turn(box, wav, typed, model, use_tools):
    """Transcribe the recording locally, then run the turn as if it was typed."""
    ss = st.session_state
    with box:
        slot = st.empty()
        with slot.container():
            with st.chat_message("user"):
                st.html(TYPING)
                st.caption(":material/mic: Transcribing your recording…")
        try:
            heard = speech.transcribe(wav)
        except Exception as e:  # e.g. the model couldn't be downloaded
            heard = {"text": "", "error": f"Speech recognition failed: {type(e).__name__}: {e}"}
        slot.empty()
    text = " ".join(x for x in (typed, heard["text"]) if x)
    if not text:
        ss.agent_error = heard.get("error") or ("No speech recognised in the recording. "
                                                "Please try again, or type your question.")
        ss[AGENT_BUSY] = False
        st.rerun()
    ss.messages.append({"role": "user", "content": text,
                        "voice": {"language": heard.get("language"), "seconds": heard.get("seconds")}})
    _run_turn(box, model, use_tools)


def _run_turn(box, model, use_tools):
    ss = st.session_state
    with box:
        render_message(ss.messages[-1])
        with st.chat_message("assistant"):
            status = st.status("Analyzing…", expanded=True)
            answer_box = st.empty()
            answer_box.html(TYPING)
            streamed = {"text": "", "shown": 0.0}

            def on_event(e):
                if e["type"] == "delta":
                    if not streamed["text"]:
                        status.update(label="Writing the assessment…")
                    streamed["text"] += e["text"]
                    if time.time() - streamed["shown"] > 0.1:  # redraw at most 10x per second
                        answer_box.markdown(streamed["text"] + " ▌")
                        streamed["shown"] = time.time()
                elif e["type"] == "thought":  # the streamed text was a note before tool calls
                    streamed["text"] = ""
                    answer_box.html(TYPING)
                    status.write(f":material/psychology: {e['text'][:300]}")
                    status.update(label="Analyzing…")
                elif e["type"] == "retry":
                    status.update(label=f"Retrying: {e['reason']} (waiting {e['wait']:.0f}s)…")
                elif e["type"] == "tool":
                    status.write(f":material/build: `{e['name']}` {json.dumps(e['args'], ensure_ascii=False)}")
                    status.update(label="Analyzing…")

            history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
            try:
                out = run_agent(history, model=model, use_tools=use_tools, on_event=on_event)
            except Exception as e:
                ss.messages.pop()
                ss.agent_error = f"{type(e).__name__}: {e}"
                ss[AGENT_BUSY] = False
                st.rerun()
    geometry = latest_geometry(out["trace"])
    if geometry:
        ss[AGENT_ROUTES] = geometry
    ss.messages.append({"role": "assistant", "content": out["answer"], "trace": out["trace"],
                        "usage": out["usage"], "model": model, "tools": use_tools})
    ss[AGENT_BUSY] = False
    st.rerun()


def chat_page():
    """The chat on its own page, without the map."""
    ui.inject_css()
    ui.header(busy=st.session_state.get(AGENT_BUSY, False))
    run_turn = chat_panel(height=640)
    if run_turn:
        run_turn()


if __name__ == "__main__":
    st.set_page_config(page_title="Corridor Risk Agent", page_icon="🛰️", layout="wide")
    chat_page()
