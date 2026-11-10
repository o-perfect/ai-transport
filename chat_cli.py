"""Terminal chat with the agent.

    python chat_cli.py                      # interactive, deepseek-chat
    python chat_cli.py -m qwen --no-tools   # baseline: plain LLM, no tools
    python chat_cli.py -q "Suez is closed, what about Shanghai-Rotterdam?"
"""

import argparse
import json

from agent.core import MODELS, run_agent


def show(ev):
    if ev["type"] == "tool":
        args = json.dumps(ev["args"], ensure_ascii=False)
        status = "ERROR " + ev["result"]["error"] if "error" in ev["result"] else "ok"
        print(f"  [tool] {ev['name']}({args}) -> {status} ({ev['seconds']}s)")
    elif ev["type"] == "thought":
        print(f"  [thinking] {ev['text'][:200]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="deepseek-chat", choices=MODELS)
    p.add_argument("--no-tools", action="store_true", help="baseline mode without tools")
    p.add_argument("-q", "--question", help="ask one question and exit")
    a = p.parse_args()

    history = []
    while True:
        q = a.question or input("\nyou> ").strip()
        if not q or q in ("exit", "quit"):
            break
        history.append({"role": "user", "content": q})
        out = run_agent(history, model=a.model, use_tools=not a.no_tools, on_event=show)
        history.append({"role": "assistant", "content": out["answer"]})
        print("\nagent>\n" + out["answer"])
        print(f"\n  [usage] {out['usage']}")
        if a.question:
            break


if __name__ == "__main__":
    main()
