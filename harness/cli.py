"""Terminal runner for an agent config: `python run.py` or `python run.py -p "task"`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .agent import Agent
from .config import AgentConfig

DIM, CYAN, RED, RESET = "\033[2m", "\033[36m", "\033[31m", "\033[0m"


async def _run_one(agent: Agent, text: str, show_thinking: bool) -> bool:
    mode = None
    ok = True
    async for ev in agent.send(text):
        t = ev["type"]
        if t in ("text", "thinking") and t != mode:
            if t == "thinking" and not show_thinking:
                continue
            print(f"\n{DIM}[thinking]{RESET} " if t == "thinking" else "\n", end="")
            mode = t
        if t == "text":
            print(ev["text"], end="", flush=True)
        elif t == "thinking" and show_thinking:
            print(f"{DIM}{ev['text']}{RESET}", end="", flush=True)
        elif t == "tool_call":
            mode = None
            print(f"\n{CYAN}→ {ev['name']}({json.dumps(ev['input'])[:200]}){RESET}")
        elif t == "tool_result":
            colour = RED if ev["is_error"] else DIM
            preview = ev["output"].strip().replace("\n", " ")[:200]
            print(f"{colour}  ← {preview}{RESET}")
        elif t == "error":
            print(f"\n{RED}error: {ev['message']}{RESET}")
            ok = False
    print()
    return ok


def main(config_path: str = "agent.json") -> None:
    ap = argparse.ArgumentParser(description="Run an agent built with Agent Harness Builder")
    ap.add_argument("-c", "--config", default=config_path)
    ap.add_argument("-p", "--prompt", help="Run a single task and exit")
    ap.add_argument("--thinking", action="store_true", help="Show the model's thinking")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    agent = Agent(AgentConfig.load(cfg_path), base_dir=cfg_path.parent)
    print(f"{CYAN}{agent.config.name}{RESET} · {agent.config.provider.model} · "
          f"tools: {', '.join(t.name for t in agent.tools) or 'none'}")

    async def run() -> bool:
        try:
            if args.prompt:
                return await _run_one(agent, args.prompt, args.thinking)
            while True:
                try:
                    text = await asyncio.to_thread(input, "\n> ")
                except (EOFError, KeyboardInterrupt):
                    return True
                if text.strip() in ("exit", "quit"):
                    return True
                if text.strip():
                    await _run_one(agent, text, args.thinking)
        finally:
            await agent.aclose()

    sys.exit(0 if asyncio.run(run()) else 1)


if __name__ == "__main__":
    main()
