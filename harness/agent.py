"""The agent loop: model turn -> run requested tools -> feed results back -> repeat."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from .config import AgentConfig
from .providers import make_provider
from .tools import ToolContext, build_tools


class Agent:
    def __init__(self, config: AgentConfig, base_dir: str | Path = "."):
        self.config = config
        self.tools = build_tools(config)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.provider = make_provider(config.provider)
        workspace = (Path(base_dir) / config.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self.ctx = ToolContext(workspace=workspace)

    async def _run_tool(self, call: dict) -> dict:
        tool = self.tools_by_name.get(call["name"])
        if tool is None:
            output, is_error = f"Unknown tool '{call['name']}'", True
        elif "__invalid_json__" in call["input"]:
            output, is_error = "Tool arguments were not valid JSON; please retry.", True
        else:
            output, is_error = await tool.run(call["input"], self.ctx)
        return {"id": call["id"], "name": call["name"], "output": output, "is_error": is_error}

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def send(self, user_text: str) -> AsyncIterator[dict]:
        """Run one user message to completion, yielding events as they happen."""
        self.provider.add_user(user_text)
        for _ in range(self.config.max_turns):
            turn_end = None
            async for event in self.provider.stream_turn(self.config.system_prompt, self.tools):
                if event["type"] == "turn_end":
                    turn_end = event
                else:
                    yield event
            if turn_end is None or turn_end["error"]:
                yield {"type": "error", "message": (turn_end or {}).get("error") or "provider returned nothing"}
                return
            calls = turn_end["tool_calls"]
            if not calls:
                yield {"type": "done"}
                return
            for call in calls:
                yield {"type": "tool_call", **call}
            # Independent tool calls run concurrently.
            results = await asyncio.gather(*(self._run_tool(c) for c in calls))
            for r in results:
                yield {"type": "tool_result", **r}
            self.provider.add_tool_results(list(results))
        yield {"type": "error", "message": f"Stopped after max_turns={self.config.max_turns}."}
