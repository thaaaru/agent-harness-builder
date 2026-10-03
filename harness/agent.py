"""The agent loop: model turn -> run requested tools -> feed results back -> repeat.

Events yielded by Agent.send():
    text, thinking          streamed model output        {"text": str}
    usage                   tokens for one model call    {"input_tokens", "output_tokens", "cache_read_tokens"}
    tool_call               the model wants a tool       {"id", "name", "input"}
    approval_request        waiting for a human          {"id", "name", "input"}
    approval_result         the human's answer           {"id", "approved"}
    tool_result             a tool finished              {"id", "name", "output", "is_error"}
    error                   the run stopped early        {"message"}
    done                    the model finished normally
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from .config import AgentConfig
from .hooks import CodeHooks, HookChain, Hooks, JsonlLogger
from .providers import make_provider
from .tools import ToolContext, build_tools

# Called with a tool call; returns True to allow it.
Approver = Callable[[dict], Awaitable[bool]]


class Agent:
    def __init__(self, config: AgentConfig, base_dir: str | Path = ".",
                 approver: Approver | None = None, hooks: list[Hooks] | None = None):
        self.config = config
        h = config.harness
        self.tools = build_tools(config)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.provider = make_provider(config.provider, h)
        self.approver = approver
        base = Path(base_dir)
        workspace = (base / config.workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self.ctx = ToolContext(workspace=workspace)

        chain: list[Hooks] = list(hooks or [])
        if h.hooks_code.strip():
            chain.append(CodeHooks(h.hooks_code))
        if h.log_dir:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_path = (base / h.log_dir / f"{config.name}-{stamp}-{uuid.uuid4().hex[:6]}.jsonl").resolve()
            chain.append(JsonlLogger(self.log_path))
        else:
            self.log_path = None
        self.hooks = HookChain(chain)

    def needs_approval(self, tool_name: str) -> bool:
        required = self.config.harness.require_approval
        return "*" in required or tool_name in required

    async def aclose(self) -> None:
        await self.provider.aclose()

    async def send(self, user_text: str) -> AsyncIterator[dict]:
        """Run one user message to completion, yielding events as they happen."""
        async for event in self._send(user_text):
            await self.hooks.on_event(event)
            yield event

    async def _execute(self, call: dict) -> dict:
        tool = self.tools_by_name.get(call["name"])
        if tool is None:
            output, is_error = f"Unknown tool '{call['name']}'", True
        elif isinstance(call["input"], dict) and "__invalid_json__" in call["input"]:
            output, is_error = "Tool arguments were not valid JSON; please retry.", True
        else:
            timeout = self.config.harness.tool_timeout
            try:
                output, is_error = await asyncio.wait_for(tool.run(call["input"], self.ctx), timeout)
            except asyncio.TimeoutError:
                output, is_error = f"Tool timed out after {timeout}s", True
        result = {"id": call["id"], "name": call["name"], "output": output, "is_error": is_error}
        override = await self.hooks.after_tool(call, result)
        if override is not None:
            result["output"] = override
        return result

    async def _send(self, user_text: str) -> AsyncIterator[dict]:
        self.provider.add_user(user_text)
        budget = self.config.harness.token_budget
        used = 0
        for _ in range(self.config.max_turns):
            turn_end = None
            async for event in self.provider.stream_turn(self.config.system_prompt, self.tools):
                if event["type"] == "turn_end":
                    turn_end = event
                    continue
                if event["type"] == "usage":
                    used += event["input_tokens"] + event["output_tokens"]
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

            if budget and used >= budget:
                # Every tool call must still get a result, or the next request is invalid.
                self.provider.add_tool_results([
                    {"id": c["id"], "name": c["name"], "output": "Not run: token budget exhausted.", "is_error": True}
                    for c in calls])
                yield {"type": "error", "message": f"Token budget of {budget:,} reached ({used:,} used)."}
                return

            blocked: dict[str, str] = {}
            for call in calls:
                reason = await self.hooks.before_tool(call)
                if reason:
                    blocked[call["id"]] = f"Blocked: {reason}"
                elif self.needs_approval(call["name"]):
                    yield {"type": "approval_request", **call}
                    approved = bool(self.approver and await self.approver(call))
                    yield {"type": "approval_result", "id": call["id"], "approved": approved}
                    if not approved:
                        blocked[call["id"]] = "The user denied this tool call."

            async def run(call: dict) -> dict:
                if call["id"] in blocked:
                    return {"id": call["id"], "name": call["name"], "output": blocked[call["id"]], "is_error": True}
                return await self._execute(call)

            # Independent tool calls run concurrently.
            results = list(await asyncio.gather(*(run(c) for c in calls)))
            for r in results:
                yield {"type": "tool_result", **r}
            self.provider.add_tool_results(results)
        yield {"type": "error", "message": f"Stopped after max_turns={self.config.max_turns}."}
