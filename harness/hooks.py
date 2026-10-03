"""Hooks: extension points that let you change harness behaviour without editing the loop.

Subclass Hooks and pass instances to Agent(hooks=[...]), or put plain functions with the
same names in the config's `harness.hooks_code`. Every method may be sync or async.

    on_event(event)              observe every event the agent emits
    before_tool(call)            return a string to block the call (the string is the reason)
    after_tool(call, result)     return a string to replace the tool's output
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


class Hooks:
    def on_event(self, event: dict) -> None:
        return None

    def before_tool(self, call: dict) -> str | None:
        return None

    def after_tool(self, call: dict, result: dict) -> str | None:
        return None


class CodeHooks(Hooks):
    """Hooks defined as top-level functions in a Python source string."""

    NAMES = ("on_event", "before_tool", "after_tool")

    def __init__(self, source: str):
        namespace: dict = {}
        exec(compile(source, "<hooks>", "exec"), namespace)
        found = False
        for name in self.NAMES:
            fn = namespace.get(name)
            if callable(fn):
                setattr(self, name, fn)
                found = True
        if not found and source.strip():
            raise ValueError(f"hooks code defines none of: {', '.join(self.NAMES)}")


class JsonlLogger(Hooks):
    """Appends every event, with a timestamp, to a JSONL file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def on_event(self, event: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), **event}, default=str) + "\n")


class HookChain:
    """Runs several Hooks in order. The first block or output override wins."""

    def __init__(self, hooks: list[Hooks]):
        self.hooks = hooks

    async def on_event(self, event: dict) -> None:
        for h in self.hooks:
            await _maybe_await(h.on_event(event))

    async def before_tool(self, call: dict) -> str | None:
        for h in self.hooks:
            reason = await _maybe_await(h.before_tool(call))
            if reason:
                return str(reason)
        return None

    async def after_tool(self, call: dict, result: dict) -> str | None:
        for h in self.hooks:
            override = await _maybe_await(h.after_tool(call, result))
            if override is not None:
                return str(override)
        return None
