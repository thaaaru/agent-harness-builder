"""Tool registry: built-in tools plus user-defined Python tools."""

from __future__ import annotations

import asyncio
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import AgentConfig, CustomTool

MAX_OUTPUT = 20_000


@dataclass
class ToolContext:
    workspace: Path


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[dict, ToolContext], str]

    async def run(self, args: dict, ctx: ToolContext) -> tuple[str, bool]:
        """Returns (output, is_error). Never raises."""
        problem = validate_args(self.parameters, args)
        if problem:
            return f"Invalid arguments: {problem}", True
        try:
            out = await asyncio.to_thread(self.fn, args, ctx)
        except Exception as e:  # tool errors go back to the model, not up the stack
            return f"{type(e).__name__}: {e}", True
        out = out if isinstance(out, str) else str(out)
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n... [truncated {len(out) - MAX_OUTPUT} chars]"
        return out, False


_JSON_TYPES = {
    "string": str, "integer": int, "number": (int, float),
    "boolean": bool, "object": dict, "array": list,
}


def validate_args(schema: dict, args: object) -> str | None:
    """Minimal JSON-schema check: object shape, required keys, primitive types."""
    if not isinstance(args, dict):
        return "expected a JSON object"
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required field '{key}'"
    for key, spec in schema.get("properties", {}).items():
        expected = _JSON_TYPES.get(spec.get("type", ""))
        if key in args and expected and not isinstance(args[key], expected):
            return f"field '{key}' should be {spec['type']}"
        if key in args and spec.get("type") in ("integer", "number") and isinstance(args[key], bool):
            return f"field '{key}' should be {spec['type']}"
    return None


def _resolve(ctx: ToolContext, path: str) -> Path:
    """Resolve a model-supplied path and refuse anything outside the workspace."""
    target = (ctx.workspace / path).resolve()
    if not target.is_relative_to(ctx.workspace):
        raise PermissionError(f"path '{path}' is outside the workspace")
    return target


# ---- built-in tools ---------------------------------------------------------

def _read_file(args: dict, ctx: ToolContext) -> str:
    return _resolve(ctx, args["path"]).read_text(errors="replace")


def _write_file(args: dict, ctx: ToolContext) -> str:
    target = _resolve(ctx, args["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(args["content"])
    return f"Wrote {len(args['content'])} chars to {args['path']}"


def _edit_file(args: dict, ctx: ToolContext) -> str:
    target = _resolve(ctx, args["path"])
    text = target.read_text()
    count = text.count(args["old_text"])
    if count != 1:
        raise ValueError(f"old_text must match exactly once (found {count} matches)")
    target.write_text(text.replace(args["old_text"], args["new_text"]))
    return f"Edited {args['path']}"


def _list_files(args: dict, ctx: ToolContext) -> str:
    root = _resolve(ctx, args.get("path", "."))
    pattern = args.get("pattern", "*")
    paths = sorted(p for p in root.rglob(pattern) if p.is_file())[:500]
    return "\n".join(str(p.relative_to(ctx.workspace)) for p in paths) or "(no files)"


def _bash(args: dict, ctx: ToolContext) -> str:
    proc = subprocess.run(
        args["command"], shell=True, cwd=ctx.workspace, capture_output=True,
        text=True, timeout=args.get("timeout", 60),
    )
    return f"exit code {proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"


def _http_get(args: dict, ctx: ToolContext) -> str:
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        raise ValueError("only http(s) URLs are allowed")
    req = urllib.request.Request(url, headers={"User-Agent": "agent-harness/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return f"HTTP {resp.status}\n" + resp.read(MAX_OUTPUT * 2).decode(errors="replace")


def _obj(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


BUILTIN_TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("read_file", "Read a UTF-8 text file from the workspace.",
         _obj({"path": {"type": "string", "description": "Path relative to the workspace"}}, ["path"]),
         _read_file),
    Tool("write_file", "Create or overwrite a text file in the workspace.",
         _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
         _write_file),
    Tool("edit_file", "Replace one exact, unique occurrence of old_text with new_text in a workspace file.",
         _obj({"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
              ["path", "old_text", "new_text"]),
         _edit_file),
    Tool("list_files", "List files in the workspace recursively, optionally filtered by a glob pattern.",
         _obj({"path": {"type": "string", "description": "Directory, default '.'"},
               "pattern": {"type": "string", "description": "Glob such as '*.py', default '*'"}}, []),
         _list_files),
    Tool("bash", "Run a shell command in the workspace directory and return stdout, stderr and exit code. "
                 "Not sandboxed: it runs with your user's permissions.",
         _obj({"command": {"type": "string"},
               "timeout": {"type": "integer", "description": "Seconds, default 60"}}, ["command"]),
         _bash),
    Tool("http_get", "Fetch a URL with an HTTP GET request and return the response body as text.",
         _obj({"url": {"type": "string"}}, ["url"]),
         _http_get),
]}


def compile_custom_tool(spec: CustomTool) -> Tool:
    """Turn user-supplied Python source into a Tool. The source must define run(args)."""
    namespace: dict = {}
    exec(compile(spec.code, f"<tool {spec.name}>", "exec"), namespace)
    run = namespace.get("run")
    if not callable(run):
        raise ValueError(f"custom tool '{spec.name}' must define a function run(args)")
    params = spec.parameters or {"type": "object", "properties": {}}
    return Tool(spec.name, spec.description, params, lambda args, ctx: run(args))


def build_tools(config: AgentConfig) -> list[Tool]:
    tools = []
    for name in config.tools:
        if name not in BUILTIN_TOOLS:
            raise ValueError(f"unknown built-in tool '{name}'")
        tools.append(BUILTIN_TOOLS[name])
    tools.extend(compile_custom_tool(t) for t in config.custom_tools)
    names = [t.name for t in tools]
    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique")
    return tools
