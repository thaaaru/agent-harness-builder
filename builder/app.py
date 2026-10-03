"""Agent Harness Builder: web UI to design, test and export AI agents.

Run:  harness-builder            (or: python -m builder)   then open http://127.0.0.1:8765
Saved agents go in ./agents and test-chat workspaces in ./playground of the directory
you start it from (override with --dir).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError

import harness
from harness import Agent, AgentConfig
from harness.config import NAME_RE
from harness.tools import BUILTIN_TOOLS

from .templates import HOOKS_EXAMPLE, TEMPLATES

STATIC = Path(__file__).parent / "static"
HARNESS_SRC = Path(harness.__file__).parent
APPROVAL_TIMEOUT = 600  # seconds before an unanswered approval counts as "deny"

MODEL_SUGGESTIONS = {
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "claude-fable-5-1"],
    "openai": ["gpt-5", "gpt-5-mini", "llama3.3", "qwen3", "mistral-small"],
}


@dataclass
class Session:
    agent: Agent
    # Approval requests waiting for the browser, keyed by tool call id.
    pending: dict[str, asyncio.Future] = field(default_factory=dict)


def _approver(pending: dict[str, asyncio.Future]):
    async def approve(call: dict) -> bool:
        fut = asyncio.get_running_loop().create_future()
        pending[call["id"]] = fut
        try:
            return await asyncio.wait_for(fut, APPROVAL_TIMEOUT)
        except asyncio.TimeoutError:
            return False
        finally:
            pending.pop(call["id"], None)
    return approve


class ChatIn(BaseModel):
    text: str


class ApprovalIn(BaseModel):
    approve: bool


def _parse(data: dict) -> AgentConfig:
    try:
        return AgentConfig.model_validate(data)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise HTTPException(422, msg)


def create_app(data_dir: str | Path = ".") -> FastAPI:
    data_dir = Path(data_dir).resolve()
    agents_dir = data_dir / "agents"
    playground_dir = data_dir / "playground"
    agents_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Agent Harness Builder")
    sessions: dict[str, Session] = {}

    def agent_path(name: str) -> Path:
        if not NAME_RE.fullmatch(name):
            raise HTTPException(400, "invalid agent name")
        return agents_dir / f"{name}.json"

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    def meta():
        return {
            "builtin_tools": [{"name": t.name, "description": t.description} for t in BUILTIN_TOOLS.values()],
            "models": MODEL_SUGGESTIONS,
            "templates": TEMPLATES,
            "hooks_example": HOOKS_EXAMPLE,
            "defaults": AgentConfig().model_dump(),
        }

    @app.get("/api/agents")
    def list_agents():
        out = []
        for p in sorted(agents_dir.glob("*.json")):
            try:
                cfg = AgentConfig.load(p)
            except (ValidationError, json.JSONDecodeError):
                continue
            out.append({"name": cfg.name, "description": cfg.description, "model": cfg.provider.model})
        return out

    @app.get("/api/agents/{name}")
    def get_agent(name: str):
        path = agent_path(name)
        if not path.exists():
            raise HTTPException(404, "agent not found")
        return AgentConfig.load(path).model_dump()

    @app.put("/api/agents/{name}")
    def save_agent(name: str, data: dict):
        cfg = _parse(data)
        if cfg.name != name:
            raise HTTPException(400, "name in URL and body differ")
        cfg.dump(agent_path(name))
        return {"ok": True}

    @app.delete("/api/agents/{name}")
    def delete_agent(name: str):
        agent_path(name).unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/api/sessions")
    async def create_session(data: dict):
        cfg = _parse(data)
        pending: dict[str, asyncio.Future] = {}
        try:
            agent = Agent(cfg, base_dir=playground_dir / cfg.name, approver=_approver(pending))
        except Exception as e:  # bad custom-tool or hooks code, missing SDK, ...
            raise HTTPException(422, f"{type(e).__name__}: {e}")
        sid = uuid.uuid4().hex
        session = sessions[sid] = Session(agent, pending)
        log = session.agent.log_path
        return {"session_id": sid, "workspace": str(session.agent.ctx.workspace),
                "log_path": str(log) if log else None}

    @app.post("/api/sessions/{sid}/messages")
    async def chat(sid: str, body: ChatIn):
        session = sessions.get(sid)
        if session is None:
            raise HTTPException(404, "session expired; start a new chat")

        async def events():
            try:
                async for ev in session.agent.send(body.text):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/api/sessions/{sid}/approvals/{call_id}")
    def answer_approval(sid: str, call_id: str, body: ApprovalIn):
        session = sessions.get(sid)
        fut = session.pending.get(call_id) if session else None
        if fut is None or fut.done():
            raise HTTPException(404, "no pending approval with that id")
        fut.set_result(body.approve)
        return {"ok": True}

    @app.post("/api/export")
    def export(data: dict):
        cfg = _parse(data)
        return Response(build_export(cfg), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{cfg.name}.zip"'})

    return app


def build_export(cfg: AgentConfig) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        base = cfg.name
        for f in sorted(HARNESS_SRC.glob("*.py")):
            z.write(f, f"{base}/harness/{f.name}")
        z.writestr(f"{base}/agent.json", cfg.model_dump_json(indent=2) + "\n")
        z.writestr(f"{base}/run.py", RUN_PY)
        sdk = "anthropic>=1.0" if cfg.provider.type == "anthropic" else "openai>=1.0"
        z.writestr(f"{base}/requirements.txt", f"{sdk}\npydantic>=2.0\n")
        z.writestr(f"{base}/README.md", _readme(cfg))
    return buf.getvalue()


RUN_PY = '''"""Run this agent:  python run.py   (interactive)  or  python run.py -p "your task" """
from harness.cli import main

if __name__ == "__main__":
    main("agent.json")
'''


def _readme(cfg: AgentConfig) -> str:
    p, h = cfg.provider, cfg.harness
    if p.type == "anthropic":
        auth = f"export {p.api_key_env or 'ANTHROPIC_API_KEY'}=..."
    else:
        auth = f"export {p.api_key_env or 'OPENAI_API_KEY'}=...   # not needed for local servers like Ollama"
    tools = ", ".join(cfg.tools + [t.name for t in cfg.custom_tools]) or "none"
    features = []
    if h.require_approval:
        features.append(f"asks before running: {', '.join(h.require_approval)}")
    if h.token_budget:
        features.append(f"token budget {h.token_budget:,} per message")
    if h.clear_tool_results:
        features.append(f"clears old tool results above {h.clear_trigger_tokens:,} tokens")
    if h.log_dir:
        features.append(f"logs JSONL transcripts to `{h.log_dir}`")
    if h.hooks_code.strip():
        features.append("custom hooks (see `harness.hooks_code` in agent.json)")
    return f"""# {cfg.name}

{cfg.description or "An AI agent built with Agent Harness Builder."}

- Provider: `{p.type}` · Model: `{p.model}`{f" · Base URL: `{p.base_url}`" if p.base_url else ""}
- Tools: {tools}
- File and shell tools are confined to `{cfg.workspace}`
{"".join(f"- Harness: {f}" + chr(10) for f in features)}
## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
{auth}
python run.py                     # interactive chat
python run.py -p "your task"      # one-shot, exits non-zero on error
python run.py --yes               # approve every tool call without asking
python run.py --thinking          # also print the model's thinking
```

## Use from Python

```python
import asyncio
from harness import Agent, AgentConfig, Hooks

class Audit(Hooks):
    def before_tool(self, call):
        if call["name"] == "bash" and "rm " in call["input"].get("command", ""):
            return "rm is not allowed"          # block the call with this reason

async def approve(call):
    return True                             # your own approval logic

agent = Agent(AgentConfig.load("agent.json"), approver=approve, hooks=[Audit()])

async def main():
    async for event in agent.send("Hello"):
        if event["type"] == "text":
            print(event["text"], end="")
    await agent.aclose()

asyncio.run(main())
```

Edit `agent.json` to change the prompt, model, tools or harness settings. Add a provider
by subclassing `harness.providers.Provider` and registering it in `PROVIDERS`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent Harness Builder web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--dir", default=".", help="Where to keep agents/ and playground/ (default: cwd)")
    args = ap.parse_args()

    import uvicorn
    print(f"Agent Harness Builder → http://{args.host}:{args.port}")
    uvicorn.run(create_app(args.dir), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
