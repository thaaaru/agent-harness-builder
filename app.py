"""Agent Harness Builder: web UI to design, test and export AI agents.

Run:  .venv/bin/python app.py   then open http://127.0.0.1:8765
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError

from harness import Agent, AgentConfig
from harness.tools import BUILTIN_TOOLS
from templates import TEMPLATES

ROOT = Path(__file__).parent
AGENTS_DIR = ROOT / "agents"
PLAYGROUND_DIR = ROOT / "playground"  # workspaces for test chats
AGENTS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Agent Harness Builder")
sessions: dict[str, Agent] = {}

MODEL_SUGGESTIONS = {
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-4-5", "claude-fable-5-1"],
    "openai": ["gpt-5", "gpt-5-mini", "llama3.3", "qwen3", "mistral-small"],
}


def _parse(data: dict) -> AgentConfig:
    try:
        return AgentConfig.model_validate(data)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        raise HTTPException(422, msg)


def _agent_path(name: str) -> Path:
    path = AGENTS_DIR / f"{name}.json"
    if path.parent != AGENTS_DIR or not name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "invalid agent name")
    return path


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/meta")
def meta():
    return {
        "builtin_tools": [{"name": t.name, "description": t.description} for t in BUILTIN_TOOLS.values()],
        "models": MODEL_SUGGESTIONS,
        "templates": TEMPLATES,
        "defaults": AgentConfig().model_dump(),
    }


@app.get("/api/agents")
def list_agents():
    out = []
    for p in sorted(AGENTS_DIR.glob("*.json")):
        try:
            cfg = AgentConfig.load(p)
            out.append({"name": cfg.name, "description": cfg.description, "model": cfg.provider.model})
        except (ValidationError, json.JSONDecodeError):
            continue
    return out


@app.get("/api/agents/{name}")
def get_agent(name: str):
    path = _agent_path(name)
    if not path.exists():
        raise HTTPException(404, "agent not found")
    return AgentConfig.load(path).model_dump()


@app.put("/api/agents/{name}")
def save_agent(name: str, data: dict):
    cfg = _parse(data)
    if cfg.name != name:
        raise HTTPException(400, "name in URL and body differ")
    cfg.dump(_agent_path(name))
    return {"ok": True}


@app.delete("/api/agents/{name}")
def delete_agent(name: str):
    _agent_path(name).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/sessions")
def create_session(data: dict):
    cfg = _parse(data)
    try:
        agent = Agent(cfg, base_dir=PLAYGROUND_DIR / cfg.name)
    except Exception as e:  # bad custom-tool code, missing SDK, etc.
        raise HTTPException(422, f"{type(e).__name__}: {e}")
    sid = uuid.uuid4().hex
    sessions[sid] = agent
    return {"session_id": sid, "workspace": str(agent.ctx.workspace)}


class ChatIn(BaseModel):
    text: str


@app.post("/api/sessions/{sid}/messages")
async def chat(sid: str, body: ChatIn):
    agent = sessions.get(sid)
    if agent is None:
        raise HTTPException(404, "session expired; start a new chat")

    async def events():
        try:
            async for ev in agent.send(body.text):
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.post("/api/export")
def export(data: dict):
    cfg = _parse(data)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        base = cfg.name
        for f in sorted((ROOT / "harness").glob("*.py")):
            z.write(f, f"{base}/harness/{f.name}")
        z.writestr(f"{base}/agent.json", cfg.model_dump_json(indent=2) + "\n")
        z.writestr(f"{base}/run.py", RUN_PY)
        sdk = "anthropic>=1.0" if cfg.provider.type == "anthropic" else "openai>=1.0"
        z.writestr(f"{base}/requirements.txt", f"{sdk}\npydantic>=2.0\n")
        z.writestr(f"{base}/README.md", _readme(cfg))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{cfg.name}.zip"'})


RUN_PY = '''"""Run this agent:  python run.py   (interactive)  or  python run.py -p "your task" """
from harness.cli import main

if __name__ == "__main__":
    main("agent.json")
'''


def _readme(cfg: AgentConfig) -> str:
    p = cfg.provider
    if p.type == "anthropic":
        auth = f"export {p.api_key_env or 'ANTHROPIC_API_KEY'}=..."
    else:
        auth = f"export {p.api_key_env or 'OPENAI_API_KEY'}=...   # not needed for local servers like Ollama"
    tools = ", ".join(cfg.tools + [t.name for t in cfg.custom_tools]) or "none"
    return f"""# {cfg.name}

{cfg.description or "An AI agent built with Agent Harness Builder."}

- Provider: `{p.type}` · Model: `{p.model}`{f" · Base URL: `{p.base_url}`" if p.base_url else ""}
- Tools: {tools}
- File and shell tools are confined to `{cfg.workspace}`

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
{auth}
python run.py                     # interactive chat
python run.py -p "your task"      # one-shot, exits non-zero on error
python run.py --thinking          # also print the model's thinking
```

## Use from Python

```python
import asyncio
from harness import Agent, AgentConfig

agent = Agent(AgentConfig.load("agent.json"))

async def main():
    async for event in agent.send("Hello"):
        if event["type"] == "text":
            print(event["text"], end="")

asyncio.run(main())
```

Edit `agent.json` to change the prompt, model or tools. Add a provider by
subclassing `harness.providers.Provider` and registering it in `PROVIDERS`.
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
