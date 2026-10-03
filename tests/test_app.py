"""The builder web app: config CRUD, playground sessions with approvals, and export."""

import io
import json
import subprocess
import sys
import threading
import urllib.request
import zipfile

import pytest
from conftest import ServerThread, openai_turn
from fastapi.testclient import TestClient

from builder.app import build_export, create_app
from harness import AgentConfig


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_meta(client):
    meta = client.get("/api/meta").json()
    assert {t["name"] for t in meta["builtin_tools"]} >= {"read_file", "bash"}
    assert meta["defaults"]["harness"]["tool_timeout"] == 120
    assert "def before_tool" in meta["hooks_example"]
    for t in meta["templates"]:  # every template must be a valid config
        AgentConfig.model_validate({**meta["defaults"], **t["config"],
                                    "harness": {**meta["defaults"]["harness"], **t["config"].get("harness", {})}})


def test_index_served(client):
    assert "Agent Harness Builder" in client.get("/").text


def test_agent_crud(client, tmp_path):
    cfg = {"name": "demo", "tools": ["bash"], "harness": {"require_approval": ["bash"]}}
    assert client.put("/api/agents/demo", json=cfg).json() == {"ok": True}
    assert (tmp_path / "agents" / "demo.json").exists()
    assert client.get("/api/agents").json()[0]["name"] == "demo"
    assert client.get("/api/agents/demo").json()["harness"]["require_approval"] == ["bash"]
    assert client.delete("/api/agents/demo").json() == {"ok": True}
    assert client.get("/api/agents/demo").status_code == 404


@pytest.mark.parametrize("name, body, status", [
    ("Bad", {"name": "Bad"}, 422),          # invalid name in body
    ("a", {"name": "b"}, 400),              # mismatch
    ("a.b", {"name": "a.b"}, 422),         # path characters
])
def test_agent_validation(client, name, body, status):
    assert client.put(f"/api/agents/{name}", json=body).status_code == status


def test_session_rejects_broken_code(client):
    r = client.post("/api/sessions", json={"name": "x", "custom_tools": [{"name": "t", "description": "d", "code": "oops("}]})
    assert r.status_code == 422 and "SyntaxError" in r.json()["detail"]
    r = client.post("/api/sessions", json={"name": "x", "harness": {"hooks_code": "x = 1"}})
    assert r.status_code == 422


def test_unknown_session_and_approval(client):
    assert client.post("/api/sessions/nope/messages", json={"text": "hi"}).status_code == 404
    assert client.post("/api/sessions/nope/approvals/1", json={"approve": True}).status_code == 404


def _post(url, data):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10)


@pytest.mark.parametrize("approve", [True, False])
def test_chat_with_approval_round_trip(mock_api, tmp_path, approve):
    """A real server: the SSE stream pauses at approval_request until the browser answers."""
    mock_api.openai.extend([openai_turn(tool=("list_files", {})), openai_turn("all done")])
    cfg = {"name": "demo", "tools": ["list_files"], "harness": {"require_approval": ["list_files"]},
           "provider": {"type": "openai", "model": "m", "base_url": mock_api.url + "/v1"}}
    with ServerThread(create_app(tmp_path)) as srv:
        sid = json.load(_post(f"{srv.url}/api/sessions", cfg))["session_id"]
        events = []

        def stream():
            with _post(f"{srv.url}/api/sessions/{sid}/messages", {"text": "hi"}) as resp:
                for line in resp:
                    if line.startswith(b"data: "):
                        ev = json.loads(line[6:])
                        events.append(ev)
                        if ev["type"] == "approval_request":
                            _post(f"{srv.url}/api/sessions/{sid}/approvals/{ev['id']}", {"approve": approve})

        t = threading.Thread(target=stream)
        t.start()
        t.join(15)
    kinds = [e["type"] for e in events]
    assert kinds == ["approval_request" if k == "x" else k for k in
                     ["tool_call", "x", "approval_result", "tool_result", "text", "done"]]
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["is_error"] is (not approve)


def test_export_contents(client):
    r = client.post("/api/export", json={"name": "demo", "harness": {"require_approval": ["bash"], "log_dir": "./logs"}})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"demo/run.py", "demo/agent.json", "demo/requirements.txt", "demo/README.md",
            "demo/harness/agent.py", "demo/harness/hooks.py", "demo/harness/providers.py"} <= names
    readme = z.read("demo/README.md").decode()
    assert "asks before running: bash" in readme and "logs JSONL" in readme


def test_exported_agent_runs_standalone(mock_api, tmp_path):
    mock_api.openai.extend([openai_turn(tool=("write_file", {"path": "out.txt", "content": "made by agent"})),
                            openai_turn("Wrote it.")])
    cfg = AgentConfig.model_validate({"name": "demo", "tools": ["write_file"],
                                      "provider": {"type": "openai", "model": "m", "base_url": mock_api.url + "/v1"}})
    zipfile.ZipFile(io.BytesIO(build_export(cfg))).extractall(tmp_path)
    proc = subprocess.run([sys.executable, "run.py", "-p", "write the file"], cwd=tmp_path / "demo",
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Wrote it." in proc.stdout
    assert (tmp_path / "demo" / "workspace" / "out.txt").read_text() == "made by agent"
