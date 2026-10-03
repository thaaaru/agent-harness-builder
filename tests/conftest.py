"""Shared fixtures: a scripted fake provider and a local mock of the Anthropic/OpenAI APIs."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from harness import Agent, AgentConfig, Provider


# ---- fake provider -----------------------------------------------------------

class FakeProvider(Provider):
    """Replays scripted turns. Each turn is a list of tool calls ([] = final answer)."""

    def __init__(self, turns: list[list[dict]], usage: int = 10):
        super().__init__(AgentConfig().provider)
        self.turns = deque(turns)
        self.usage = usage
        self.tool_results: list[list[dict]] = []

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results):
        self.tool_results.append(results)

    async def stream_turn(self, system, tools):
        yield {"type": "text", "text": "thinking about it"}
        yield {"type": "usage", "input_tokens": self.usage, "output_tokens": self.usage, "cache_read_tokens": 0}
        calls = self.turns.popleft() if self.turns else []
        yield {"type": "turn_end", "tool_calls": calls, "error": None}

    async def aclose(self):
        pass


def call(id: str, name: str, **input) -> dict:
    return {"id": id, "name": name, "input": input}


@pytest.fixture
def make_agent(tmp_path):
    def make(turns, approver=None, hooks=None, usage=10, **cfg) -> Agent:
        agent = Agent(AgentConfig.model_validate(cfg), base_dir=tmp_path, approver=approver, hooks=hooks)
        agent.provider = FakeProvider(turns, usage=usage)
        return agent
    return make


async def collect(agent: Agent, text: str = "go") -> list[dict]:
    return [ev async for ev in agent.send(text)]


# ---- local server helper -------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, app):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(5)


# ---- mock model APIs -------------------------------------------------------------

class MockAPI:
    """Serves queued, pre-built streaming responses and records every request."""

    def __init__(self):
        self.anthropic: deque[list[dict]] = deque()
        self.openai: deque[list[dict]] = deque()
        self.requests: list[dict] = []
        self.app = FastAPI()

        @self.app.post("/v1/messages")
        async def messages(req: Request):
            self.requests.append({"headers": dict(req.headers), "body": await req.json()})
            events = self.anthropic.popleft()
            body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
            return StreamingResponse(iter([body]), media_type="text/event-stream")

        @self.app.post("/v1/chat/completions")
        async def chat(req: Request):
            self.requests.append({"headers": dict(req.headers), "body": await req.json()})
            chunks = self.openai.popleft()
            base = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m"}
            body = "".join(f"data: {json.dumps({**base, 'choices': [c]})}\n\n" for c in chunks)
            return StreamingResponse(iter([body + "data: [DONE]\n\n"]), media_type="text/event-stream")


def anthropic_turn(text="", thinking=None, tool=None, stop_reason=None, stop_details=None) -> list[dict]:
    """Build the SSE events of one Claude response."""
    blocks, i = [], 0
    if thinking is not None:
        blocks += [
            {"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": thinking}},
            {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": "sig"}},
            {"type": "content_block_stop", "index": i}]
        i += 1
    if text:
        blocks += [
            {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": i}]
        i += 1
    if tool:
        name, args = tool
        blocks += [
            {"type": "content_block_start", "index": i,
             "content_block": {"type": "tool_use", "id": f"tu_{name}", "name": name, "input": {}}},
            {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}},
            {"type": "content_block_stop", "index": i}]
    stop = stop_reason or ("tool_use" if tool else "end_turn")
    msg = {"id": "msg", "type": "message", "role": "assistant", "model": "m", "content": [],
           "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 1}}
    delta = {"stop_reason": stop, "stop_sequence": None}
    if stop_details:
        delta["stop_details"] = stop_details
    return ([{"type": "message_start", "message": msg}] + blocks +
            [{"type": "message_delta", "delta": delta, "usage": {"output_tokens": 20}},
             {"type": "message_stop"}])


def openai_turn(text="", tool=None, reasoning=None, finish=None) -> list[dict]:
    chunks = []
    if reasoning:
        chunks.append({"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}, "finish_reason": None})
    if text:
        chunks.append({"index": 0, "delta": {"content": text}, "finish_reason": None})
    if tool:
        name, args = tool
        raw = json.dumps(args)
        half = len(raw) // 2  # split the arguments across two chunks, like real streams
        chunks.append({"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": raw[:half]}}]},
            "finish_reason": None})
        chunks.append({"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[half:]}}]},
                       "finish_reason": None})
    chunks.append({"index": 0, "delta": {}, "finish_reason": finish or ("tool_calls" if tool else "stop")})
    return chunks


@pytest.fixture(scope="session")
def mock_server():
    api = MockAPI()
    with ServerThread(api.app) as srv:
        api.url = srv.url
        yield api


@pytest.fixture
def mock_api(mock_server, monkeypatch):
    mock_server.anthropic.clear()
    mock_server.openai.clear()
    mock_server.requests.clear()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return mock_server
