"""Provider adapters against a local mock of the real HTTP APIs."""

import asyncio

from conftest import anthropic_turn, collect, openai_turn

from harness import Agent, AgentConfig


def agent_for(mock, tmp_path, ptype="anthropic", **cfg):
    provider = {"type": ptype, "base_url": mock.url if ptype == "anthropic" else mock.url + "/v1",
                "model": "claude-opus-5-5" if ptype == "anthropic" else "llama3", **cfg.pop("provider", {})}
    return Agent(AgentConfig.model_validate({"tools": ["list_files"], "provider": provider, **cfg}), base_dir=tmp_path)


async def run_all(agent, *texts):
    out = []
    for t in texts:
        out.append(await collect(agent, t))
    await agent.aclose()
    return out


# ---- Anthropic ------------------------------------------------------------------------

def test_anthropic_tool_loop_and_history(mock_api, tmp_path):
    mock_api.anthropic.extend([
        anthropic_turn("Looking", thinking="hmm", tool=("list_files", {"pattern": "*"})),
        anthropic_turn("Nothing here."),
        anthropic_turn("Again."),
    ])
    first, second = asyncio.run(run_all(agent_for(mock_api, tmp_path), "hi", "again"))
    assert [e["type"] for e in first] == ["thinking", "text", "usage", "tool_call", "tool_result",
                                          "text", "usage", "done"]
    assert first[0]["text"] == "hmm"
    assert first[3]["input"] == {"pattern": "*"}

    # Third request replays the whole history unchanged, thinking block included.
    msgs = mock_api.requests[2]["body"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["type"] for b in msgs[1]["content"]] == ["thinking", "text", "tool_use"]
    assert msgs[1]["content"][0]["signature"] == "sig"
    assert msgs[2]["content"][0]["type"] == "tool_result"


def test_anthropic_request_shape(mock_api, tmp_path):
    mock_api.anthropic.append(anthropic_turn("ok"))
    agent = agent_for(mock_api, tmp_path, provider={"effort": "high"},
                      harness={"clear_tool_results": True, "clear_trigger_tokens": 50000, "keep_tool_results": 2})
    asyncio.run(run_all(agent, "hi"))
    req = mock_api.requests[0]
    body = req["body"]
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert body["tools"][0]["eager_input_streaming"] is True
    assert body["context_management"]["edits"][0] == {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 50000},
        "keep": {"type": "tool_uses", "value": 2},
    }
    assert req["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01,context-management-2025-06-27"


def test_anthropic_minimal_request_for_haiku(mock_api, tmp_path):
    mock_api.anthropic.append(anthropic_turn("ok"))
    asyncio.run(run_all(agent_for(mock_api, tmp_path, provider={"model": "claude-haiku-4-5"}, tools=[]), "hi"))
    req = mock_api.requests[0]
    for key in ("thinking", "fallbacks", "output_config", "context_management", "tools"):
        assert key not in req["body"]
    assert "anthropic-beta" not in req["headers"]


def test_anthropic_refusal(mock_api, tmp_path):
    mock_api.anthropic.append(anthropic_turn(stop_reason="refusal",
                                             stop_details={"type": "refusal", "category": "cyber", "explanation": None}))
    (events,) = asyncio.run(run_all(agent_for(mock_api, tmp_path), "hi"))
    assert events[-1]["type"] == "error" and "declined" in events[-1]["message"]


def test_anthropic_truncated_tool_call_is_not_run(mock_api, tmp_path):
    mock_api.anthropic.append(anthropic_turn(tool=("list_files", {"pattern": "*"}), stop_reason="max_tokens"))
    (events,) = asyncio.run(run_all(agent_for(mock_api, tmp_path), "hi"))
    assert "tool_call" not in [e["type"] for e in events]
    assert "max_tokens" in events[-1]["message"]


# ---- OpenAI-compatible -------------------------------------------------------------------

def test_openai_tool_loop_and_history(mock_api, tmp_path):
    mock_api.openai.extend([
        openai_turn(reasoning="plan", tool=("list_files", {"pattern": "*.py"})),
        openai_turn("Done!"),
    ])
    (events,) = asyncio.run(run_all(agent_for(mock_api, tmp_path, "openai", system_prompt="be brief"), "hi"))
    assert [e["type"] for e in events] == ["thinking", "tool_call", "tool_result", "text", "done"]
    assert events[1]["input"] == {"pattern": "*.py"}  # arguments reassembled from two chunks

    msgs = mock_api.requests[1]["body"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "list_files"
    assert msgs[3]["tool_call_id"] == "call_list_files"


def test_openai_clears_old_tool_results_in_view_only(mock_api, tmp_path):
    turns = [openai_turn(tool=("list_files", {})) for _ in range(3)] + [openai_turn("end")]
    mock_api.openai.extend(turns)
    agent = agent_for(mock_api, tmp_path, "openai",
                      harness={"clear_tool_results": True, "clear_trigger_tokens": 1000, "keep_tool_results": 1})
    agent.provider.messages.append({"role": "user", "content": "x" * 8000})  # push past the trigger
    asyncio.run(run_all(agent, "hi"))

    sent = [m for m in mock_api.requests[-1]["body"]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in sent[:-1]] == ["[cleared to save context]"] * 2
    assert sent[-1]["content"] != "[cleared to save context]"
    stored = [m for m in agent.provider.messages if m["role"] == "tool"]
    assert all(m["content"] != "[cleared to save context]" for m in stored)


def test_openai_truncated_tool_call_is_not_run(mock_api, tmp_path):
    mock_api.openai.append(openai_turn(tool=("list_files", {}), finish="length"))
    (events,) = asyncio.run(run_all(agent_for(mock_api, tmp_path, "openai"), "hi"))
    assert events[-1]["type"] == "error" and "truncated" in events[-1]["message"]
