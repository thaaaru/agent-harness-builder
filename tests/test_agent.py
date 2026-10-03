import asyncio
import json

from conftest import call, collect

from harness import Hooks

CUSTOM_SLEEP = {"name": "sleep", "description": "d", "code": "import time\ndef run(args):\n    time.sleep(3)\n    return 'woke'"}


def types(events):
    return [e["type"] for e in events]


def test_loop_runs_tools_until_done(make_agent):
    agent = make_agent([[call("1", "write_file", path="a.txt", content="hi")], [call("2", "read_file", path="a.txt")], []],
                       tools=["write_file", "read_file"])
    events = asyncio.run(collect(agent))
    results = [e for e in events if e["type"] == "tool_result"]
    assert [r["output"] for r in results] == ["Wrote 2 chars to a.txt", "hi"]
    assert events[-1]["type"] == "done"
    assert len(agent.provider.tool_results) == 2


def test_parallel_calls_return_results_together(make_agent):
    agent = make_agent([[call("1", "list_files"), call("2", "list_files")], []], tools=["list_files"])
    asyncio.run(collect(agent))
    assert [r["id"] for r in agent.provider.tool_results[0]] == ["1", "2"]


def test_unknown_tool_and_bad_json_are_errors(make_agent):
    agent = make_agent([[call("1", "nope"), {"id": "2", "name": "list_files", "input": {"__invalid_json__": "{"}}], []],
                       tools=["list_files"])
    asyncio.run(collect(agent))
    out = agent.provider.tool_results[0]
    assert all(r["is_error"] for r in out)
    assert "Unknown tool" in out[0]["output"] and "not valid JSON" in out[1]["output"]


def test_max_turns(make_agent):
    agent = make_agent([[call(str(i), "list_files")] for i in range(5)], tools=["list_files"], max_turns=2)
    events = asyncio.run(collect(agent))
    assert events[-1] == {"type": "error", "message": "Stopped after max_turns=2."}
    assert len(agent.provider.tool_results) == 2  # every call still got a result


# ---- approvals -------------------------------------------------------------------

def test_approval_granted(make_agent):
    seen = []

    async def approve(c):
        seen.append(c["name"])
        return True

    agent = make_agent([[call("1", "list_files")], []], approver=approve, tools=["list_files"],
                       harness={"require_approval": ["list_files"]})
    events = asyncio.run(collect(agent))
    assert seen == ["list_files"]
    assert "approval_request" in types(events)
    assert {"type": "approval_result", "id": "1", "approved": True} in events
    assert agent.provider.tool_results[0][0]["is_error"] is False


def test_approval_denied_and_missing_approver(make_agent):
    async def deny(c):
        return False

    for approver in (deny, None):
        agent = make_agent([[call("1", "bash", command="echo hi")], []], approver=approver, tools=["bash"],
                           harness={"require_approval": ["*"]})
        asyncio.run(collect(agent))
        result = agent.provider.tool_results[0][0]
        assert result["is_error"] and result["output"] == "The user denied this tool call."


def test_tools_not_listed_skip_approval(make_agent):
    agent = make_agent([[call("1", "list_files")], []], tools=["list_files", "bash"],
                       harness={"require_approval": ["bash"]})
    events = asyncio.run(collect(agent))
    assert "approval_request" not in types(events)


# ---- hooks -------------------------------------------------------------------------

class Recorder(Hooks):
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event["type"])

    async def before_tool(self, c):  # async hooks work too
        return "no shell" if c["name"] == "bash" else None

    def after_tool(self, c, result):
        return result["output"].upper() if c["name"] == "list_files" else None


def test_hook_object(make_agent, tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "f.txt").write_text("")
    rec = Recorder()
    agent = make_agent([[call("1", "bash", command="ls"), call("2", "list_files")], []], hooks=[rec],
                       tools=["bash", "list_files"])
    events = asyncio.run(collect(agent))
    bash, files = agent.provider.tool_results[0]
    assert bash == {"id": "1", "name": "bash", "output": "Blocked: no shell", "is_error": True}
    assert files["output"] == "F.TXT"
    assert rec.events == types(events)


def test_code_hooks(make_agent):
    code = "def before_tool(call):\n    return 'nope' if call['input'].get('path') == 'secret' else None\n"
    agent = make_agent([[call("1", "read_file", path="secret")], []], tools=["read_file"], harness={"hooks_code": code})
    asyncio.run(collect(agent))
    assert agent.provider.tool_results[0][0]["output"] == "Blocked: nope"


def test_hooks_code_must_define_a_hook(make_agent):
    try:
        make_agent([], harness={"hooks_code": "x = 1"})
    except ValueError as e:
        assert "defines none of" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---- limits and logging ---------------------------------------------------------------

def test_token_budget_stops_and_answers_pending_calls(make_agent):
    agent = make_agent([[call("1", "list_files")], [call("2", "list_files")], []], usage=600, tools=["list_files"],
                       harness={"token_budget": 2000})
    events = asyncio.run(collect(agent))
    assert events[-1]["type"] == "error" and "Token budget" in events[-1]["message"]
    # Turn 1 used 1,200 tokens and ran; turn 2 crossed 2,000, so its call was answered, not run.
    assert agent.provider.tool_results[0][0]["is_error"] is False
    assert agent.provider.tool_results[1][0]["output"] == "Not run: token budget exhausted."


def test_tool_timeout(make_agent):
    agent = make_agent([[call("1", "sleep")], []], custom_tools=[CUSTOM_SLEEP], harness={"tool_timeout": 1})
    asyncio.run(collect(agent))
    result = agent.provider.tool_results[0][0]
    assert result["is_error"] and "timed out after 1s" in result["output"]


def test_jsonl_logging(make_agent):
    agent = make_agent([[call("1", "list_files")], []], tools=["list_files"], harness={"log_dir": "logs"})
    events = asyncio.run(collect(agent))
    lines = [json.loads(line) for line in agent.log_path.read_text().splitlines()]
    assert [line["type"] for line in lines] == types(events)
    assert all("ts" in line for line in lines)
