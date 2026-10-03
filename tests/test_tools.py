import asyncio

import pytest

from harness import AgentConfig
from harness.config import CustomTool
from harness.tools import BUILTIN_TOOLS, ToolContext, build_tools, compile_custom_tool, validate_args

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}, "required": ["s"]}


@pytest.mark.parametrize("args, problem", [
    ({"s": "x"}, None),
    ({"s": "x", "n": 3}, None),
    ({}, "missing required field 's'"),
    ({"s": 1}, "field 's' should be string"),
    ({"s": "x", "n": "3"}, "field 'n' should be integer"),
    ({"s": "x", "n": True}, "field 'n' should be integer"),
    ([1, 2], "expected a JSON object"),
])
def test_validate_args(args, problem):
    assert validate_args(SCHEMA, args) == problem


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(workspace=tmp_path.resolve())


def run(name, args, ctx):
    return asyncio.run(BUILTIN_TOOLS[name].run(args, ctx))


def test_file_tools_round_trip(ctx):
    assert run("write_file", {"path": "a/b.txt", "content": "hello world"}, ctx) == ("Wrote 11 chars to a/b.txt", False)
    assert run("read_file", {"path": "a/b.txt"}, ctx) == ("hello world", False)
    assert run("edit_file", {"path": "a/b.txt", "old_text": "world", "new_text": "there"}, ctx)[1] is False
    assert run("read_file", {"path": "a/b.txt"}, ctx)[0] == "hello there"
    assert run("list_files", {}, ctx)[0] == "a/b.txt"


def test_edit_requires_unique_match(ctx):
    run("write_file", {"path": "f", "content": "x x"}, ctx)
    out, err = run("edit_file", {"path": "f", "old_text": "x", "new_text": "y"}, ctx)
    assert err and "found 2 matches" in out


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "a/../../x"])
def test_paths_cannot_escape_workspace(ctx, path):
    out, err = run("read_file", {"path": path}, ctx)
    assert err and "outside the workspace" in out


def test_bash_runs_in_workspace(ctx):
    out, err = run("bash", {"command": "pwd"}, ctx)
    assert not err and str(ctx.workspace) in out and "exit code 0" in out


def test_http_get_rejects_other_schemes(ctx):
    out, err = run("http_get", {"url": "file:///etc/passwd"}, ctx)
    assert err and "only http(s)" in out


def test_long_output_is_truncated(ctx):
    run("write_file", {"path": "big", "content": "x" * 50_000}, ctx)
    out, _ = run("read_file", {"path": "big"}, ctx)
    assert len(out) < 21_000 and "truncated" in out


def test_custom_tool(ctx):
    tool = compile_custom_tool(CustomTool(name="up", description="d", code="def run(args):\n    return args['s'].upper()",
                                          parameters=SCHEMA))
    assert asyncio.run(tool.run({"s": "hi"}, ctx)) == ("HI", False)
    assert asyncio.run(tool.run({}, ctx))[1] is True


def test_custom_tool_exceptions_become_errors(ctx):
    tool = compile_custom_tool(CustomTool(name="boom", description="d", code="def run(args):\n    raise KeyError('k')"))
    out, err = asyncio.run(tool.run({}, ctx))
    assert err and out.startswith("KeyError")


def test_custom_tool_needs_run():
    with pytest.raises(ValueError, match="must define a function run"):
        compile_custom_tool(CustomTool(name="t", description="d", code="x = 1"))


def test_build_tools_rejects_unknown_and_duplicates():
    with pytest.raises(ValueError, match="unknown built-in"):
        build_tools(AgentConfig(tools=["nope"]))
    dup = CustomTool(name="bash", description="d", code="def run(a): return ''")
    with pytest.raises(ValueError, match="unique"):
        build_tools(AgentConfig(tools=["bash"], custom_tools=[dup]))
