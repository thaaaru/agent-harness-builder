# Extending the harness

The runtime in `harness/` is small on purpose (about 900 lines). There are four extension
points, from least to most code:

| You want to… | Use | Code needed |
|---|---|---|
| Give the agent a new ability | a **custom tool** | one function |
| Watch, block or rewrite what the agent does | **hooks** | one to three functions |
| Ask a human before risky actions | an **approver** | one async function |
| Use a different LLM API | a **provider** | one class, three methods |

Everything below works both in the builder UI and in exported projects.

## Custom tools

In the UI: **Custom tools → + Add tool**. In `agent.json`:

```json
"custom_tools": [{
  "name": "get_weather",
  "description": "Get the current weather for a city.",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
  "code": "def run(args):\n    return f\"Sunny in {args['city']}\""
}]
```

- `run(args)` gets arguments already checked against `parameters` (required keys and basic types).
- Return a string. If it raises, the model sees the exception as an error result and can retry.
- It runs in a worker thread and is abandoned after `harness.tool_timeout` seconds.

From Python, add a `harness.tools.Tool` to `agent.tools` and `agent.tools_by_name`, or
add an entry to `harness.tools.BUILTIN_TOOLS` so every agent can enable it by name.

## Hooks

Hooks are optional functions, and any of them can be `async`:

```python
def before_tool(call):            # call = {"id", "name", "input"}
    """Return a string to block the call. The model sees it as the reason."""

def after_tool(call, result):     # result = {"id", "name", "output", "is_error"}
    """Return a string to replace the tool's output, or None to keep it."""

def on_event(event):
    """See every event: text, thinking, usage, tool_call, approval_request,
    approval_result, tool_result, error, done."""
```

You can provide them in three ways:

1. In the UI: paste them into **Harness → Hooks** (stored as `harness.hooks_code`).
2. In Python: subclass `harness.Hooks` and pass `Agent(config, hooks=[MyHooks()])`.
3. Built in: `harness.JsonlLogger` is a hook. Setting `harness.log_dir` turns it on.

When several hooks are present, they run in order. The first one that blocks a call, or
returns a replacement output, wins.

Example: a bash allow-list plus redaction of secrets in tool output:

```python
import re
ALLOWED = ("ls", "cat", "grep", "python", "pytest", "git status", "git diff")

def before_tool(call):
    if call["name"] == "bash" and not call["input"]["command"].startswith(ALLOWED):
        return "command not on the allow-list"

def after_tool(call, result):
    return re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED]", result["output"])
```

## Approvals

Put tool names (or `"*"` for every tool) in `harness.require_approval`. Before such a
call runs, the agent emits `approval_request` and waits on the approver:

```python
async def approver(call: dict) -> bool:
    return call["name"] != "bash" or input(f"run {call['input']}? ") == "y"

agent = Agent(config, approver=approver)
```

- With no approver, calls that need approval are **denied**. Safe is the default.
- The builder UI shows Approve and Deny buttons.
- `harness-run` / `run.py` asks in the terminal. `--yes` approves everything.

## Providers

A provider keeps the conversation in its API's **native** format and implements three methods:

```python
from harness.providers import PROVIDERS, Provider

class MyProvider(Provider):
    def add_user(self, text: str) -> None: ...
    def add_tool_results(self, results: list[dict]) -> None: ...   # one entry per tool call
    async def stream_turn(self, system: str, tools: list) :
        # yield {"type": "text" | "thinking", "text": ...} while streaming
        # yield {"type": "usage", "input_tokens": .., "output_tokens": .., "cache_read_tokens": ..}
        # finish with exactly one:
        yield {"type": "turn_end", "tool_calls": [{"id", "name", "input"}, ...], "error": None}

PROVIDERS["mine"] = MyProvider
```

Then add `"mine"` to the `type` Literal in `harness/config.py` (`ProviderConfig`).

Rules worth keeping:

- **Append-only history.** Append the model's response exactly as received, and never
  edit earlier messages. Claude's thinking blocks are only valid against the exact history
  that produced them, so the Anthropic provider uses server-side context editing instead of
  trimming. The OpenAI provider trims only in the request it sends (`_context_view`) and
  never in what it stores.
- **Never run a truncated tool call.** If output was cut off while the model was writing a
  tool call, return an error in `turn_end` instead of running it.
- Turn API errors into `turn_end` with an `error` message rather than raising.

## Harness settings reference

`agent.json → harness`:

| Field | Default | Meaning |
|---|---|---|
| `require_approval` | `[]` | Tool names (or `"*"`) that need a human yes |
| `tool_timeout` | `120` | Seconds before a tool is abandoned |
| `token_budget` | `null` | Stop once one message has used this many input + output tokens |
| `clear_tool_results` | `false` | Clear old tool results when the context is large |
| `clear_trigger_tokens` | `100000` | When to start clearing |
| `keep_tool_results` | `3` | How many recent results to keep |
| `log_dir` | `null` | Directory for JSONL transcripts |
| `api_retries` | `2` | SDK retries for rate limits, 5xx and connection errors |
| `hooks_code` | `""` | Python source with hook functions |

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests never call a real model. `tests/conftest.py` has a scripted `FakeProvider` for
loop logic and a local mock of the Anthropic and OpenAI streaming APIs for provider code.
