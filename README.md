# Agent Harness Builder

[![CI](https://github.com/thaaaru/agent-harness-builder/actions/workflows/ci.yml/badge.svg)](https://github.com/thaaaru/agent-harness-builder/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Design an AI agent in a web form, test it in a live playground, and export it as a small
Python project you own. The harness behind each agent (the loop, approvals, hooks, budgets,
context clearing and logging) is a readable ~900-line package. Use it as is, or fork it as
the start of your own harness.

- **Any model.** Use Claude (Anthropic API) or anything that speaks the OpenAI Chat
  Completions API: OpenAI, Ollama, LM Studio, vLLM, OpenRouter, Groq and others.
- **Tools.** Built-in file, shell and HTTP tools confined to a workspace folder, plus
  your own tools written as one Python function.
- **Harness controls.** Human approval per tool, Python hooks to block or rewrite tool
  calls, token budgets, tool timeouts, clearing of old tool results, JSONL transcripts and
  API retries.
- **No lock-in.** Export a zip containing `run.py`, `agent.json` and the harness source.
  It needs only `anthropic` or `openai` plus `pydantic`.

## Quick start

```bash
pip install git+https://github.com/thaaaru/agent-harness-builder
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY; nothing is needed for local Ollama
harness-builder                  # → http://127.0.0.1:8765
```

Saved agents go in `./agents/` and test-chat workspaces in `./playground/` (change this with `--dir`).

### Your first agent in two minutes

1. Click **Coding assistant** under *New from template*. It comes with file and shell
   tools, asks before running `bash`, and logs transcripts.
2. Change the model if you like, for example to `llama3.3` with provider
   *OpenAI-compatible* and base URL `http://localhost:11434/v1` for Ollama.
3. In the playground, type: *"Create fizzbuzz.py and run it"*. You'll see the agent
   think, write the file, and stop to ask you before it runs anything.
4. Click **Save**, then **Export .zip**. Unzip it and run it anywhere:

```bash
cd coder && pip install -r requirements.txt
python run.py                       # interactive chat
python run.py -p "add tests"        # one-shot; exit code 1 on error
python run.py --yes                 # approve every tool call without asking
```

`harness-run path/to/agent.json` does the same without exporting.

## Use the harness from Python

```python
import asyncio
from harness import Agent, AgentConfig, Hooks

class NoDeletes(Hooks):
    def before_tool(self, call):
        if call["name"] == "bash" and "rm " in call["input"]["command"]:
            return "deleting files is not allowed"      # blocks the call, tells the model why

async def approve(call):
    return input(f"Run {call['name']}? [y/N] ") == "y"

agent = Agent(AgentConfig.load("agent.json"), approver=approve, hooks=[NoDeletes()])

async def main():
    async for event in agent.send("Clean up the build folder"):
        if event["type"] == "text":
            print(event["text"], end="", flush=True)
    await agent.aclose()

asyncio.run(main())
```

**[docs/extending.md](docs/extending.md)** covers custom tools, hooks, approvals, writing a
new provider, and every harness setting.

## How it works

```
             ┌──────────────── Agent.send(text) ────────────────┐
user text ──▶│ provider.stream_turn() ──▶ text / thinking events │
             │        │ tool calls                               │
             │        ▼                                          │
             │ hooks.before_tool ─▶ approval? ─▶ run (timeout)   │
             │        │                     concurrently         │
             │        ▼                                          │
             │ hooks.after_tool ─▶ provider.add_tool_results()   │
             │        └──── repeat until no tool calls, ─────────┘
             │              max_turns, or token_budget
```

| File | What it does |
|---|---|
| `harness/agent.py` | The loop, approvals, budget and timeouts |
| `harness/providers.py` | Anthropic and OpenAI-compatible adapters (history kept in each API's native format) |
| `harness/tools.py` | Built-in tools, custom-tool compiler, argument validation |
| `harness/hooks.py` | `Hooks` base class, code hooks, JSONL logger |
| `harness/config.py` | `agent.json` schema |
| `harness/cli.py` | Terminal runner |
| `builder/` | FastAPI server and the single-file web UI |

## Safety

The server only listens on 127.0.0.1. File tools refuse paths outside the agent's
workspace. **`bash`, `http_get`, custom tools and hooks run unsandboxed with your user's
permissions.** Turn on *Ask before running* for anything risky, and run untrusted agents
in a container or VM.

## Development

```bash
git clone https://github.com/thaaaru/agent-harness-builder && cd agent-harness-builder
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q             # no API keys needed: model APIs are mocked locally
.venv/bin/harness-builder
```

Issues and pull requests are welcome.

## License

[MIT](LICENSE)
