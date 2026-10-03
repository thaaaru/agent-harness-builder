# Agent Harness Builder

A local web app for designing, testing and exporting AI agents.

- **Build**: name, system prompt, model, built-in tools, and custom Python tools, all in a form
- **Test**: chat with the agent in the playground, with streaming text, thinking and tool calls shown live
- **Export**: download a standalone `.zip` (`run.py` + `agent.json` + the small `harness/` package)

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # and/or OPENAI_API_KEY; nothing is needed for Ollama
.venv/bin/python app.py             # → http://127.0.0.1:8765
```

## Providers

| Provider | Covers |
|---|---|
| `anthropic` | Claude models: adaptive thinking, effort, prompt caching, refusal fallback |
| `openai` | Any OpenAI-compatible Chat Completions API: OpenAI, Ollama (`http://localhost:11434/v1`), LM Studio, vLLM, OpenRouter, Groq… |

To add a provider, subclass `harness.providers.Provider` (`add_user`, `stream_turn`,
`add_tool_results`) and register it in `PROVIDERS`. Each provider keeps history in its
native format, so nothing is lost in translation.

## Layout

```
app.py              FastAPI server: saving, playground sessions (SSE), zip export
templates.py        starter templates
static/index.html   the builder UI (no build step)
harness/            the runtime that is also shipped inside every export
  config.py         AgentConfig, the agent.json schema
  agent.py          the loop: model turn → tools (concurrent) → results → repeat
  providers.py      Anthropic + OpenAI-compatible adapters
  tools.py          built-in tools, custom-tool compiler, arg validation
  cli.py            terminal REPL / one-shot runner
agents/             saved agent configs
playground/         per-agent workspaces used by test chats
```

## Safety notes

The server binds to 127.0.0.1 only. File tools are confined to the agent's workspace.
`bash` and custom tools run unsandboxed with your user's permissions, so only enable them
for agents you trust with your machine.
