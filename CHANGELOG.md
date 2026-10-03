# Changelog

## 0.1.0 — 2026-10-03

First public release.

### Install

Requires Python 3.10+ and Git.

```bash
pipx install git+https://github.com/thaaaru/agent-harness-builder@v0.1.0
# or, in a virtual environment:
pip install git+https://github.com/thaaaru/agent-harness-builder@v0.1.0
```

Then run `harness-builder` and open http://127.0.0.1:8765. See the
[README](README.md#install) for Windows PowerShell commands, API keys and Ollama.

### Supported providers

- **Anthropic** (Claude) through the Messages API. Supports streaming, adaptive thinking,
  effort, prompt caching, server-side refusal fallback and server-side context editing.
- **OpenAI-compatible** Chat Completions APIs: OpenAI, Ollama, LM Studio, vLLM, OpenRouter
  and similar. The model must support tool (function) calling.

### Features

- Web builder: templates, model settings, built-in and custom Python tools, harness
  settings, a streaming playground with Approve/Deny prompts, and zip export.
- Harness: per-tool approval, hooks (`before_tool`, `after_tool`, `on_event`), token
  budget, tool timeout, clearing of old tool results, JSONL transcripts, API retries.
- `harness-run` terminal runner. Exported projects run with only `anthropic` or `openai`
  plus `pydantic`.

### Known limitations

- **Not a sandbox.** Only the file tools are restricted to the workspace. `bash`,
  `http_get`, custom tools and hooks run with your user's permissions. Approvals and hooks
  are controls, not isolation.
- **No authentication.** Keep the builder on 127.0.0.1.
- CI runs on Linux only (Python 3.10–3.14). macOS and Windows have not been tested. On
  Windows, `bash` runs commands through `cmd.exe`.
- Provider code is tested against local mocks of the Anthropic and OpenAI streaming APIs,
  not live models. Ollama, LM Studio and vLLM compatibility has not been verified.
- The OpenAI-compatible provider uses Chat Completions only, not the Responses API.
  *Max output tokens*, *Effort* and *Refusal fallback* apply only to Anthropic.
- The token budget is checked after each model call, so the last call can overshoot it.
- Custom-tool arguments are checked for required fields and top-level types only.
- Playground sessions are kept in memory and end when the server stops.
- The package installs top-level modules named `harness` and `builder`. Use pipx or a
  dedicated virtual environment to avoid name clashes.
- Not published to PyPI. Install from GitHub.
