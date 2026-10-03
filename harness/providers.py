"""LLM providers.

Every provider keeps the conversation in its own native message format (so nothing is
lost in translation, e.g. Claude's thinking blocks) and exposes the same three calls:

    add_user(text)                 append a user message
    stream_turn(system, tools)     async-iterate events for one model response
    add_tool_results(results)      append the results of the tool calls just made

stream_turn yields UI events ({"type": "text" | "thinking" | "usage", ...}) and ends
with exactly one {"type": "turn_end", "tool_calls": [...], "error": str | None}.
To add a provider, subclass Provider, implement these three calls and register it in PROVIDERS.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

from .config import HarnessConfig, ProviderConfig
from .tools import Tool

# Models that accept the server-side `fallbacks: "default"` parameter.
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CONTEXT_EDITING_BETA = "context-management-2025-06-27"
CLEARED = "[cleared to save context]"


class Provider:
    def __init__(self, cfg: ProviderConfig, harness: HarnessConfig | None = None):
        self.cfg = cfg
        self.harness = harness or HarnessConfig()
        self.messages: list[dict] = []

    def _api_key(self) -> str | None:
        return os.environ.get(self.cfg.api_key_env) if self.cfg.api_key_env else None

    def add_user(self, text: str) -> None: ...
    def add_tool_results(self, results: list[dict]) -> None: ...
    def stream_turn(self, system: str, tools: list[Tool]) -> AsyncIterator[dict]: ...

    async def aclose(self) -> None:
        if client := getattr(self, "client", None):
            await client.close()


class AnthropicProvider(Provider):
    def __init__(self, cfg: ProviderConfig, harness: HarnessConfig | None = None):
        super().__init__(cfg, harness)
        import anthropic
        self._anthropic = anthropic
        kwargs: dict = {"max_retries": self.harness.api_retries}
        if key := self._api_key():
            kwargs["api_key"] = key
        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url
        # No key passed -> SDK resolves ANTHROPIC_API_KEY / auth token / `ant auth login` profile.
        self.client = anthropic.AsyncAnthropic(**kwargs)

    def add_user(self, text: str) -> None:
        block = {"type": "text", "text": text}
        # If the last turn was cut short we may already end on a user message; merge into it.
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"].append(block)
        else:
            self.messages.append({"role": "user", "content": [block]})

    def add_tool_results(self, results: list[dict]) -> None:
        # All results for one assistant turn go back in a single user message.
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r["id"], "content": r["output"], "is_error": r["is_error"]}
            for r in results
        ]})

    def _params(self, system: str, tools: list[Tool]) -> dict:
        model = self.cfg.model
        params: dict = {
            "model": model,
            "max_tokens": self.cfg.max_tokens,
            "messages": self.messages,
            "cache_control": {"type": "ephemeral"},  # automatic prompt caching
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters,
                 "eager_input_streaming": True}
                for t in tools
            ]
        if "haiku" not in model and "claude-3" not in model:
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
        if self.cfg.effort:
            params["output_config"] = {"effort": self.cfg.effort}
        extra_body: dict = {}
        betas: list[str] = []
        if self.cfg.refusal_fallback and model in FALLBACK_MODELS:
            extra_body["fallbacks"] = "default"
            betas.append(FALLBACK_BETA)
        if self.harness.clear_tool_results:
            # Server-side context editing: never rewrite past messages client-side, because
            # replayed thinking blocks are only valid against the exact history that produced them.
            extra_body["context_management"] = {"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": self.harness.clear_trigger_tokens},
                "keep": {"type": "tool_uses", "value": self.harness.keep_tool_results},
            }]}
            betas.append(CONTEXT_EDITING_BETA)
        if extra_body:
            params["extra_body"] = extra_body
        if betas:
            params["extra_headers"] = {"anthropic-beta": ",".join(betas)}
        return params

    async def stream_turn(self, system: str, tools: list[Tool]) -> AsyncIterator[dict]:
        json_retries = 0
        while True:
            try:
                async with self.client.messages.stream(**self._params(system, tools)) as stream:
                    async for event in stream:
                        if event.type == "text":
                            yield {"type": "text", "text": event.text}
                        elif event.type == "thinking":
                            yield {"type": "thinking", "text": event.thinking}
                    response = await stream.get_final_message()
            except ValueError:
                # Tool input JSON the SDK could not parse at all: re-issue the turn (bounded).
                json_retries += 1
                if json_retries > 2:
                    yield {"type": "turn_end", "tool_calls": [], "error": "model produced unparseable tool input"}
                    return
                continue
            except self._anthropic.APIError as e:
                yield {"type": "turn_end", "tool_calls": [], "error": f"Anthropic API error: {e}"}
                return

            u = response.usage
            yield {"type": "usage", "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                   "cache_read_tokens": u.cache_read_input_tokens or 0}

            if response.stop_reason == "pause_turn":
                self.messages.append({"role": "assistant", "content": response.content})
                continue
            if response.stop_reason == "refusal":
                category = getattr(response.stop_details, "category", None)
                yield {"type": "turn_end", "tool_calls": [],
                       "error": f"The model declined this request (category: {category})."}
                return

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if tool_uses and response.stop_reason == "max_tokens":
                # A truncated tool input parses as a partial object: never run it.
                yield {"type": "turn_end", "tool_calls": [],
                       "error": "Hit max_tokens while writing a tool call; raise max_tokens."}
                return

            # Append the full content (thinking blocks included) unchanged.
            self.messages.append({"role": "assistant", "content": response.content})
            yield {"type": "turn_end", "error": None,
                   "tool_calls": [{"id": b.id, "name": b.name, "input": b.input} for b in tool_uses]}
            return


class OpenAICompatibleProvider(Provider):
    """OpenAI Chat Completions API: OpenAI, Ollama, LM Studio, vLLM, OpenRouter, Groq, ..."""

    def __init__(self, cfg: ProviderConfig, harness: HarnessConfig | None = None):
        super().__init__(cfg, harness)
        import openai
        self._openai = openai
        # Local servers usually ignore the key but the SDK requires one.
        key = self._api_key() or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.client = openai.AsyncOpenAI(api_key=key, base_url=cfg.base_url or None,
                                         max_retries=self.harness.api_retries)

    def _context_view(self) -> list[dict]:
        """History as sent: old tool results blanked once the context is large.

        The stored history is never modified, so the view is a pure function of it."""
        h = self.harness
        if not h.clear_tool_results:
            return self.messages
        approx_tokens = sum(len(json.dumps(m, default=str)) for m in self.messages) // 4
        if approx_tokens < h.clear_trigger_tokens:
            return self.messages
        tool_idx = [i for i, m in enumerate(self.messages) if m["role"] == "tool"]
        clear = set(tool_idx[:-h.keep_tool_results] if h.keep_tool_results else tool_idx)
        return [{**m, "content": CLEARED} if i in clear else m for i, m in enumerate(self.messages)]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[dict]) -> None:
        for r in results:
            content = f"ERROR: {r['output']}" if r["is_error"] else r["output"]
            self.messages.append({"role": "tool", "tool_call_id": r["id"], "content": content})

    async def stream_turn(self, system: str, tools: list[Tool]) -> AsyncIterator[dict]:
        params: dict = {
            "model": self.cfg.model,
            "messages": ([{"role": "system", "content": system}] if system else []) + self._context_view(),
            "stream": True,
        }
        if tools:
            params["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        text_parts: list[str] = []
        calls: dict[int, dict] = {}
        finish = None
        try:
            stream = await self.client.chat.completions.create(**params)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                finish = choice.finish_reason or finish
                # Reasoning models on Ollama/vLLM/OpenRouter expose this as a non-standard field.
                extra = getattr(delta, "model_extra", None) or {}
                reasoning = extra.get("reasoning_content") or extra.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    yield {"type": "thinking", "text": reasoning}
                if delta.content:
                    text_parts.append(delta.content)
                    yield {"type": "text", "text": delta.content}
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
        except self._openai.APIError as e:
            yield {"type": "turn_end", "tool_calls": [], "error": f"API error: {e}"}
            return

        if calls and finish == "length":
            yield {"type": "turn_end", "tool_calls": [], "error": "Output was truncated while writing a tool call."}
            return

        ordered = [calls[i] for i in sorted(calls)]
        for n, c in enumerate(ordered):
            c["id"] = c["id"] or f"call_{len(self.messages)}_{n}"
        assistant: dict = {"role": "assistant", "content": "".join(text_parts) or None}
        if ordered:
            assistant["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in ordered
            ]
        self.messages.append(assistant)

        tool_calls = []
        for c in ordered:
            try:
                args = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": c["arguments"]}
            tool_calls.append({"id": c["id"], "name": c["name"], "input": args})
        yield {"type": "turn_end", "tool_calls": tool_calls, "error": None}


PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAICompatibleProvider,
}


def make_provider(cfg: ProviderConfig, harness: HarnessConfig | None = None) -> Provider:
    return PROVIDERS[cfg.type](cfg, harness)
