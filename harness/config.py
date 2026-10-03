"""Agent configuration: the single JSON document that fully describes an agent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ProviderConfig(BaseModel):
    type: Literal["anthropic", "openai"] = "anthropic"
    model: str = "claude-opus-5-5"
    # OpenAI-compatible only: e.g. http://localhost:11434/v1 for Ollama.
    base_url: str | None = None
    # Name of the environment variable holding the API key (never the key itself).
    api_key_env: str | None = None
    max_tokens: int = Field(default=64000, ge=256, le=128000)
    # Anthropic only: low | medium | high | xhigh | max. None = model default.
    effort: str | None = None
    # Anthropic only: re-run declined requests on Anthropic's recommended fallback model.
    refusal_fallback: bool = True


class CustomTool(BaseModel):
    name: str
    description: str
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    # Python source that defines `def run(args: dict) -> str`.
    code: str

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]{0,63}", v):
            raise ValueError("tool name must be letters, digits, _ or -")
        return v


class HarnessConfig(BaseModel):
    """How the loop behaves, independent of which model or tools are used."""

    # Tool names that need a human "yes" before running; "*" means every tool.
    require_approval: list[str] = Field(default_factory=list)
    # Seconds before a running tool is abandoned and reported as an error.
    tool_timeout: int = Field(default=120, ge=1, le=3600)
    # Stop after this many input+output tokens for one user message. None = no limit.
    token_budget: int | None = Field(default=None, ge=1000)
    # Clear old tool results once the context grows past clear_trigger_tokens.
    clear_tool_results: bool = False
    clear_trigger_tokens: int = Field(default=100_000, ge=1000)
    keep_tool_results: int = Field(default=3, ge=0)
    # Directory (relative to the agent) for JSONL transcripts. None = no logging.
    log_dir: str | None = None
    # Times the SDK retries rate limits, 5xx and connection errors.
    api_retries: int = Field(default=2, ge=0, le=10)
    # Python source defining any of: on_event(event), before_tool(call), after_tool(call, result).
    hooks_code: str = ""


class AgentConfig(BaseModel):
    name: str = "my-agent"
    description: str = ""
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    system_prompt: str = "You are a helpful assistant."
    tools: list[str] = Field(default_factory=list)
    custom_tools: list[CustomTool] = Field(default_factory=list)
    max_turns: int = Field(default=25, ge=1, le=500)
    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    # Agent working directory: file tools are confined to it; bash starts in it but is not confined.
    workspace: str = "./workspace"

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        if not NAME_RE.fullmatch(v):
            raise ValueError("name must be lowercase letters, digits, - or _ (max 64)")
        return v

    @classmethod
    def load(cls, path: str | Path) -> "AgentConfig":
        return cls.model_validate(json.loads(Path(path).read_text()))

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")
