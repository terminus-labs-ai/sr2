"""Agent configuration for SR2Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"
    name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 4096
    extra: dict[str, Any] = Field(default_factory=dict)


class PersonaConfig(BaseModel):
    """Agent persona."""

    system_prompt: str


class ContextConfig(BaseModel):
    """Context engine configuration.

    These fields map to SR2's PipelineConfig. The mapping is handled
    by SR2Runtime._build_sr2_config().

    For advanced usage, ``pipeline_override`` allows passing raw
    PipelineConfig fields directly — useful for features not exposed
    in the simplified schema (degradation, intent detection, etc.).
    """

    context_window: int = 32768
    conversation: dict[str, Any] = Field(default_factory=lambda: {
        "active_turns": 10,
        "buffer_turns": 5,
        "compaction": "summarize",
    })
    memory: dict[str, Any] = Field(default_factory=lambda: {
        "enabled": False,
    })
    pipeline_override: dict[str, Any] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    """Tool definition reference."""

    name: str
    module: str
    config: dict[str, Any] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    """Output behavior."""

    format: str = "freeform"
    schema_ref: str | None = None
    max_tool_iterations: int = 10


class AgentConfig(BaseModel):
    """Complete agent configuration."""

    name: str
    description: str = ""
    model: ModelConfig
    persona: PersonaConfig
    context: ContextConfig = Field(default_factory=ContextConfig)
    tools: list[ToolConfig] = Field(default_factory=list)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        """Load agent config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> AgentConfig:
        """Load agent config from a dictionary."""
        return cls(**data)
