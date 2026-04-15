"""Pydantic config models for SR2 Bridge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class BridgeForwardingConfig(BaseModel):
    """Upstream API forwarding settings."""

    upstream_url: str = Field(
        default="https://api.anthropic.com",
        description="Upstream API base URL to forward requests to.",
    )
    timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Timeout for upstream requests in seconds.",
    )
    model: str | None = Field(
        default=None,
        description="Override model for upstream requests. When set, the bridge rewrites "
        "the model field in the request body before forwarding. None = passthrough.",
    )
    fast_model: str | None = Field(
        default=None,
        description="Override model for 'fast/small' upstream requests (e.g. haiku). "
        "When set, requests for known small models are rewritten to this. None = use 'model'.",
    )
    max_context_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Max context tokens for the upstream model. When set, the bridge "
        "logs a warning if the optimized request exceeds this limit. Advisory only.",
    )


class BridgeSessionConfig(BaseModel):
    """Session configuration — mirrors runtime SessionConfig pattern.

    The session name is defined here. All requests use this session unless the
    client sends an X-SR2-Session-ID header to override (enabling cross-client
    session sharing).
    """

    name: str = Field(
        default="default",
        description="Session name. All requests use this session unless the client "
        "sends an X-SR2-Session-ID header to override.",
    )
    idle_timeout_minutes: int = Field(
        default=120,
        ge=1,
        description="Minutes of inactivity before a session is cleaned up.",
    )
    persistence: bool = Field(
        default=False,
        description="Persist session state to SQLite. Survives bridge restarts.",
    )


class BridgeLLMModelConfig(BaseModel):
    """Config for a single bridge-internal LLM purpose."""

    model: str = Field(description="Model identifier (e.g. 'claude-haiku-4-5-20251001').")
    max_tokens: int = Field(default=1024, ge=1, description="Max tokens per response.")
    api_key: str | None = Field(
        default=None,
        description="Dedicated API key. When None, borrows from proxied request headers.",
    )
    api_base: str | None = Field(
        default=None,
        description="Override API base URL. When None, uses the bridge upstream_url.",
    )


class BridgeLLMConfig(BaseModel):
    """LLM config for bridge-internal calls. Each purpose is independently configurable."""

    extraction: BridgeLLMModelConfig | None = Field(
        default=None,
        description="Model for memory extraction (structured JSON output).",
    )
    summarization: BridgeLLMModelConfig | None = Field(
        default=None,
        description="Model for conversation summarization.",
    )
    embedding: BridgeLLMModelConfig | None = Field(
        default=None,
        description="Embedding model for semantic memory search.",
    )


class BridgeDegradationConfig(BaseModel):
    """Circuit breaker and degradation settings for bridge-internal stages."""

    circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        description="Consecutive failures before circuit breaker opens for a stage.",
    )
    circuit_breaker_cooldown_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Seconds before retrying after circuit breaker opens.",
    )


class BridgeMemoryConfig(BaseModel):
    """Memory system settings for the bridge."""

    enabled: bool = Field(default=False, description="Enable memory extraction and retrieval.")
    backend: Literal["sqlite", "postgres"] = Field(
        default="sqlite",
        description="Memory store backend. 'sqlite' for local, 'postgres' for shared (requires sr2-pro).",
    )
    db_path: str = Field(
        default="sr2_bridge_memory.db",
        description="Path to SQLite database (backend=sqlite only).",
    )
    database_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL (backend=postgres only). "
        "Example: postgresql://user:pass@host:5432/dbname",
    )
    max_memories_per_turn: int = Field(
        default=5,
        ge=1,
        description="Max memories to extract per conversation turn.",
    )
    retrieval_top_k: int = Field(
        default=10,
        ge=1,
        description="Max memories to retrieve per request.",
    )
    retrieval_max_tokens: int = Field(
        default=2000,
        ge=0,
        description="Max tokens for retrieved memory content.",
    )
    retrieval_strategy: Literal["hybrid", "keyword"] = Field(
        default="keyword",
        description="Retrieval strategy. 'keyword' needs no embeddings. "
        "'hybrid' combines keyword + semantic (requires embedding model config).",
    )


class BridgeLoggingConfig(BaseModel):
    """JSONL request/response payload logging."""

    enabled: bool = Field(default=False, description="Enable JSONL request/response logging.")
    output_path: str = Field(
        default="sr2_bridge_requests.jsonl",
        description="Path to JSONL log file.",
    )
    log_system_prompt: bool = Field(
        default=True,
        description="Include full system prompt text in log entries.",
    )
    log_messages: bool = Field(
        default=True,
        description="Include full message content (vs summary only).",
    )
    log_rebuilt_body: bool = Field(
        default=True,
        description="Log the final body sent upstream.",
    )
    max_content_length: int | None = Field(
        default=None,
        ge=1,
        description="Truncate content fields to N chars. None = no limit.",
    )


class BridgeSystemPromptConfig(BaseModel):
    """Config-driven system prompt transformation.

    Applies a transform to the client's system prompt before SR2 injection
    (summaries, memories). Default is no-op (prepend with no content).
    """

    transform: Literal["prepend", "append", "replace", "wrap"] = Field(
        default="prepend",
        description="How to apply custom content to the client's system prompt.",
    )
    content: str | None = Field(
        default=None,
        description="Inline custom system prompt content.",
    )
    content_file: str | None = Field(
        default=None,
        description="Path to file containing custom system prompt. "
        "'content' takes precedence if both are set.",
    )

    @model_validator(mode="after")
    def validate_content_source(self) -> Self:
        if self.transform != "prepend" and not self.content and not self.content_file:
            raise ValueError(
                f"system_prompt.content or content_file required for transform={self.transform!r}"
            )
        if self.transform == "wrap" and self.content and "{original}" not in self.content:
            raise ValueError("wrap transform requires {original} placeholder in content")
        return self

    @property
    def resolved_content(self) -> str | None:
        """Resolve content from inline or file source."""
        if self.content:
            return self.content
        if self.content_file:
            try:
                return Path(self.content_file).read_text()
            except FileNotFoundError:
                logger.warning("system_prompt.content_file not found: %s", self.content_file)
                return None
        return None


class BridgeConfig(BaseModel):
    """Top-level bridge server configuration."""

    port: int = Field(default=9200, ge=1, le=65535, description="Port to listen on.")
    host: str = Field(default="127.0.0.1", description="Host to bind to.")
    forwarding: BridgeForwardingConfig = Field(
        default_factory=BridgeForwardingConfig,
        description="Upstream forwarding settings.",
    )
    session: BridgeSessionConfig = Field(
        default_factory=BridgeSessionConfig,
        description="Session identification settings.",
    )
    llm: BridgeLLMConfig = Field(
        default_factory=BridgeLLMConfig,
        description="LLM config for bridge-internal calls (extraction, summarization, etc.).",
    )
    degradation: BridgeDegradationConfig = Field(
        default_factory=BridgeDegradationConfig,
        description="Circuit breaker and degradation settings.",
    )
    memory: BridgeMemoryConfig = Field(
        default_factory=BridgeMemoryConfig,
        description="Memory extraction and retrieval settings.",
    )
    logging: BridgeLoggingConfig = Field(
        default_factory=BridgeLoggingConfig,
        description="JSONL request/response payload logging.",
    )
    system_prompt: BridgeSystemPromptConfig = Field(
        default_factory=BridgeSystemPromptConfig,
        description="Config-driven system prompt transformation.",
    )
    tool_type_overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Custom tool name to content type mappings for compaction. "
            "Keys are substrings matched against tool names (case-insensitive), "
            "values are content types: file_content, code_execution, tool_output."
        ),
    )
    allowed_passthrough_paths: list[str] = Field(
        default=["/v1/messages/count_tokens", "/v1/messages/batches"],
        description=(
            "API paths that are forwarded to upstream without optimization. "
            "Paths not in this list (and not /v1/messages, /health, /metrics) return 404."
        ),
    )
