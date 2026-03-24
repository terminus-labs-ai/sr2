"""Pydantic config models for SR2 Bridge."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class BridgeSessionConfig(BaseModel):
    """Session identification settings."""

    strategy: Literal["system_hash", "header", "single"] = Field(
        default="system_hash",
        description=(
            "How to identify sessions. "
            "'system_hash' hashes the system prompt (works with Claude Code). "
            "'header' uses X-SR2-Session-ID header. "
            "'single' treats all requests as one session."
        ),
    )
    idle_timeout_minutes: int = Field(
        default=120,
        ge=1,
        description="Minutes of inactivity before a session is cleaned up.",
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
    intent: BridgeLLMModelConfig | None = Field(
        default=None,
        description="Model for intent/topic-shift detection.",
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
    db_path: str = Field(
        default="sr2_bridge_memory.db",
        description="Path to SQLite database for memory persistence.",
    )
    max_memories_per_turn: int = Field(
        default=5, ge=1, description="Max memories to extract per conversation turn.",
    )
    retrieval_top_k: int = Field(
        default=10, ge=1, description="Max memories to retrieve per request.",
    )
    retrieval_max_tokens: int = Field(
        default=2000, ge=0, description="Max tokens for retrieved memory content.",
    )
    retrieval_strategy: Literal["hybrid", "keyword"] = Field(
        default="keyword",
        description="Retrieval strategy. 'keyword' needs no embeddings. "
        "'hybrid' combines keyword + semantic (requires embedding model config).",
    )


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
    allowed_passthrough_paths: list[str] = Field(
        default=["/v1/messages/count_tokens", "/v1/messages/batches"],
        description=(
            "API paths that are forwarded to upstream without optimization. "
            "Paths not in this list (and not /v1/messages, /health, /metrics) return 404."
        ),
    )
