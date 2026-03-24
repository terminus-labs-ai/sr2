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
    allowed_passthrough_paths: list[str] = Field(
        default=["/v1/messages/count_tokens", "/v1/messages/batches"],
        description=(
            "API paths that are forwarded to upstream without optimization. "
            "Paths not in this list (and not /v1/messages, /health, /metrics) return 404."
        ),
    )
