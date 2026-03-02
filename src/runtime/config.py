"""Pydantic models for runtime config (agent.yaml)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelParams(BaseModel):
    """LLM sampling/generation parameters passed to the provider API."""

    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature (0=deterministic, 2=max randomness).",
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Nucleus sampling threshold.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling. Not supported by all providers.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2,
        le=2,
        description="Penalize frequent tokens (-2 to 2).",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2,
        le=2,
        description="Penalize already-used tokens (-2 to 2).",
    )
    stop: list[str] | None = Field(
        default=None,
        description="Stop sequences.",
    )

    def to_api_kwargs(self) -> dict[str, Any]:
        """Return only non-None fields as a dict for LiteLLM kwargs."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class RuntimeDatabaseConfig(BaseModel):
    """Database connection settings."""

    url: str | None = Field(
        default=None,
        description="PostgreSQL connection string. Supports ${VAR} env var substitution.",
    )
    pool_min: int = Field(default=2, ge=1, description="Minimum connection pool size.")
    pool_max: int = Field(default=10, ge=1, description="Maximum connection pool size.")


class LLMModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    name: str = Field(description="Model identifier.")
    api_base: str | None = Field(default=None, description="API base URL.")
    max_tokens: int = Field(default=4096, ge=1, description="Max tokens per response.")
    stream: bool = Field(default=False, description="Enable streaming for this model.")
    model_params: ModelParams = Field(
        default_factory=ModelParams, description="Sampling parameters."
    )


class RuntimeLLMConfig(BaseModel):
    """LLM model and endpoint settings."""

    model: LLMModelConfig = Field(
        default_factory=lambda: LLMModelConfig(name="claude-sonnet-4-20250514"),
        description="Main LLM model configuration.",
    )
    fast_model: LLMModelConfig = Field(
        default_factory=lambda: LLMModelConfig(name="claude-haiku-4-5-20251001", max_tokens=1000),
        description="Fast model for extraction, summarization, intent detection.",
    )
    embedding: LLMModelConfig = Field(
        default_factory=lambda: LLMModelConfig(name="text-embedding-3-small"),
        description="Embedding model for memory retrieval.",
    )


class RuntimeLoopConfig(BaseModel):
    """LLM loop settings."""

    max_iterations: int = Field(
        default=25, ge=1, description="Max tool-call loop iterations before stopping."
    )


class RuntimeSessionConfig(BaseModel):
    """Default session settings for sessions not explicitly configured."""

    max_turns: int = Field(
        default=200,
        ge=1,
        description="Default max turns for sessions not explicitly configured.",
    )
    idle_timeout_minutes: int = Field(
        default=60,
        ge=1,
        description="Default idle timeout in minutes before session cleanup.",
    )


class InterfaceSessionConfig(BaseModel):
    """Session config for an interface."""

    name: str = Field(
        description="Session name. Use {request.session_id} for dynamic names.",
    )
    lifecycle: Literal["persistent", "ephemeral", "rolling"] = Field(
        default="persistent", description="Session lifecycle policy."
    )


class InterfaceConfig(BaseModel):
    """Interface plugin definition."""

    plugin: str = Field(description="Plugin type: telegram | timer | http | a2a")
    session: InterfaceSessionConfig | None = Field(
        default=None, description="Session config for this interface."
    )
    pipeline: str | None = Field(
        default=None,
        description="Path to pipeline config (relative to config_dir).",
    )

    model_config = {"extra": "allow"}


class MCPResourceConfig(BaseModel):
    """Config for auto-loading an MCP resource into the pipeline."""

    uri: str = Field(description="Resource URI to read.")
    subscribe: bool = Field(
        default=False, description="Subscribe to change notifications for this resource."
    )


class MCPPromptConfig(BaseModel):
    """Config for an MCP prompt to auto-load."""

    name: str = Field(description="Prompt name on the server.")
    arguments: dict[str, str] = Field(
        default_factory=dict, description="Arguments to fill the prompt template."
    )


class MCPSamplingConfig(BaseModel):
    """Sampling policy for MCP server-initiated LLM requests."""

    enabled: bool = Field(default=False, description="Enable sampling requests from this server.")
    policy: Literal["auto_approve", "log_only", "deny"] = Field(
        default="log_only",
        description="Approval policy. 'auto_approve' runs requests silently. "
        "'log_only' logs and runs. 'deny' rejects all sampling requests.",
    )
    max_tokens: int = Field(default=1024, ge=1, description="Max tokens per sampling request.")
    rate_limit_per_minute: int = Field(
        default=10, ge=1, description="Max sampling requests per minute per server."
    )


class MCPServerConfig(BaseModel):
    """MCP server connection config - mcp_servers"""

    name: str = Field(description="Server name for logging.")
    url: str = Field(description="Command (stdio) or URL (http).")
    transport: Literal["stdio", "http", "sse"] = Field(
        default="stdio", description="Transport protocol."
    )
    tools: list[str] | None = Field(
        default=None, description="Curated tool list. None = all tools."
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP headers for http/sse transport (e.g. Authorization). "
        "Supports ${VAR} env var substitution.",
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables for the server process."
    )
    args: list[str] | None = Field(default=None, description="Additional args for stdio transport.")
    roots: list[str] | None = Field(
        default=None,
        description="Root URIs to advertise to the server (e.g. 'file:///home/user/project'). "
        "Supports ${VAR} env var substitution.",
    )
    resources: list[MCPResourceConfig] | None = Field(
        default=None, description="Resources to auto-discover from this server."
    )
    expose_resources_as_tools: bool = Field(
        default=False,
        description="Register mcp_list_resources and mcp_read_resource as agent tools.",
    )
    prompts: list[MCPPromptConfig] | None = Field(
        default=None, description="Prompts to auto-load from this server."
    )
    expose_prompts_as_tools: bool = Field(
        default=False, description="Register mcp_get_prompt as an agent tool."
    )
    sampling: MCPSamplingConfig = Field(
        default_factory=MCPSamplingConfig,
        description="Sampling policy for server-initiated LLM requests.",
    )


class StreamContentConfig(BaseModel):
    """Controls what gets streamed beyond text deltas."""

    tool_status: bool = Field(default=True, description="Stream tool invocation status.")
    tool_results: bool = Field(default=False, description="Stream tool result content.")


class RuntimeConfig(BaseModel):
    """Runtime settings."""

    database: RuntimeDatabaseConfig = Field(
        default_factory=RuntimeDatabaseConfig,
        description="Database connection settings.",
    )
    llm: RuntimeLLMConfig = Field(
        default_factory=RuntimeLLMConfig,
        description="LLM model and endpoint settings.",
    )
    loop: RuntimeLoopConfig = Field(
        default_factory=RuntimeLoopConfig, description="LLM loop settings."
    )
    session: RuntimeSessionConfig = Field(
        default_factory=RuntimeSessionConfig, description="Default session settings."
    )
    stream_content: StreamContentConfig = Field(
        default_factory=StreamContentConfig,
        description="Controls what gets streamed beyond text deltas.",
    )


class AgentYAMLConfig(BaseModel):
    """Full agent.yaml model — combines pipeline config + runtime config."""

    agent_name: str | None = Field(default=None, description="Agent display name.")
    extends: str | None = Field(default=None, description="Parent config to inherit from.")
    system_prompt: str = Field(default="", description="System prompt for the LLM.")
    pipeline: dict = Field(default_factory=dict, description="Pipeline configuration.")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig, description="Runtime settings.")
    interfaces: dict[str, InterfaceConfig] = Field(
        default_factory=dict, description="Interface plugin definitions."
    )
    sessions: dict[str, dict] = Field(
        default_factory=dict, description="Named session configurations."
    )
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list, description="MCP server connections."
    )

    model_config = {"extra": "ignore"}
