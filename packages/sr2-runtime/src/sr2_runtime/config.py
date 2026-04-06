"""Pydantic models for runtime config (agent.yaml)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelParams(BaseModel):
    """LLM sampling/generation parameters passed to the provider API.

    Known fields are validated; unknown fields (e.g. thinking,
    max_thinking_tokens, min_p) are passed through to the provider API as-is.
    """

    model_config = ConfigDict(extra="allow")

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

    # Fields that LiteLLM accepts as top-level kwargs.
    # Everything else goes into ``extra_body`` so it reaches the API
    # without LiteLLM rejecting it as an unsupported parameter.
    _LITELLM_NATIVE_FIELDS: frozenset[str] = frozenset(
        {
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        }
    )

    def to_api_kwargs(self) -> dict[str, Any]:
        """Return only non-None fields as a dict for LiteLLM kwargs.

        Known LiteLLM params are top-level; provider-specific extras
        (e.g. ``thinking``, ``max_thinking_tokens``, ``num_batch``)
        go into ``extra_body`` so they pass through to the API untouched.
        """
        top_level: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        for k, v in self.model_dump().items():
            if v is None:
                continue
            if k in self._LITELLM_NATIVE_FIELDS:
                top_level[k] = v
            else:
                extra_body[k] = v
        if extra_body:
            top_level["extra_body"] = extra_body
        return top_level


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


class RuntimeBridgeConfig(BaseModel):
    """Bridge adapter configuration within the runtime.

    When ``adapter`` is set, the agent delegates LLM execution to a bridge
    adapter instead of using the built-in LLMLoop.  The adapter receives
    SR2's full compiled context and handles the agentic loop itself.
    """

    adapter: str | None = Field(
        default=None,
        description="Bridge adapter name (e.g. 'claude_code').  None = use built-in LLMLoop.",
    )
    claude_code: dict[str, Any] = Field(
        default_factory=dict,
        description="Claude Code adapter settings.  Validated by the adapter's config model "
        "at instantiation time via the adapter registry.",
    )


class STTProviderConfig(BaseModel):
    """Speech-to-text provider configuration.

    Uses an OpenAI-compatible STT API by default, which covers Whisper,
    Groq Whisper, Azure Speech, and any provider exposing a
    ``/v1/audio/transcriptions`` endpoint.
    """

    provider: str = Field(
        default="openai_compatible",
        description="STT provider type. 'openai_compatible' works with any "
        "OpenAI-compatible transcription API (Whisper, Groq, Deepgram, etc.).",
    )
    api_base: str | None = Field(
        default=None,
        description="STT API base URL. Supports ${VAR} env var substitution.",
    )
    model: str | None = Field(
        default=None,
        description="STT model identifier (e.g. 'Systran/faster-whisper-small', 'whisper-1').",
    )


class VoiceMediaConfig(BaseModel):
    """Voice and audio message processing.  Requires sr2-pro + an STT provider."""

    enabled: bool = Field(
        default=False,
        description="Accept and transcribe voice/audio messages.  Requires sr2-pro.",
    )
    stt: STTProviderConfig = Field(
        default_factory=STTProviderConfig,
        description="Speech-to-text provider for voice and audio messages.",
    )


class PhotoMediaConfig(BaseModel):
    """Photo message processing.  Requires sr2-pro + a vision-capable LLM."""

    enabled: bool = Field(
        default=False,
        description="Accept and process photo messages.  Requires sr2-pro.",
    )


class DocumentMediaConfig(BaseModel):
    """Document attachment processing.  Requires sr2-pro."""

    enabled: bool = Field(
        default=False,
        description="Accept and process document attachments.  Requires sr2-pro.",
    )


class MediaConfig(BaseModel):
    """Per-media-type processing toggles.

    Each media type can be independently enabled or disabled.  All media
    features require sr2-pro's ``MediaProcessor``.  When a type is disabled,
    the corresponding Telegram handler is not registered and incoming
    messages of that type are silently ignored.
    """

    voice: VoiceMediaConfig = Field(
        default_factory=VoiceMediaConfig,
        description="Voice and audio message processing config.",
    )
    photo: PhotoMediaConfig = Field(
        default_factory=PhotoMediaConfig,
        description="Photo message processing config.",
    )
    document: DocumentMediaConfig = Field(
        default_factory=DocumentMediaConfig,
        description="Document attachment processing config.",
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
        default=None,
        description="Environment variables for the server process. Supports ${VAR} env var substitution.",
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


class HeartbeatConfig(BaseModel):
    """Dynamic heartbeat scheduling settings."""

    model_config = {"extra": "ignore"}

    enabled: bool = Field(default=False, description="Enable heartbeat scheduling.")
    poll_interval_seconds: int = Field(
        default=30, ge=5, description="How often the scanner checks for due heartbeats."
    )
    max_context_turns: int = Field(
        default=10, ge=0, description="Max turns from source session to carry into heartbeat."
    )
    max_pending_per_agent: int = Field(
        default=100, ge=1, description="Max pending heartbeats per agent."
    )


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
    bridge: RuntimeBridgeConfig = Field(
        default_factory=RuntimeBridgeConfig,
        description="Bridge adapter configuration.  When set, delegates LLM execution "
        "to a bridge adapter instead of the built-in LLMLoop.",
    )
    session: RuntimeSessionConfig = Field(
        default_factory=RuntimeSessionConfig, description="Default session settings."
    )
    media: MediaConfig = Field(
        default_factory=MediaConfig,
        description="Multimedia processing (photos, documents, voice/audio). Requires sr2-pro.",
    )
    stream_content: StreamContentConfig = Field(
        default_factory=StreamContentConfig,
        description="Controls what gets streamed beyond text deltas.",
    )
    heartbeat: HeartbeatConfig = Field(
        default_factory=HeartbeatConfig,
        description="Dynamic heartbeat scheduling settings.",
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
