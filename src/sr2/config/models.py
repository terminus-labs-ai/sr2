from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from sr2.tools.models import ToolStateConfig, ToolTransitionConfig


class ContentItemConfig(BaseModel):
    key: str = Field(description="Unique key for this content item")
    source: str = Field(description="Content resolver name")
    max_tokens: int | None = Field(default=None, description="Max tokens for this item")
    optional: bool = Field(
        default=False, description="If True, skip without error if resolver fails"
    )
    model_config = {"extra": "allow"}


class LayerConfig(BaseModel):
    name: str = Field(description="Layer name")
    cache_policy: str = Field(default="immutable", description="Cache policy name")
    contents: list[ContentItemConfig] = Field(description="Content items in this layer")


class KVCacheConfig(BaseModel):
    strategy: Literal["append_only", "maximize_prefix_reuse", "no_cache_optimization"] = Field(
        default="append_only",
        description="KV-cache optimization strategy. 'append_only' keeps a stable prefix for "
        "cache reuse. 'maximize_prefix_reuse' reorders layers for maximum prefix hit. "
        "'no_cache_optimization' disables cache-aware ordering.",
    )
    compaction_timing: Literal["post_llm_async", "immediate", "disabled"] = Field(
        default="post_llm_async",
        description="When to run compaction. 'post_llm_async' runs after LLM response without "
        "blocking. 'immediate' runs synchronously before next turn. "
        "'disabled' turns off compaction entirely.",
    )
    summarization_timing: Literal["natural_breakpoint", "token_threshold", "disabled"] = Field(
        default="natural_breakpoint",
        description="When to trigger summarization. 'natural_breakpoint' summarizes at topic "
        "shifts or pauses. 'token_threshold' summarizes when token usage exceeds "
        "the configured threshold. 'disabled' turns off summarization.",
    )
    memory_refresh: Literal["on_topic_shift", "every_n_turns", "session_start_only", "disabled"] = (
        Field(
            default="on_topic_shift",
            description="When to refresh retrieved memories. 'on_topic_shift' refreshes when intent "
            "detection flags a topic change. 'every_n_turns' refreshes at a fixed interval. "
            "'session_start_only' loads memories once at session start. "
            "'disabled' turns off memory refresh.",
        )
    )
    memory_refresh_interval: int = Field(
        default=10, description="Only used if memory_refresh=every_n_turns"
    )


class CompactionRuleConfig(BaseModel):
    type: str = Field(description="Content type to match")
    strategy: str = Field(description="Compaction strategy name")
    max_compacted_tokens: int = Field(
        default=80,
        description="Maximum token count for compacted output. Content is truncated to fit "
        "within this budget. Only used by strategies that produce variable-length output "
        "(e.g., schema_and_sample).",
    )
    recovery_hint: bool = Field(
        default=False,
        description="If True, append a hint explaining how to re-fetch the original content "
        "(e.g., 'Re-fetch with [tool_name]'). Helps the agent recover compacted data "
        "when needed.",
    )
    model_config = {"extra": "allow"}


class CostGateConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable cost-aware compaction gating. When True, each compaction candidate "
        "is evaluated against cache invalidation costs before being compacted.",
    )
    fallback_model: str | None = Field(
        default=None,
        description="Fallback model name for LiteLLM pricing lookup when the request model "
        "is unknown (e.g., 'claude-sonnet-4-6').",
    )
    custom_pricing: dict[str, float] | None = Field(
        default=None,
        description="Custom per-token pricing in $/MTok. Expected keys: 'input', 'cache_write', "
        "'cache_read'. Takes priority over LiteLLM lookup.",
    )
    min_net_savings_usd: float = Field(
        default=0.001,
        ge=0.0,
        description="Minimum net dollar savings required for compaction to be allowed. "
        "Set to 0 to allow any cost-positive compaction.",
    )


class CompactionConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable compaction. When True, verbose tool outputs and file contents "
        "are replaced with compact references. Compacted content can be "
        "re-fetched by the agent using just-in-time retrieval tools.",
    )
    raw_window: int = Field(default=5, description="Keep last N turns in full detail")
    min_content_size: int = Field(default=100, description="Don't compact below this token count")
    cost_gate: CostGateConfig = Field(
        default_factory=CostGateConfig,
        description="Cache-cost-aware compaction gating. When enabled, each turn is evaluated "
        "against prompt caching economics before compaction is applied.",
    )
    rules: list[CompactionRuleConfig] = Field(
        default_factory=list,
        description="Compaction rules. Each rule matches a content_type and applies a strategy. "
        "Available strategies: schema_and_sample, reference, result_summary, supersede, collapse.",
    )


class SummarizationConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable automatic summarization. When True, older conversation history is "
        "summarized to reclaim token budget while preserving key information.",
    )
    trigger: Literal["token_threshold", "topic_shift", "manual"] = Field(
        default="token_threshold",
        description="What triggers summarization. 'token_threshold' triggers when token usage "
        "exceeds the threshold fraction. 'topic_shift' triggers on detected topic "
        "changes. 'manual' only triggers via explicit API call.",
    )
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    model: str = Field(default="fast", description="Model identifier for summarization")
    preserve_recent_turns: int = Field(
        default=3,
        description="Number of most recent turns to exclude from summarization. These turns "
        "are always kept in full detail regardless of token pressure.",
    )
    output_format: Literal["structured", "prose"] = Field(
        default="structured",
        description="Format for generated summaries. 'structured' produces categorized bullet "
        "points (decisions, issues, preferences). 'prose' produces a narrative paragraph.",
    )
    injection: Literal["flat", "selective"] = Field(
        default="flat",
        description="How summaries are injected into context. 'flat' inserts the full summary "
        "as a single block. 'selective' inserts only the sections relevant to the "
        "current topic (requires intent detection).",
    )
    preserve: list[str] = Field(
        default_factory=lambda: [
            "decisions_and_reasoning",
            "unresolved_issues",
            "user_preferences_expressed",
            "key_facts_and_data",
            "error_context",
        ],
        description="Categories of information to always preserve in summaries. These categories "
        "are extracted and retained even under heavy token pressure.",
    )
    discard: list[str] = Field(
        default_factory=lambda: [
            "successful_routine_actions",
            "redundant_confirmations",
            "exploration_dead_ends",
        ],
        description="Categories of information to discard during summarization. Content matching "
        "these categories is dropped to save tokens.",
    )
    compacted_max_tokens: int = Field(
        default=6000,
        ge=1000,
        description="Maximum token budget for the compacted zone before summarization triggers. "
        "Summarization fires when compacted_tokens > threshold * compacted_max_tokens. "
        "For heavy-traffic proxies (e.g. Claude Code), raise to 50000-100000.",
    )


class RetrievalConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable retrieval-augmented context. When True, relevant memories and "
        "documents are retrieved and injected into the context window.",
    )
    strategy: Literal["hybrid", "semantic", "keyword", "scoped"] = Field(
        default="hybrid",
        description="Retrieval strategy. 'hybrid' combines semantic and keyword search. "
        "'semantic' uses embedding similarity only. 'keyword' uses BM25/keyword "
        "matching. 'scoped' restricts retrieval to the current topic scope.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Maximum number of retrieved items to include in context.",
    )
    max_tokens: int = Field(
        default=4000,
        ge=0,
        description="Maximum total tokens allocated for retrieved content. Results are "
        "truncated to fit within this budget.",
    )


class IntentDetectionConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable intent detection. When True, user messages are classified to detect "
        "topic shifts, which can trigger memory refresh and selective summarization.",
    )
    model: str = Field(default="fast")


class ToolMaskingConfig(BaseModel):
    strategy: Literal["prefill", "allowed_list", "logit_mask", "none"] = Field(
        default="allowed_list",
        description="Tool masking strategy. 'allowed_list' only exposes allowed tools. "
        "'prefill' uses assistant prefill to guide tool selection. 'logit_mask' "
        "applies logit bias to suppress disallowed tools. 'none' exposes all tools.",
    )
    initial_state: str = Field(
        default="default",
        description="Initial tool state name. References a named state in the tool masking "
        "configuration that defines which tools are available at conversation start.",
    )


class DegradationConfig(BaseModel):
    circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        description="Number of consecutive failures before the circuit breaker opens. "
        "When open, requests are short-circuited to prevent cascading failures.",
    )
    circuit_breaker_cooldown_minutes: int = Field(
        default=5,
        ge=1,
        description="Minutes to wait before retrying after the circuit breaker opens. "
        "After this cooldown, the circuit enters half-open state and allows a trial request.",
    )


class LLMModelOverride(BaseModel):
    """Per-interface override for a single model. All fields optional."""

    name: str | None = Field(default=None, description="Model identifier override.")
    api_base: str | None = Field(default=None, description="API base URL override.")
    max_tokens: int | None = Field(default=None, description="Max tokens per response override.")
    model_params: dict[str, Any] | None = Field(
        default=None,
        description="Sampling parameter overrides (temperature, top_p, etc.).",
    )


class LLMConfig(BaseModel):
    """Per-interface LLM overrides. All fields optional — None means use agent default."""

    model: LLMModelOverride | None = Field(default=None, description="Main model override.")
    fast_model: LLMModelOverride | None = Field(default=None, description="Fast model override.")
    embedding: LLMModelOverride | None = Field(
        default=None, description="Embedding model override."
    )


class KeySchemaEntry(BaseModel):
    prefix: str = Field(description="Key prefix, e.g. 'user.preference'")
    description: str = Field(default="", description="Human-readable description of this prefix")
    examples: list[str] = Field(default_factory=list, description="Example keys for LLM guidance")
    model_config = {"extra": "allow"}


class MemoryScopeConfig(BaseModel):
    """Scope configuration for memory isolation and sharing."""

    allowed_read: list[str] = Field(
        default_factory=lambda: ["private"],
        description="Hard boundary — scopes this agent is permitted to read from.",
    )
    allowed_write: list[str] = Field(
        default_factory=lambda: ["private"],
        description="Hard boundary — scopes this agent is permitted to write to.",
    )
    default_read: str | None = Field(
        default=None,
        description="Single-scope fallback for reads. Inferred from first of allowed_read if omitted.",
    )
    default_write: str | None = Field(
        default=None,
        description="Default scope for newly saved/extracted memories. Inferred from first of allowed_write if omitted.",
    )
    agent_name: str | None = Field(
        default=None,
        description="Agent name for private memory isolation. Required for multi-agent setups.",
    )

    @model_validator(mode="after")
    def _resolve_and_validate(self):
        # Infer defaults from allowed lists when not explicitly set
        if self.default_read is None and self.allowed_read:
            self.default_read = self.allowed_read[0]
        if self.default_write is None and self.allowed_write:
            self.default_write = self.allowed_write[0]
        # Validate defaults are within allowed boundaries
        if self.default_read is not None and self.default_read not in self.allowed_read:
            raise ValueError(
                f"default_read '{self.default_read}' not in allowed_read {self.allowed_read}"
            )
        if self.default_write is not None and self.default_write not in self.allowed_write:
            raise ValueError(
                f"default_write '{self.default_write}' not in allowed_write {self.allowed_write}"
            )
        return self


class MemoryConfig(BaseModel):
    extract: bool = Field(
        default=True,
        description="Whether to automatically extract memories from conversation turns. "
        "When False, memories can still be saved explicitly via save_memory tool.",
    )
    scope: MemoryScopeConfig | None = Field(
        default=None,
        description="Scope configuration for memory isolation and sharing. "
        "When absent, all memories are accessible (backward compatible).",
    )
    key_schema: list[KeySchemaEntry] = Field(
        default_factory=list,
        description="Key prefix schema for memory extraction. Guides the LLM to use "
        "consistent, dot-notation keys.",
    )


class PipelineConfig(BaseModel):
    """Root config model. Represents a fully resolved interface config."""

    extends: str | None = Field(default=None, description="Parent config to inherit from")
    system_prompt: str | None = Field(
        default=None,
        description="Override the agent-level system prompt for this interface/pipeline.",
    )
    token_budget: int = Field(
        default=32000,
        ge=1000,
        description="Total token budget for the context window. The pipeline distributes this "
        "budget across layers. Higher values allow more context but increase cost "
        "and latency.",
    )
    pre_rot_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of token budget that triggers pre-emptive context rotation. "
        "When remaining budget drops below this fraction, summarization and "
        "compaction are triggered proactively to avoid hitting the hard limit.",
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="Per-interface LLM overrides")
    kv_cache: KVCacheConfig = Field(
        default_factory=KVCacheConfig, description="KV-cache optimization settings"
    )
    compaction: CompactionConfig = Field(
        default_factory=CompactionConfig, description="Content compaction settings"
    )
    summarization: SummarizationConfig = Field(
        default_factory=SummarizationConfig, description="Conversation summarization settings"
    )
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig, description="Retrieval-augmented context settings"
    )
    intent_detection: IntentDetectionConfig = Field(
        default_factory=IntentDetectionConfig,
        description="Intent detection and topic shift settings",
    )
    tool_masking: ToolMaskingConfig = Field(
        default_factory=ToolMaskingConfig, description="Tool visibility and masking settings"
    )
    tool_states: list[ToolStateConfig] = Field(
        default_factory=lambda: [ToolStateConfig(name="default")],
        description="Tool visibility states. Each state defines allowed/denied tools.",
    )
    tool_transitions: list[ToolTransitionConfig] = Field(
        default_factory=list,
        description="Rules for transitioning between tool states.",
    )
    tool_schema_max_tokens: int | None = Field(
        default=None,
        ge=100,
        description="Max tokens to allocate for tool schemas. If set, schemas are truncated "
        "to fit within this budget. None = no limit (all schemas sent as-is). "
        "Truncation strategy: drop descriptions, then drop parameters, then drop entire tools.",
    )
    degradation: DegradationConfig = Field(
        default_factory=DegradationConfig, description="Circuit breaker and degradation settings"
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory extraction settings"
    )
    layers: list[LayerConfig] = Field(
        default_factory=list,
        description="Context window layers, ordered from most stable to least stable",
    )
