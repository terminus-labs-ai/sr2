"""Pydantic models for SR2 v2 YAML configuration.

These models define the declarative API — users describe what they want,
SR2 executes it. Each model maps directly to the layer model in the
redesign plan.

Design principles:
- DRY: Shared sub-models reused across layer configs.
- OCP: New provider/reducer types added via plugins, not config changes.
- SRP: Each model represents one concept (layer, memory, circuit breaker).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from sr2.core.models import CachePolicy


# --- Layer Config ---


class MemoryReadConfig(BaseModel):
    """Memory retrieval configuration within a layer."""
    scope: list[str] = Field(default=["private"], description="Memory scopes to read from")
    max_tokens: int | None = Field(default=None, description="Token budget for retrieved memories")
    max_per_turn: int | None = Field(default=None, description="Max memories per conversation turn")


class MemoryWriteConfig(BaseModel):
    """Memory writing configuration within a layer."""
    scope: list[str] = Field(default=["private"], description="Memory scopes to write to")


class MemoryConfig(BaseModel):
    """Full memory configuration for a layer (read and write are independent)."""
    read: MemoryReadConfig | bool | None = Field(default=None, description="Read config, False to disable")
    write: MemoryWriteConfig | bool | None = Field(default=None, description="Write config, False to disable")


class CompactionConfig(BaseModel):
    """Compaction rules configuration."""
    rules: list[str] = Field(default=["schema_and_sample", "result_summary"], description="Compaction rules to apply")


class SummarizationConfig(BaseModel):
    """Summarization configuration, including cross-layer references."""
    scope: list[str] = Field(default=[], description="Layer names this summarization covers (cross-layer ref)")
    preserve: list[str] = Field(default=["decisions", "errors", "preferences"], description="What to preserve in summaries")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration for a content provider."""
    threshold: int = Field(default=3, description="Consecutive failures before opening")
    cooldown: int = Field(default=300, description="Seconds before retrying (half-open state)")


class ProviderConfig(BaseModel):
    """Base configuration for any content provider within a layer."""
    max_tokens: int | None = Field(default=None, description="Token budget for this provider")
    circuit_breaker: CircuitBreakerConfig | None = Field(default=None, description="Circuit breaker settings")
    fallback: str | None = Field(default=None, description="Fallback strategy: 'cached', 'none', 'static'")
    reducer: str | None = Field(default=None, description="Reducer plugin name to apply")
    extra: dict[str, Any] = Field(default_factory=dict, description="Provider-specific config")


# --- Layer Definition ---


class LayerConfig(BaseModel):
    """A single layer in the pipeline.

    A layer is a container with named content providers. Each provider
    has its own config and budget. The layer assembles them into a
    unified block in the final context.

    Properties:
        name: Unique layer identifier.
        max_tokens: Total token budget for this layer.
        cache: Cache policy (static, ephemeral, none).
        priority: Determines what gets cut when over total budget (higher = kept first).
        window: How many recent items to include (for session-based content).

    Content providers are declared as named keys:
        session_history, memory, compaction, summarization, tools, etc.
    """

    name: str = Field(..., description="Unique layer identifier")
    max_tokens: int | None = Field(default=None, description="Total token budget for this layer")
    cache: CachePolicy | str | None = Field(default=None, description="Cache policy")
    priority: int = Field(default=50, description="Priority for budget shedding (higher = kept first)")
    window: int | None = Field(default=None, description="Recent items window for session content")

    # Content providers - declared as named attributes
    # The pipeline engine resolves the key names dynamically via entry points
    session_history: ProviderConfig | None = Field(default=None)
    memory: MemoryConfig | None = Field(default=None)
    compaction: CompactionConfig | None = Field(default=None)
    summarization: SummarizationConfig | None = Field(default=None)
    tools: ProviderConfig | None = Field(default=None)

    # Allow arbitrary provider keys for plugin-defined providers
    extra: dict[str, Any] = Field(default_factory=dict)


# --- Pipeline Config ---


class PipelineConfig(BaseModel):
    """Complete SR2 pipeline configuration.

    This is the top-level config that the facade accepts.
    Users provide this as YAML, we parse it into this model.
    """

    layers: list[LayerConfig] = Field(default_factory=list, description="Ordered layer definitions")
    total_budget: int | None = Field(default=None, description="Total token budget across all layers")
    defaults: dict[str, Any] = Field(default_factory=dict, description="Default values for all layers")
