"""Pipeline result types.

CompiledContext is the output of process() — what the harness sends to the LLM.
PostProcessResult is the output of post_process() — memory actions, maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LayerResult:
    """Result from processing a single layer."""
    name: str
    content: str
    tokens: int
    cache_hit: bool = False
    providers: dict[str, int] = field(default_factory=dict)  # provider_name -> tokens
    shed: bool = False  # Was this layer shed due to budget pressure?


@dataclass
class PipelineMetrics:
    """Metrics snapshot from a process() call."""
    total_tokens: int = 0
    total_budget: int | None = None
    layers: dict[str, int] = field(default_factory=dict)  # layer_name -> tokens
    layers_shed: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    providers_failed: list[str] = field(default_factory=list)
    fallbacks_activated: list[str] = field(default_factory=list)
    reduction_savings: int = 0  # tokens saved by reducers
    memories_retrieved: int = 0
    memories_written: int = 0
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledContext:
    """Output of process() — ready-to-use context for the LLM call.

    Contains the assembled messages/blocks and metadata about
    what went in and why.
    """
    layers: list[LayerResult]
    metrics: PipelineMetrics
    total_tokens: int = 0

    def to_text(self) -> str:
        """Assemble all layer content into a single string."""
        return "\n\n".join(layer.content for layer in self.layers if layer.content)


@dataclass
class MaintenanceAction:
    """A maintenance action performed during post_process()."""
    kind: str  # "merge", "archive", "stale_check", "cleanup"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostProcessResult:
    """Output of post_process() — side effects and metrics."""
    memories_extracted: list[str] = field(default_factory=list)  # memory IDs
    maintenance_actions: list[MaintenanceAction] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
