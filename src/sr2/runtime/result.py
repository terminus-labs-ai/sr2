"""Result types for SR2Runtime execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeMetrics:
    """Metrics from a single agent execution."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    compaction_events: int = 0
    cache_hit_rate: float = 0.0
    wall_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "compaction_events": self.compaction_events,
            "cache_hit_rate": self.cache_hit_rate,
            "wall_time_ms": self.wall_time_ms,
        }


@dataclass
class RuntimeResult:
    """Result from SR2Runtime.execute()."""

    output: str
    success: bool = True
    error: str | None = None
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
