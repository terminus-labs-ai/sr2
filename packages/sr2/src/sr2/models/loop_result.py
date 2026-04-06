"""Result types for LLM loop execution.

These live in sr2 (core) because they are shared by both sr2-runtime
(which produces them) and sr2-bridge (which also produces them via
execution adapters like Claude Code CLI).  Neither package should
import from the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    """Record of a single tool call during the loop."""

    tool_name: str
    arguments: dict
    result: str
    duration_ms: float
    success: bool
    error: str | None = None
    call_id: str = ""
    iteration: int = 0  # Which loop iteration this call belongs to


@dataclass
class LoopResult:
    """Result of a full LLM loop execution."""

    response_text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_tokens: int = 0
    stopped_reason: str = "complete"  # complete | max_iterations | error

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.total_input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.total_input_tokens
