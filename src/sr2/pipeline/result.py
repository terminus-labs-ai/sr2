from dataclasses import dataclass, field
from typing import Literal
import time


@dataclass
class StageResult:
    stage_name: str
    status: Literal["success", "degraded", "failed"]
    fallback_used: bool = False
    tokens_used: int = 0
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class PipelineResult:
    stages: list[StageResult] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    cache_hit_rate: float | None = None
    config_used: str = ""

    def add_stage(self, result: StageResult) -> None:
        """Add a stage result and update totals."""
        self.stages.append(result)
        self.total_tokens += result.tokens_used
        self.total_duration_ms += result.duration_ms

    @property
    def has_failures(self) -> bool:
        return any(s.status == "failed" for s in self.stages)

    @property
    def has_degradations(self) -> bool:
        return any(s.status == "degraded" for s in self.stages)

    @property
    def overall_status(self) -> Literal["success", "degraded", "failed"]:
        if self.has_failures:
            return "failed"
        if self.has_degradations:
            return "degraded"
        return "success"


class StageTimer:
    """Context manager for timing a pipeline stage."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self._start: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.duration_ms = (time.perf_counter() - self._start) * 1000

    def result(
        self,
        status: Literal["success", "degraded", "failed"],
        tokens: int = 0,
        fallback: bool = False,
        error: str | None = None,
    ) -> StageResult:
        return StageResult(
            stage_name=self.stage_name,
            status=status,
            fallback_used=fallback,
            tokens_used=tokens,
            duration_ms=self.duration_ms,
            error=error,
        )
