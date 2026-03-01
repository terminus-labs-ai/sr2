"""Pipeline metric collector — extracts metrics from PipelineResult."""

import uuid
from collections.abc import Callable

from sr2.metrics.definitions import MetricNames, MetricSnapshot
from sr2.pipeline.result import PipelineResult


class MetricCollector:
    """Collects metrics from pipeline invocations.

    Called after every pipeline run. Extracts metrics from PipelineResult
    and stores them for export.
    """

    def __init__(self, agent_name: str, max_history: int = 10000):
        self._agent_name = agent_name
        self._max_history = max_history
        self._snapshots: list[MetricSnapshot] = []
        self._on_collect_callbacks: list[Callable[[MetricSnapshot], None]] = []

    def on_collect(self, callback: Callable[[MetricSnapshot], None]) -> None:
        """Register a callback to fire on every collect() call."""
        self._on_collect_callbacks.append(callback)

    def collect(
        self,
        result: PipelineResult,
        interface_type: str,
        extra_metrics: dict[str, float] | None = None,
    ) -> MetricSnapshot:
        """Extract metrics from a PipelineResult and store.

        Returns the MetricSnapshot for immediate use.
        """
        snapshot = MetricSnapshot(
            invocation_id=f"inv_{uuid.uuid4().hex[:12]}",
            agent_name=self._agent_name,
            interface_type=interface_type,
        )

        # Pipeline-level metrics
        snapshot.add(MetricNames.PIPELINE_TOTAL_TOKENS, result.total_tokens, "tokens")
        snapshot.add(
            MetricNames.PIPELINE_TOTAL_DURATION_MS, result.total_duration_ms, "ms"
        )

        if result.cache_hit_rate is not None:
            snapshot.add(MetricNames.CACHE_HIT_RATE, result.cache_hit_rate, "ratio")

        # Per-stage metrics
        for stage in result.stages:
            snapshot.add(
                MetricNames.STAGE_DURATION_MS,
                stage.duration_ms,
                "ms",
                stage=stage.stage_name,
            )
            snapshot.add(
                MetricNames.STAGE_TOKENS,
                stage.tokens_used,
                "tokens",
                stage=stage.stage_name,
            )

            # Track fallbacks
            if stage.fallback_used:
                snapshot.add(
                    MetricNames.FALLBACK_RATE,
                    1.0,
                    "event",
                    stage=stage.stage_name,
                )

            # Track full degradation
            if stage.status == "failed":
                snapshot.add(MetricNames.FULL_DEGRADATION_EVENTS, 1.0, "event")

        # Extra metrics (e.g., retrieval precision, compaction ratio)
        if extra_metrics:
            for name, value in extra_metrics.items():
                snapshot.add(name, value)

        # Store
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_history:
            self._snapshots = self._snapshots[-self._max_history :]

        # Notify registered callbacks
        for cb in self._on_collect_callbacks:
            cb(snapshot)

        return snapshot

    @property
    def snapshots(self) -> list[MetricSnapshot]:
        return list(self._snapshots)

    def get_latest(self, n: int = 10) -> list[MetricSnapshot]:
        """Get the N most recent snapshots."""
        return self._snapshots[-n:]

    def get_metric_history(self, metric_name: str, n: int = 100) -> list[float]:
        """Get the last N values for a specific metric."""
        values = []
        for snapshot in reversed(self._snapshots):
            m = snapshot.get(metric_name)
            if m:
                values.append(m.value)
            if len(values) >= n:
                break
        values.reverse()
        return values

    def get_average(self, metric_name: str, n: int = 100) -> float | None:
        """Get the average of the last N values for a metric."""
        history = self.get_metric_history(metric_name, n)
        if not history:
            return None
        return sum(history) / len(history)

    def clear(self) -> None:
        """Clear all collected snapshots."""
        self._snapshots.clear()
