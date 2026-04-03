"""Tests for pipeline metric collector."""

from sr2.metrics.collector import MetricCollector
from sr2.metrics.definitions import MetricNames
from sr2.pipeline.result import PipelineResult, StageResult


def _make_result(
    total_tokens: int = 100,
    total_duration_ms: float = 50.0,
    cache_hit_rate: float | None = None,
    stages: list[StageResult] | None = None,
) -> PipelineResult:
    result = PipelineResult(
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
        cache_hit_rate=cache_hit_rate,
    )
    if stages:
        result.stages = stages
    return result


class TestMetricCollector:

    def test_collect_extracts_tokens_and_duration(self):
        """collect() extracts pipeline tokens and duration."""
        collector = MetricCollector(agent_name="agent")
        result = _make_result(total_tokens=200, total_duration_ms=75.0)
        snap = collector.collect(result, "user_message")

        tokens = snap.get(MetricNames.PIPELINE_TOTAL_TOKENS)
        duration = snap.get(MetricNames.PIPELINE_TOTAL_DURATION_MS)
        assert tokens.value == 200
        assert duration.value == 75.0

    def test_collect_extracts_per_stage_metrics(self):
        """collect() extracts per-stage metrics."""
        stages = [
            StageResult(stage_name="system", status="success", tokens_used=50, duration_ms=10.0),
            StageResult(stage_name="memory", status="success", tokens_used=30, duration_ms=20.0),
        ]
        collector = MetricCollector(agent_name="agent")
        snap = collector.collect(_make_result(stages=stages), "api")

        # Should have stage metrics
        stage_metrics = [m for m in snap.metrics if m.name == MetricNames.STAGE_TOKENS]
        assert len(stage_metrics) == 2

    def test_collect_tracks_fallback_events(self):
        """collect() tracks fallback events."""
        stages = [
            StageResult(
                stage_name="retrieval", status="degraded",
                fallback_used=True, tokens_used=10, duration_ms=5.0,
            ),
        ]
        collector = MetricCollector(agent_name="agent")
        snap = collector.collect(_make_result(stages=stages), "api")

        fallback = snap.get(MetricNames.FALLBACK_RATE)
        assert fallback is not None
        assert fallback.value == 1.0

    def test_collect_adds_extra_metrics(self):
        """collect() adds extra_metrics."""
        collector = MetricCollector(agent_name="agent")
        snap = collector.collect(
            _make_result(),
            "api",
            extra_metrics={"custom_metric": 0.95},
        )

        custom = snap.get("custom_metric")
        assert custom is not None
        assert custom.value == 0.95

    def test_get_latest_returns_correct_count(self):
        """get_latest(n) returns correct count."""
        collector = MetricCollector(agent_name="agent")
        for _ in range(5):
            collector.collect(_make_result(), "api")

        assert len(collector.get_latest(3)) == 3
        assert len(collector.get_latest(10)) == 5

    def test_get_metric_history_chronological(self):
        """get_metric_history() returns values in chronological order."""
        collector = MetricCollector(agent_name="agent")
        for tokens in [10, 20, 30]:
            collector.collect(_make_result(total_tokens=tokens), "api")

        history = collector.get_metric_history(MetricNames.PIPELINE_TOTAL_TOKENS)
        assert history == [10, 20, 30]

    def test_get_average_computes_correctly(self):
        """get_average() computes correctly."""
        collector = MetricCollector(agent_name="agent")
        for tokens in [10, 20, 30]:
            collector.collect(_make_result(total_tokens=tokens), "api")

        avg = collector.get_average(MetricNames.PIPELINE_TOTAL_TOKENS)
        assert avg == 20.0

    def test_get_average_returns_none_for_unknown(self):
        """get_average() returns None for unknown metric."""
        collector = MetricCollector(agent_name="agent")
        assert collector.get_average("nonexistent") is None

    def test_history_capped_at_max(self):
        """History capped at max_history."""
        collector = MetricCollector(agent_name="agent", max_history=3)
        for _ in range(5):
            collector.collect(_make_result(), "api")

        assert len(collector.snapshots) == 3

    def test_clear_empties_snapshots(self):
        """clear() empties all snapshots."""
        collector = MetricCollector(agent_name="agent")
        collector.collect(_make_result(), "api")
        collector.clear()

        assert len(collector.snapshots) == 0
