"""Tests for MetricsManager — extracted metrics/observability logic from SR2 facade.

Tests the public interface of MetricsManager and MetricSources, which will live
at sr2.metrics.manager. These tests define the extraction boundary: MetricsManager
owns all session tracking state and metrics methods that were previously on SR2.

Implementation notes for Agent C:
- MetricsManager lives at src/sr2/metrics/manager.py.
- MetricSources is a dataclass in the same file — holds read-only references to
  components that provide metric data (conversation, engine, post_processor,
  retriever, memory_store, trace).
- Constructor: MetricsManager(collector, sources, token_budget, alerts=None,
  push_exporters=[], pull_exporter_name=None).
- All session tracking state (_session_start_times, _session_turn_counts,
  _token_savings_cumulative, _last_actual_usage, _actual_input_tokens_history,
  _estimate_drift_history, _last_compiled_tokens, _max_usage_history) moves
  from SR2 to MetricsManager.
- collect() is async and contains the ~226-line metrics collection logic
  currently in SR2.collect_metrics().
- report_actual_usage(), estimate_drift(), export_metrics(), record_compiled_tokens()
  move from SR2 to MetricsManager.
- SR2 should delegate its collect_metrics(), report_actual_usage(), estimate_drift(),
  and export_metrics() methods to its MetricsManager instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

from sr2.metrics.collector import MetricCollector
from sr2.pipeline.result import ActualTokenUsage, PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_collector() -> MetricCollector:
    return MetricCollector(agent_name="test-agent")


def _make_usage(
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cached_tokens: int = 0,
) -> ActualTokenUsage:
    return ActualTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


def _make_pipeline_result(total_tokens: int = 1000) -> PipelineResult:
    return PipelineResult(total_tokens=total_tokens)


def _make_mock_sources() -> Any:
    """Build a mock MetricSources with all required fields stubbed."""
    from sr2.metrics.manager import MetricSources

    # Conversation mock
    conversation = MagicMock()
    conversation.get_raw_window_utilization = MagicMock(return_value=0.5)
    zones = MagicMock()
    zones.raw = []
    zones.compacted = []
    zones.summarized = []
    zones.total_tokens = 0
    conversation.zones = MagicMock(return_value=zones)
    conversation.get_zone_transitions = MagicMock(return_value={})

    # Engine mock
    engine = MagicMock()
    engine.truncation_events = 0
    cb = MagicMock()
    cb.status = MagicMock(return_value={})
    engine._circuit_breaker = cb

    # Post-processor mock
    post_processor = MagicMock()
    post_processor.last_memories_extracted = 0
    post_processor.last_conflicts_detected = 0
    post_processor.last_compaction_result = None
    post_processor.last_summarization_result = None

    # Retriever mock
    retriever = MagicMock()
    retriever._total_retrievals = 0

    # Memory store mock
    memory_store = MagicMock()
    memory_store.count = AsyncMock(return_value=0)

    # Trace mock (None = no tracing)
    trace = None

    return MetricSources(
        conversation=conversation,
        engine=engine,
        post_processor=post_processor,
        retriever=retriever,
        memory_store=memory_store,
        trace=trace,
    )


def _make_manager(**overrides) -> Any:
    """Build a MetricsManager with sensible defaults."""
    from sr2.metrics.manager import MetricsManager

    kwargs = {
        "collector": _make_collector(),
        "sources": _make_mock_sources(),
        "token_budget": 8000,
    }
    kwargs.update(overrides)
    return MetricsManager(**kwargs)


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


class TestImportability:
    """MetricsManager and MetricSources must be importable from sr2.metrics.manager."""

    def test_metrics_manager_importable(self):
        """MetricsManager is importable from sr2.metrics.manager."""
        from sr2.metrics.manager import MetricsManager

        assert MetricsManager is not None

    def test_metric_sources_importable(self):
        """MetricSources is importable from sr2.metrics.manager."""
        from sr2.metrics.manager import MetricSources

        assert MetricSources is not None


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """MetricsManager can be created with required args."""

    def test_construct_with_required_args(self):
        """MetricsManager(collector, sources, token_budget) creates an instance."""
        from sr2.metrics.manager import MetricsManager

        mgr = _make_manager()
        assert isinstance(mgr, MetricsManager)

    def test_construct_with_optional_args(self):
        """MetricsManager accepts optional alerts, push_exporters, pull_exporter_name."""
        from sr2.metrics.manager import MetricsManager

        mgr = _make_manager(
            alerts=MagicMock(),
            push_exporters=[MagicMock()],
            pull_exporter_name="prometheus",
        )
        assert isinstance(mgr, MetricsManager)


# ---------------------------------------------------------------------------
# 3. MetricSources fields
# ---------------------------------------------------------------------------


class TestMetricSourcesFields:
    """MetricSources dataclass has all required fields."""

    def test_has_conversation_field(self):
        """MetricSources has a 'conversation' field."""
        from sr2.metrics.manager import MetricSources

        sources = _make_mock_sources()
        assert hasattr(sources, "conversation")

    def test_has_engine_field(self):
        """MetricSources has an 'engine' field."""
        sources = _make_mock_sources()
        assert hasattr(sources, "engine")

    def test_has_post_processor_field(self):
        """MetricSources has a 'post_processor' field."""
        sources = _make_mock_sources()
        assert hasattr(sources, "post_processor")

    def test_has_retriever_field(self):
        """MetricSources has a 'retriever' field."""
        sources = _make_mock_sources()
        assert hasattr(sources, "retriever")

    def test_has_memory_store_field(self):
        """MetricSources has a 'memory_store' field."""
        sources = _make_mock_sources()
        assert hasattr(sources, "memory_store")

    def test_has_trace_field(self):
        """MetricSources has a 'trace' field."""
        sources = _make_mock_sources()
        assert hasattr(sources, "trace")

    def test_trace_can_be_none(self):
        """MetricSources.trace can be None (tracing disabled)."""
        sources = _make_mock_sources()
        assert sources.trace is None  # our helper sets it to None


# ---------------------------------------------------------------------------
# 4. report_actual_usage stores usage
# ---------------------------------------------------------------------------


class TestReportActualUsage:
    """report_actual_usage() stores data keyed by session_id."""

    def test_stores_usage_by_session_id(self):
        """Calling report_actual_usage stores the ActualTokenUsage for that session."""
        mgr = _make_manager()
        usage = _make_usage(input_tokens=1500)
        mgr.report_actual_usage(usage, session_id="s1")

        assert mgr._last_actual_usage.get("s1") is not None
        assert mgr._last_actual_usage["s1"].input_tokens == 1500

    def test_separate_sessions_independent(self):
        """Data for session A and session B are stored independently."""
        mgr = _make_manager()
        mgr.report_actual_usage(_make_usage(input_tokens=1000), session_id="a")
        mgr.report_actual_usage(_make_usage(input_tokens=5000), session_id="b")

        assert mgr._last_actual_usage["a"].input_tokens == 1000
        assert mgr._last_actual_usage["b"].input_tokens == 5000

    def test_latest_usage_overwrites_previous(self):
        """Calling report_actual_usage twice for the same session keeps latest."""
        mgr = _make_manager()
        mgr.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")
        mgr.report_actual_usage(_make_usage(input_tokens=2000), session_id="s1")

        assert mgr._last_actual_usage["s1"].input_tokens == 2000

    def test_tracks_input_token_history(self):
        """report_actual_usage appends to the input token history for the session."""
        mgr = _make_manager()
        mgr.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")
        mgr.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        history = mgr._actual_input_tokens_history.get("s1", [])
        assert len(history) == 2
        assert history[0] == 1000
        assert history[1] == 1200

    def test_input_token_history_truncated_at_max(self):
        """History is truncated to _max_usage_history entries."""
        mgr = _make_manager()
        max_h = mgr._max_usage_history

        for i in range(max_h + 10):
            mgr.report_actual_usage(_make_usage(input_tokens=i), session_id="s1")

        history = mgr._actual_input_tokens_history.get("s1", [])
        assert len(history) == max_h
        # Should keep the most recent entries
        assert history[-1] == max_h + 10 - 1

    def test_report_actual_usage_records_drift_entry(self):
        """report_actual_usage appends to drift history when compiled tokens exist."""
        mgr = _make_manager()
        mgr.record_compiled_tokens("s1", 1000)
        mgr.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        drift_history = mgr._estimate_drift_history.get("s1", [])
        assert len(drift_history) == 1
        expected = (1200 - 1000) / 1200
        assert drift_history[0] == pytest.approx(expected, rel=1e-3)

    def test_drift_history_truncated_at_max(self):
        """Drift history is truncated to _max_usage_history entries."""
        mgr = _make_manager()
        max_h = mgr._max_usage_history

        for i in range(max_h + 10):
            mgr.record_compiled_tokens("s1", 1000)
            mgr.report_actual_usage(
                _make_usage(input_tokens=1200 + i), session_id="s1"
            )

        drift_history = mgr._estimate_drift_history.get("s1", [])
        assert len(drift_history) == max_h


# ---------------------------------------------------------------------------
# 5. record_compiled_tokens stores tokens
# ---------------------------------------------------------------------------


class TestRecordCompiledTokens:
    """record_compiled_tokens() stores compiled token count by session_id."""

    def test_stores_by_session_id(self):
        """Calling record_compiled_tokens stores the value keyed by session_id."""
        mgr = _make_manager()
        mgr.record_compiled_tokens("s1", 4500)

        assert mgr._last_compiled_tokens.get("s1") == 4500

    def test_overwrites_on_subsequent_calls(self):
        """Latest call overwrites the previous value for the same session."""
        mgr = _make_manager()
        mgr.record_compiled_tokens("s1", 4500)
        mgr.record_compiled_tokens("s1", 5000)

        assert mgr._last_compiled_tokens["s1"] == 5000

    def test_separate_sessions_independent(self):
        """Tokens for session A and B are stored independently."""
        mgr = _make_manager()
        mgr.record_compiled_tokens("a", 1000)
        mgr.record_compiled_tokens("b", 2000)

        assert mgr._last_compiled_tokens["a"] == 1000
        assert mgr._last_compiled_tokens["b"] == 2000


# ---------------------------------------------------------------------------
# 6. estimate_drift returns None for unknown session
# ---------------------------------------------------------------------------


class TestEstimateDriftUnknown:
    """estimate_drift() returns None when no data exists for the session."""

    def test_returns_none_for_unknown_session(self):
        """No data = None."""
        mgr = _make_manager()
        assert mgr.estimate_drift("nonexistent") is None

    def test_returns_none_for_session_with_no_compiled_tokens(self):
        """If report_actual_usage was called but no compiled tokens were recorded,
        drift cannot be calculated — still returns None."""
        mgr = _make_manager()
        # Report usage without ever recording compiled tokens
        mgr.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")

        # No compiled tokens means no drift entries
        assert mgr.estimate_drift("s1") is None


# ---------------------------------------------------------------------------
# 7. estimate_drift returns float after usage reported
# ---------------------------------------------------------------------------


class TestEstimateDriftWithData:
    """estimate_drift() returns a float when both compiled tokens and actual usage exist."""

    def test_returns_float(self):
        """With compiled tokens + actual usage, drift is a float."""
        mgr = _make_manager()
        mgr.record_compiled_tokens("s1", 1000)
        mgr.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        drift = mgr.estimate_drift("s1")
        assert isinstance(drift, float)

    def test_drift_value_correct(self):
        """drift = (actual - estimated) / actual."""
        mgr = _make_manager()
        mgr.record_compiled_tokens("s1", 1000)
        mgr.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        drift = mgr.estimate_drift("s1")
        expected = (1200 - 1000) / 1200
        assert drift == pytest.approx(expected, rel=1e-3)

    def test_rolling_average_across_multiple_reports(self):
        """estimate_drift returns the mean of all drift entries for the session."""
        mgr = _make_manager()

        # Turn 1: estimated=1000, actual=1200 -> drift=(1200-1000)/1200
        mgr.record_compiled_tokens("s1", 1000)
        mgr.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        # Turn 2: estimated=900, actual=1000 -> drift=(1000-900)/1000
        mgr.record_compiled_tokens("s1", 900)
        mgr.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")

        drift = mgr.estimate_drift("s1")
        d1 = (1200 - 1000) / 1200
        d2 = (1000 - 900) / 1000
        expected = (d1 + d2) / 2
        assert drift == pytest.approx(expected, rel=1e-3)

    def test_drift_per_session_isolation(self):
        """Drift for session A is independent from session B."""
        mgr = _make_manager()

        mgr.record_compiled_tokens("a", 1000)
        mgr.report_actual_usage(_make_usage(input_tokens=1200), session_id="a")

        mgr.record_compiled_tokens("b", 500)
        mgr.report_actual_usage(_make_usage(input_tokens=1000), session_id="b")

        drift_a = mgr.estimate_drift("a")
        drift_b = mgr.estimate_drift("b")

        assert drift_a != drift_b
        assert drift_a == pytest.approx((1200 - 1000) / 1200, rel=1e-3)
        assert drift_b == pytest.approx((1000 - 500) / 1000, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. export_metrics returns string
# ---------------------------------------------------------------------------


class TestExportMetrics:
    """export_metrics() returns a non-empty string (Prometheus format)."""

    def _has_pull_exporter(self) -> bool:
        """Check if a pull exporter (e.g. prometheus) is available."""
        try:
            from sr2.metrics.registry import get_pull_exporter

            get_pull_exporter("prometheus")
            return True
        except ImportError:
            return False

    def test_returns_string(self):
        """export_metrics() returns a str."""
        if not self._has_pull_exporter():
            pytest.skip("No pull exporter installed (sr2-pro required)")

        mgr = _make_manager(pull_exporter_name="prometheus")

        # Need at least one snapshot for non-empty output
        mgr._collector.collect(_make_pipeline_result(), "test")

        result = mgr.export_metrics()
        assert isinstance(result, str)

    def test_returns_non_empty_after_collection(self):
        """After collecting at least one snapshot, export is non-empty."""
        if not self._has_pull_exporter():
            pytest.skip("No pull exporter installed (sr2-pro required)")

        mgr = _make_manager(pull_exporter_name="prometheus")
        mgr._collector.collect(_make_pipeline_result(), "test")

        result = mgr.export_metrics()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 9. collect is async and accepts pipeline_result
# ---------------------------------------------------------------------------


class TestCollectMethod:
    """collect() is async and accepts pipeline_result."""

    @pytest.mark.asyncio
    async def test_collect_is_async(self):
        """collect() is a coroutine function."""
        import inspect
        from sr2.metrics.manager import MetricsManager

        assert inspect.iscoroutinefunction(MetricsManager.collect)

    @pytest.mark.asyncio
    async def test_collect_accepts_pipeline_result(self):
        """collect() can be called with a pipeline_result and does not raise."""
        mgr = _make_manager()
        pipeline_result = _make_pipeline_result(total_tokens=1200)

        # Basic smoke test — should not raise
        await mgr.collect(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1200,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="test_session",
        )

    @pytest.mark.asyncio
    async def test_collect_stores_snapshot(self):
        """After collect(), the collector has a new snapshot."""
        mgr = _make_manager()
        pipeline_result = _make_pipeline_result(total_tokens=1200)

        await mgr.collect(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1200,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="test_session",
        )

        assert mgr._collector.last_snapshot is not None

    @pytest.mark.asyncio
    async def test_collect_includes_actual_usage_when_reported(self):
        """When report_actual_usage was called, collect() includes actual token metrics."""
        from sr2.metrics.definitions import MetricNames

        mgr = _make_manager()
        mgr.report_actual_usage(
            _make_usage(input_tokens=1500, output_tokens=300, cached_tokens=600),
            session_id="s1",
        )

        await mgr.collect(
            pipeline_result=_make_pipeline_result(total_tokens=1500),
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1500,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="s1",
        )

        snapshot = mgr._collector.last_snapshot
        metric_names = {m.name for m in snapshot.metrics}
        assert MetricNames.ACTUAL_INPUT_TOKENS in metric_names
        assert MetricNames.ACTUAL_OUTPUT_TOKENS in metric_names
        assert MetricNames.ACTUAL_CACHED_TOKENS in metric_names


# ---------------------------------------------------------------------------
# 10. Session tracking state is on MetricsManager
# ---------------------------------------------------------------------------


class TestSessionTrackingState:
    """All session tracking state lives on MetricsManager, not SR2."""

    def test_has_session_start_times(self):
        """MetricsManager has _session_start_times dict."""
        mgr = _make_manager()
        assert isinstance(mgr._session_start_times, dict)

    def test_has_session_turn_counts(self):
        """MetricsManager has _session_turn_counts dict."""
        mgr = _make_manager()
        assert isinstance(mgr._session_turn_counts, dict)

    def test_has_token_savings_cumulative(self):
        """MetricsManager has _token_savings_cumulative dict."""
        mgr = _make_manager()
        assert isinstance(mgr._token_savings_cumulative, dict)

    def test_has_last_actual_usage(self):
        """MetricsManager has _last_actual_usage dict."""
        mgr = _make_manager()
        assert isinstance(mgr._last_actual_usage, dict)

    def test_has_actual_input_tokens_history(self):
        """MetricsManager has _actual_input_tokens_history dict."""
        mgr = _make_manager()
        assert isinstance(mgr._actual_input_tokens_history, dict)

    def test_has_estimate_drift_history(self):
        """MetricsManager has _estimate_drift_history dict."""
        mgr = _make_manager()
        assert isinstance(mgr._estimate_drift_history, dict)

    def test_has_last_compiled_tokens(self):
        """MetricsManager has _last_compiled_tokens dict."""
        mgr = _make_manager()
        assert isinstance(mgr._last_compiled_tokens, dict)

    def test_has_max_usage_history(self):
        """MetricsManager has _max_usage_history int."""
        mgr = _make_manager()
        assert isinstance(mgr._max_usage_history, int)
        assert mgr._max_usage_history == 50


# ---------------------------------------------------------------------------
# 11. SR2 delegates to MetricsManager
# ---------------------------------------------------------------------------


class TestSR2Delegation:
    """After refactoring, SR2 should delegate metrics methods to MetricsManager."""

    def _make_sr2(self) -> Any:
        """Build a minimal SR2 instance with no file I/O."""
        from sr2.config.models import (
            MemoryConfig,
            PipelineConfig,
            RetrievalConfig,
            SummarizationConfig,
        )
        from sr2.sr2 import SR2, SR2Config

        config = PipelineConfig(
            token_budget=8000,
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )
        return SR2(
            SR2Config(
                config_dir="/tmp",
                agent_yaml={},
                preloaded_config=config,
            )
        )

    def test_sr2_has_metrics_manager(self):
        """SR2 instance should have a _metrics_manager attribute."""
        sr2 = self._make_sr2()
        assert hasattr(sr2, "_metrics_manager"), (
            "SR2 should have a _metrics_manager attribute after extraction"
        )

    def test_sr2_report_actual_usage_delegates(self):
        """SR2.report_actual_usage() should delegate to MetricsManager."""
        sr2 = self._make_sr2()
        usage = _make_usage(input_tokens=1500)
        sr2.report_actual_usage(usage, session_id="s1")

        # The MetricsManager should have the stored usage
        assert sr2._metrics_manager._last_actual_usage.get("s1") is not None
        assert sr2._metrics_manager._last_actual_usage["s1"].input_tokens == 1500

    def test_sr2_estimate_drift_delegates(self):
        """SR2.estimate_drift() should delegate to MetricsManager."""
        sr2 = self._make_sr2()

        # No data yet
        result = sr2.estimate_drift("s1")
        assert result is None

        # After providing data, should return float
        sr2._metrics_manager.record_compiled_tokens("s1", 1000)
        sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        drift = sr2.estimate_drift("s1")
        assert isinstance(drift, float)

    @pytest.mark.asyncio
    async def test_sr2_collect_metrics_delegates(self):
        """SR2.collect_metrics() should delegate to MetricsManager.collect()."""
        sr2 = self._make_sr2()
        pipeline_result = _make_pipeline_result(total_tokens=1200)

        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1200,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="test_session",
        )

        # MetricsManager's collector should have a snapshot
        assert sr2._metrics_manager._collector.last_snapshot is not None

    def test_sr2_session_tracking_on_metrics_manager(self):
        """Session tracking state should live on MetricsManager, not directly on SR2."""
        sr2 = self._make_sr2()

        # These should NOT exist directly on SR2 after extraction
        assert not hasattr(sr2, "_session_start_times"), (
            "_session_start_times should be on _metrics_manager, not SR2"
        )
        assert not hasattr(sr2, "_session_turn_counts"), (
            "_session_turn_counts should be on _metrics_manager, not SR2"
        )
        assert not hasattr(sr2, "_token_savings_cumulative"), (
            "_token_savings_cumulative should be on _metrics_manager, not SR2"
        )
        assert not hasattr(sr2, "_last_actual_usage"), (
            "_last_actual_usage should be on _metrics_manager, not SR2"
        )
        assert not hasattr(sr2, "_actual_input_tokens_history"), (
            "_actual_input_tokens_history should be on _metrics_manager, not SR2"
        )
        assert not hasattr(sr2, "_estimate_drift_history"), (
            "_estimate_drift_history should be on _metrics_manager, not SR2"
        )
        assert not hasattr(sr2, "_last_compiled_tokens"), (
            "_last_compiled_tokens should be on _metrics_manager, not SR2"
        )
