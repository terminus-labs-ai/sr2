"""Tests for Audit Fix 10: ActualTokenUsage feedback API.

Verifies that:
- ActualTokenUsage dataclass exists with the correct fields and computed properties.
- SR2.report_actual_usage() stores data per-session (not instance-level).
- Concurrent sessions do not overwrite each other.
- ActualTokenUsage is importable from the sr2 package public API.
- Metrics include actual token counts and drift after report_actual_usage() is called.
- Drift calculation is correct (estimate vs actual).
- Drift warning threshold fires at >15% drift.
- post_processor.set_actual_usage() is available.
- Backward compatibility: collect_metrics() still works if report_actual_usage() was never called.

Implementation notes for Agent C:
- ActualTokenUsage belongs in packages/sr2/src/sr2/pipeline/result.py.
- SR2.__init__ must initialise all token-tracking state as dicts keyed by session_id,
  NOT bare instance attributes, to avoid concurrent-session races.
- _last_actual_usage must be dict[str, ActualTokenUsage | None].
- _actual_input_tokens_history and _estimate_drift_history must be
  dict[str, list[...]] keyed by session_id.
- _last_compiled_tokens must also be dict[str, int | None] keyed by session_id.
- report_actual_usage() must accept session_id as a parameter and key all state on it.
- ActualTokenUsage must be re-exported from sr2.__init__ so it is importable as
  `from sr2 import ActualTokenUsage`.
- PostLLMProcessor.set_actual_usage() must exist and store the value for use during
  _run_compaction().
"""

from __future__ import annotations

import logging

import pytest

from sr2.config.models import MemoryConfig, PipelineConfig, RetrievalConfig, SummarizationConfig
from sr2.pipeline.result import ActualTokenUsage
from sr2.sr2 import SR2, SR2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sr2(token_budget: int = 8000) -> SR2:
    """Build a minimal SR2 instance with no file I/O."""
    config = PipelineConfig(
        token_budget=token_budget,
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


def _make_usage(
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cached_tokens: int = 0,
    loop_iterations: int = 1,
) -> ActualTokenUsage:
    return ActualTokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        loop_iterations=loop_iterations,
    )


# ---------------------------------------------------------------------------
# 1. ActualTokenUsage dataclass structure
# ---------------------------------------------------------------------------


class TestActualTokenUsageDataclass:
    """Verify the dataclass shape and computed properties."""

    def test_has_required_fields(self):
        """ActualTokenUsage must have input_tokens, output_tokens, cached_tokens, loop_iterations."""
        usage = ActualTokenUsage(
            input_tokens=1000,
            output_tokens=200,
            cached_tokens=400,
            loop_iterations=3,
        )
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 200
        assert usage.cached_tokens == 400
        assert usage.loop_iterations == 3

    def test_default_values_are_zero(self):
        """All fields default to 0."""
        usage = ActualTokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.loop_iterations == 0

    def test_total_tokens_property(self):
        """total_tokens == input_tokens + output_tokens."""
        usage = ActualTokenUsage(input_tokens=1000, output_tokens=300)
        assert usage.total_tokens == 1300

    def test_cache_hit_rate_property(self):
        """cache_hit_rate == cached_tokens / input_tokens."""
        usage = ActualTokenUsage(input_tokens=1000, cached_tokens=400)
        assert usage.cache_hit_rate == pytest.approx(0.4)

    def test_cache_hit_rate_zero_when_no_input(self):
        """cache_hit_rate returns 0.0 when input_tokens == 0 (no division by zero)."""
        usage = ActualTokenUsage(input_tokens=0, cached_tokens=0)
        assert usage.cache_hit_rate == 0.0

    def test_uncached_input_tokens_property(self):
        """uncached_input_tokens == max(0, input_tokens - cached_tokens)."""
        usage = ActualTokenUsage(input_tokens=1000, cached_tokens=400)
        assert usage.uncached_input_tokens == 600

    def test_uncached_input_tokens_never_negative(self):
        """uncached_input_tokens is clamped to 0 if cached > input (defensive)."""
        usage = ActualTokenUsage(input_tokens=100, cached_tokens=200)
        assert usage.uncached_input_tokens == 0


# ---------------------------------------------------------------------------
# 2. Public import from sr2 package
# ---------------------------------------------------------------------------


class TestActualTokenUsagePublicImport:
    """ActualTokenUsage must be importable from the top-level sr2 package."""

    def test_importable_from_sr2_package(self):
        """from sr2 import ActualTokenUsage must work."""
        from sr2 import ActualTokenUsage as ATU  # noqa: F401

        assert ATU is ActualTokenUsage

    def test_in_sr2_all(self):
        """ActualTokenUsage should appear in sr2.__all__."""
        import sr2

        assert "ActualTokenUsage" in sr2.__all__


# ---------------------------------------------------------------------------
# 3. SR2.report_actual_usage() API
# ---------------------------------------------------------------------------


class TestReportActualUsageMethod:
    """Verify report_actual_usage() exists and stores data per-session."""

    def test_method_exists(self):
        """SR2 must have a report_actual_usage() method."""
        sr2 = _make_sr2()
        assert hasattr(sr2, "report_actual_usage")
        assert callable(sr2.report_actual_usage)

    def test_accepts_actual_token_usage(self):
        """report_actual_usage() accepts an ActualTokenUsage instance without error."""
        sr2 = _make_sr2()
        usage = _make_usage(input_tokens=1500)
        sr2.report_actual_usage(usage, session_id="s1")  # Must not raise

    def test_session_id_stored_separately(self):
        """Data for session A and session B are stored independently."""
        sr2 = _make_sr2()

        usage_a = _make_usage(input_tokens=1000)
        usage_b = _make_usage(input_tokens=5000)

        sr2.report_actual_usage(usage_a, session_id="session_a")
        sr2.report_actual_usage(usage_b, session_id="session_b")

        # Retrieve stored values — implementation stores in _metrics_manager._last_actual_usage[session_id]
        stored_a = sr2._metrics_manager._last_actual_usage.get("session_a")
        stored_b = sr2._metrics_manager._last_actual_usage.get("session_b")

        assert stored_a is not None, "session_a usage should be stored"
        assert stored_b is not None, "session_b usage should be stored"
        assert stored_a.input_tokens == 1000
        assert stored_b.input_tokens == 5000

    def test_sessions_do_not_overwrite_each_other(self):
        """Reporting for session B must not change session A's stored usage."""
        sr2 = _make_sr2()

        sr2.report_actual_usage(_make_usage(input_tokens=1000), session_id="session_a")
        sr2.report_actual_usage(_make_usage(input_tokens=9999), session_id="session_b")

        # session_a still has 1000
        stored_a = sr2._metrics_manager._last_actual_usage.get("session_a")
        assert stored_a is not None
        assert stored_a.input_tokens == 1000

    def test_multiple_reports_for_same_session_update_latest(self):
        """Calling report_actual_usage() twice for the same session keeps latest."""
        sr2 = _make_sr2()

        sr2.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")
        sr2.report_actual_usage(_make_usage(input_tokens=2000), session_id="s1")

        stored = sr2._metrics_manager._last_actual_usage.get("s1")
        assert stored is not None
        assert stored.input_tokens == 2000

    def test_last_actual_usage_is_dict(self):
        """_last_actual_usage must be a dict (keyed by session_id), not a scalar."""
        sr2 = _make_sr2()
        assert isinstance(sr2._metrics_manager._last_actual_usage, dict), (
            "_last_actual_usage must be dict[str, ...], not a bare attribute"
        )


# ---------------------------------------------------------------------------
# 4. Drift history is per-session
# ---------------------------------------------------------------------------


class TestDriftHistoryPerSession:
    """Drift calibration history must be session-scoped."""

    def test_drift_history_is_dict(self):
        """_estimate_drift_history must be a dict keyed by session_id."""
        sr2 = _make_sr2()
        assert isinstance(sr2._metrics_manager._estimate_drift_history, dict), (
            "_estimate_drift_history must be dict[str, list[float]]"
        )

    def test_input_token_history_is_dict(self):
        """_actual_input_tokens_history must be a dict keyed by session_id."""
        sr2 = _make_sr2()
        assert isinstance(sr2._metrics_manager._actual_input_tokens_history, dict), (
            "_actual_input_tokens_history must be dict[str, list[int]]"
        )

    def test_drift_accumulates_per_session(self):
        """Multiple reports for the same session grow the drift history list."""
        sr2 = _make_sr2()

        # Seed _last_compiled_tokens for drift calculation
        sr2._metrics_manager._last_compiled_tokens = {"s1": 900}

        sr2.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")
        sr2.report_actual_usage(_make_usage(input_tokens=1100), session_id="s1")

        history = sr2._metrics_manager._estimate_drift_history.get("s1", [])
        assert len(history) == 2, f"Expected 2 drift entries, got {len(history)}"

    def test_drift_history_independent_between_sessions(self):
        """Reporting for session B does not add entries to session A's drift history."""
        sr2 = _make_sr2()

        sr2._metrics_manager._last_compiled_tokens = {"s1": 900, "s2": 900}

        sr2.report_actual_usage(_make_usage(input_tokens=1000), session_id="s1")
        sr2.report_actual_usage(_make_usage(input_tokens=1000), session_id="s2")
        sr2.report_actual_usage(_make_usage(input_tokens=1100), session_id="s2")

        history_s1 = sr2._metrics_manager._estimate_drift_history.get("s1", [])
        history_s2 = sr2._metrics_manager._estimate_drift_history.get("s2", [])

        assert len(history_s1) == 1, "s1 should have exactly 1 drift entry"
        assert len(history_s2) == 2, "s2 should have exactly 2 drift entries"


# ---------------------------------------------------------------------------
# 5. Drift calculation correctness
# ---------------------------------------------------------------------------


class TestDriftCalculation:
    """Verify the drift formula: (actual - estimated) / actual."""

    def test_drift_correct_value(self):
        """SR2 estimates 1000, provider reports 1200 -> drift ≈ 0.167."""
        sr2 = _make_sr2()
        sr2._metrics_manager._last_compiled_tokens = {"s1": 1000}

        sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        history = sr2._metrics_manager._estimate_drift_history.get("s1", [])
        assert len(history) == 1
        expected_drift = (1200 - 1000) / 1200
        assert history[0] == pytest.approx(expected_drift, rel=1e-3)

    def test_no_drift_when_no_compiled_tokens(self):
        """If _last_compiled_tokens has no entry for this session, no drift is recorded."""
        sr2 = _make_sr2()
        # Do NOT set _last_compiled_tokens for "s1"

        sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        history = sr2._metrics_manager._estimate_drift_history.get("s1", [])
        assert len(history) == 0, "No drift should be recorded if compiled tokens unknown"

    def test_no_drift_when_actual_input_is_zero(self):
        """Drift is not recorded when actual input_tokens == 0 (division by zero guard)."""
        sr2 = _make_sr2()
        sr2._metrics_manager._last_compiled_tokens = {"s1": 1000}

        sr2.report_actual_usage(_make_usage(input_tokens=0), session_id="s1")

        history = sr2._metrics_manager._estimate_drift_history.get("s1", [])
        assert len(history) == 0, "Drift must not be recorded when actual is 0"

    def test_rolling_drift_average(self):
        """estimate_drift(session_id) returns the mean of the drift history."""
        sr2 = _make_sr2()
        # Two turns: each with estimated=1000 and actual=1200
        sr2._metrics_manager._last_compiled_tokens = {"s1": 1000}
        sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")
        sr2._metrics_manager._last_compiled_tokens = {"s1": 1000}
        sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="s1")

        expected = (1200 - 1000) / 1200
        drift = sr2.estimate_drift("s1")
        assert drift is not None
        assert drift == pytest.approx(expected, rel=1e-3)

    def test_estimate_drift_returns_none_when_no_history(self):
        """estimate_drift(session_id) returns None when no data has been reported."""
        sr2 = _make_sr2()
        assert sr2.estimate_drift("nonexistent_session") is None

    def test_estimate_drift_accepts_session_id(self):
        """estimate_drift() must accept a session_id argument (per-session API)."""
        sr2 = _make_sr2()
        # Should not raise even for an unknown session
        result = sr2.estimate_drift("any_session")
        assert result is None


# ---------------------------------------------------------------------------
# 6. Drift warning threshold
# ---------------------------------------------------------------------------


class TestDriftWarningThreshold:
    """Verify that a warning is logged when drift exceeds 15%."""

    def test_warning_logged_when_drift_exceeds_threshold(self, caplog):
        """A drift of >15% must emit a WARNING log."""
        sr2 = _make_sr2()
        # Drift = (1200 - 1000) / 1200 ≈ 16.7% > 15%
        sr2._metrics_manager._last_compiled_tokens = {"warn_session": 1000}

        with caplog.at_level(logging.WARNING, logger="sr2.sr2"):
            sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="warn_session")

        assert any("drift" in record.message.lower() for record in caplog.records), (
            "Expected a WARNING log mentioning drift when drift > 15%"
        )

    def test_no_warning_when_drift_below_threshold(self, caplog):
        """Drift of ≤15% must NOT emit a drift warning."""
        sr2 = _make_sr2()
        # Drift = (1050 - 1000) / 1050 ≈ 4.8% < 15%
        sr2._metrics_manager._last_compiled_tokens = {"quiet_session": 1000}

        with caplog.at_level(logging.WARNING, logger="sr2.sr2"):
            sr2.report_actual_usage(_make_usage(input_tokens=1050), session_id="quiet_session")

        drift_warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "drift" in r.message.lower()
        ]
        assert len(drift_warnings) == 0, "No drift warning expected for small drift"


# ---------------------------------------------------------------------------
# 7. Metrics include actual usage after report_actual_usage()
# ---------------------------------------------------------------------------


class TestMetricsIncludeActualUsage:
    """collect_metrics() must emit actual token metrics when usage has been reported."""

    @pytest.mark.asyncio
    async def test_metrics_include_actual_tokens_after_report(self):
        """After report_actual_usage(), collect_metrics emits ACTUAL_INPUT_TOKENS etc."""
        from sr2.metrics.definitions import MetricNames
        from sr2.pipeline.result import PipelineResult

        sr2 = _make_sr2()
        usage = _make_usage(input_tokens=1500, output_tokens=300, cached_tokens=600)
        sr2.report_actual_usage(usage, session_id="metrics_session")

        pipeline_result = PipelineResult(total_tokens=1200)

        # Collect metrics — must not raise even with minimal args
        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1500,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.4,
            session_id="metrics_session",
        )

        snapshot = sr2._collector.last_snapshot
        assert snapshot is not None, "A snapshot must be stored after collect_metrics()"

        metric_names = {m.name for m in snapshot.metrics}
        assert MetricNames.ACTUAL_INPUT_TOKENS in metric_names, (
            f"ACTUAL_INPUT_TOKENS not in metrics: {metric_names}"
        )
        assert MetricNames.ACTUAL_OUTPUT_TOKENS in metric_names, (
            f"ACTUAL_OUTPUT_TOKENS not in metrics: {metric_names}"
        )
        assert MetricNames.ACTUAL_CACHED_TOKENS in metric_names, (
            f"ACTUAL_CACHED_TOKENS not in metrics: {metric_names}"
        )

    @pytest.mark.asyncio
    async def test_metrics_no_actual_tokens_when_not_reported(self):
        """Without report_actual_usage(), ACTUAL_INPUT_TOKENS must NOT appear in metrics."""
        from sr2.metrics.definitions import MetricNames
        from sr2.pipeline.result import PipelineResult

        sr2 = _make_sr2()
        # Do NOT call report_actual_usage()

        pipeline_result = PipelineResult(total_tokens=1200)
        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1200,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="no_report_session",
        )

        snapshot = sr2._collector.last_snapshot
        metric_names = {m.name for m in snapshot.metrics}
        assert MetricNames.ACTUAL_INPUT_TOKENS not in metric_names, (
            "ACTUAL_INPUT_TOKENS should not appear when report_actual_usage() was never called"
        )

    @pytest.mark.asyncio
    async def test_metrics_actual_token_values_match_reported(self):
        """The actual token values in metrics must match what was passed to report_actual_usage()."""
        from sr2.metrics.definitions import MetricNames
        from sr2.pipeline.result import PipelineResult

        sr2 = _make_sr2()
        usage = _make_usage(input_tokens=2345, output_tokens=456, cached_tokens=789)
        sr2.report_actual_usage(usage, session_id="values_session")

        pipeline_result = PipelineResult(total_tokens=2345)
        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=2,
            loop_total_tokens=2345,
            loop_tool_calls=1,
            loop_cache_hit_rate=0.336,
            session_id="values_session",
        )

        snapshot = sr2._collector.last_snapshot
        by_name = {m.name: m.value for m in snapshot.metrics}

        assert by_name[MetricNames.ACTUAL_INPUT_TOKENS] == pytest.approx(2345.0)
        assert by_name[MetricNames.ACTUAL_OUTPUT_TOKENS] == pytest.approx(456.0)
        assert by_name[MetricNames.ACTUAL_CACHED_TOKENS] == pytest.approx(789.0)

    @pytest.mark.asyncio
    async def test_metrics_include_drift_when_available(self):
        """When drift history exists, TOKEN_ESTIMATE_DRIFT appears in metrics."""
        from sr2.metrics.definitions import MetricNames
        from sr2.pipeline.result import PipelineResult

        sr2 = _make_sr2()
        sr2._metrics_manager._last_compiled_tokens = {"drift_session": 1000}

        # Drift ≈ 16.7% (above threshold)
        sr2.report_actual_usage(_make_usage(input_tokens=1200), session_id="drift_session")

        pipeline_result = PipelineResult(total_tokens=1200)
        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1200,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="drift_session",
        )

        snapshot = sr2._collector.last_snapshot
        metric_names = {m.name for m in snapshot.metrics}
        assert MetricNames.TOKEN_ESTIMATE_DRIFT in metric_names, (
            "TOKEN_ESTIMATE_DRIFT must appear in metrics when drift history exists"
        )

    @pytest.mark.asyncio
    async def test_metrics_session_isolation(self):
        """collect_metrics() for session A must use session A's actual usage, not session B's."""
        from sr2.metrics.definitions import MetricNames
        from sr2.pipeline.result import PipelineResult

        sr2 = _make_sr2()

        sr2.report_actual_usage(_make_usage(input_tokens=1111), session_id="session_a")
        sr2.report_actual_usage(_make_usage(input_tokens=9999), session_id="session_b")

        pipeline_result = PipelineResult(total_tokens=1111)
        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1111,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="session_a",
        )

        snapshot = sr2._collector.last_snapshot
        by_name = {m.name: m.value for m in snapshot.metrics}

        assert by_name.get(MetricNames.ACTUAL_INPUT_TOKENS) == pytest.approx(1111.0), (
            "collect_metrics for session_a should use session_a's reported usage (1111), "
            "not session_b's (9999)"
        )


# ---------------------------------------------------------------------------
# 8. PostLLMProcessor.set_actual_usage()
# ---------------------------------------------------------------------------


class TestPostProcessorSetActualUsage:
    """PostLLMProcessor must accept actual usage via set_actual_usage()."""

    def test_set_actual_usage_method_exists(self):
        """PostLLMProcessor must have a set_actual_usage() method."""
        from sr2.pipeline.post_processor import PostLLMProcessor
        from sr2.pipeline.conversation import ConversationManager
        from sr2.compaction.engine import CompactionEngine
        from sr2.summarization.engine import SummarizationEngine
        from sr2.config.models import CompactionConfig, SummarizationConfig

        conv = ConversationManager(
            compaction_engine=CompactionEngine(CompactionConfig()),
            summarization_engine=SummarizationEngine(
                config=SummarizationConfig(),
                llm_callable=None,
            ),
        )
        proc = PostLLMProcessor(conversation_manager=conv)

        assert hasattr(proc, "set_actual_usage"), "PostLLMProcessor.set_actual_usage() must exist"
        assert callable(proc.set_actual_usage)

    def test_set_actual_usage_stores_value(self):
        """set_actual_usage() stores the usage so _run_compaction can use it."""
        from sr2.pipeline.post_processor import PostLLMProcessor
        from sr2.pipeline.conversation import ConversationManager
        from sr2.compaction.engine import CompactionEngine
        from sr2.summarization.engine import SummarizationEngine
        from sr2.config.models import CompactionConfig, SummarizationConfig

        conv = ConversationManager(
            compaction_engine=CompactionEngine(CompactionConfig()),
            summarization_engine=SummarizationEngine(
                config=SummarizationConfig(),
                llm_callable=None,
            ),
        )
        proc = PostLLMProcessor(conversation_manager=conv)

        usage = _make_usage(input_tokens=3000)
        proc.set_actual_usage(usage)

        assert proc._actual_usage is not None
        assert proc._actual_usage.input_tokens == 3000

    def test_report_actual_usage_calls_post_processor(self):
        """SR2.report_actual_usage() propagates actual usage to the post processor."""
        sr2 = _make_sr2()
        usage = _make_usage(input_tokens=2500)

        sr2.report_actual_usage(usage, session_id="pp_session")

        # The post processor should now have actual usage set
        assert sr2._post_processor._actual_usage is not None
        assert sr2._post_processor._actual_usage.input_tokens == 2500

    def test_run_compaction_uses_actual_tokens_as_current_tokens(self):
        """When _actual_usage is set, _run_compaction passes actual input_tokens as current_tokens.

        This is the critical behavioral test: the whole point of report_actual_usage()
        is that compaction decisions use ground-truth numbers, not SR2's estimates.
        """
        from unittest.mock import patch, MagicMock
        from sr2.pipeline.post_processor import PostLLMProcessor
        from sr2.pipeline.conversation import ConversationManager
        from sr2.compaction.engine import CompactionEngine
        from sr2.summarization.engine import SummarizationEngine
        from sr2.config.models import CompactionConfig, SummarizationConfig

        conv = ConversationManager(
            compaction_engine=CompactionEngine(CompactionConfig()),
            summarization_engine=SummarizationEngine(
                config=SummarizationConfig(),
                llm_callable=None,
            ),
        )
        proc = PostLLMProcessor(conversation_manager=conv)
        proc.set_budget_info(token_budget=8000, current_tokens=500)  # SR2 estimate: 500

        actual_tokens = 3750
        proc.set_actual_usage(_make_usage(input_tokens=actual_tokens))

        captured_calls = []

        original_run_compaction = conv.run_compaction

        def capturing_run_compaction(**kwargs):
            captured_calls.append(kwargs)
            return original_run_compaction(**kwargs)

        with patch.object(conv, "run_compaction", side_effect=capturing_run_compaction):
            import asyncio
            asyncio.run(proc._run_compaction(session_id="test_session"))

        assert len(captured_calls) == 1, "run_compaction should be called once"
        called_current_tokens = captured_calls[0].get("current_tokens")
        assert called_current_tokens == actual_tokens, (
            f"_run_compaction must pass actual input_tokens ({actual_tokens}) "
            f"as current_tokens, not the SR2 estimate (500). Got: {called_current_tokens}"
        )


# ---------------------------------------------------------------------------
# 9. MetricNames constants
# ---------------------------------------------------------------------------


class TestMetricNamesConstants:
    """Verify the new MetricNames constants exist."""

    def test_actual_input_tokens_constant(self):
        """MetricNames.ACTUAL_INPUT_TOKENS must be defined."""
        from sr2.metrics.definitions import MetricNames

        assert hasattr(MetricNames, "ACTUAL_INPUT_TOKENS")
        assert MetricNames.ACTUAL_INPUT_TOKENS == "sr2_actual_input_tokens"

    def test_actual_output_tokens_constant(self):
        """MetricNames.ACTUAL_OUTPUT_TOKENS must be defined."""
        from sr2.metrics.definitions import MetricNames

        assert hasattr(MetricNames, "ACTUAL_OUTPUT_TOKENS")
        assert MetricNames.ACTUAL_OUTPUT_TOKENS == "sr2_actual_output_tokens"

    def test_actual_cached_tokens_constant(self):
        """MetricNames.ACTUAL_CACHED_TOKENS must be defined."""
        from sr2.metrics.definitions import MetricNames

        assert hasattr(MetricNames, "ACTUAL_CACHED_TOKENS")
        assert MetricNames.ACTUAL_CACHED_TOKENS == "sr2_actual_cached_tokens"

    def test_token_estimate_drift_constant(self):
        """MetricNames.TOKEN_ESTIMATE_DRIFT must be defined."""
        from sr2.metrics.definitions import MetricNames

        assert hasattr(MetricNames, "TOKEN_ESTIMATE_DRIFT")
        assert MetricNames.TOKEN_ESTIMATE_DRIFT == "sr2_token_estimate_drift"


# ---------------------------------------------------------------------------
# 10. Backward compatibility: collect_metrics() without prior report
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing callers that never call report_actual_usage() must still work."""

    @pytest.mark.asyncio
    async def test_collect_metrics_works_without_report(self):
        """collect_metrics() must not raise if report_actual_usage() was never called."""
        from sr2.pipeline.result import PipelineResult

        sr2 = _make_sr2()
        pipeline_result = PipelineResult(total_tokens=1000)

        # Must not raise
        await sr2.collect_metrics(
            pipeline_result=pipeline_result,
            interface="test",
            loop_iterations=1,
            loop_total_tokens=1000,
            loop_tool_calls=0,
            loop_cache_hit_rate=0.0,
            session_id="compat_session",
        )

    @pytest.mark.asyncio
    async def test_post_process_works_without_report(self):
        """post_process() must not raise if report_actual_usage() was never called."""
        sr2 = _make_sr2()

        # Must not raise — backward compat is critical
        await sr2.post_process(
            turn_number=1,
            role="assistant",
            content="Hello!",
            session_id="compat_session",
        )
