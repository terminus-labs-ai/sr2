"""MetricsManager — extracted metrics/observability logic from SR2 facade.

MetricsManager owns all session tracking state and metrics collection methods
that were previously on the SR2 class. This extraction makes metrics concerns
self-contained and testable in isolation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from sr2.metrics.collector import MetricCollector
from sr2.metrics.definitions import MetricNames
from sr2.pipeline.result import ActualTokenUsage, PipelineResult
from sr2.tools.state_machine import ToolStateMachine

logger = logging.getLogger(__name__)


@dataclass
class MetricSources:
    """Read-only references to components that provide metric data.

    These are the components MetricsManager reads from during collect().
    """

    conversation: Any  # ConversationManager
    engine: Any  # PipelineEngine
    post_processor: Any  # PostLLMProcessor
    retriever: Any  # HybridRetriever
    memory_store: Any  # MemoryStore
    trace: Any  # TraceCollector | None


class MetricsManager:
    """Owns all session tracking state and metrics collection logic.

    Extracted from SR2 facade. The SR2 class delegates its metrics methods
    (collect_metrics, report_actual_usage, estimate_drift, export_metrics)
    to an instance of this class.
    """

    def __init__(
        self,
        collector: MetricCollector,
        sources: MetricSources,
        token_budget: int,
        alerts: Any = None,
        push_exporters: list | None = None,
        pull_exporter_name: str | None = None,
    ) -> None:
        self._collector = collector
        self._sources = sources
        self._token_budget = token_budget
        self._alerts = alerts
        self._push_exporters = push_exporters or []
        self._pull_exporter_name = pull_exporter_name

        # Session tracking state (moved from SR2)
        self._session_start_times: dict[str, float] = {}
        self._session_turn_counts: dict[str, int] = {}
        self._token_savings_cumulative: dict[str, float] = {}

        # Actual usage tracking (ground truth from LLM provider), keyed by session_id
        self._last_actual_usage: dict[str, Any] = {}
        self._actual_input_tokens_history: dict[str, list] = {}
        self._estimate_drift_history: dict[str, list] = {}
        self._last_compiled_tokens: dict[str, Any] = {}
        self._max_usage_history: int = 50

    def record_compiled_tokens(self, session_id: str, tokens: int) -> None:
        """Record the compiled token estimate for a session.

        Called by SR2.process() after context compilation so that
        report_actual_usage() can compute drift.
        """
        self._last_compiled_tokens[session_id] = tokens

    def report_actual_usage(
        self,
        usage: ActualTokenUsage,
        session_id: str = "default",
    ) -> None:
        """Report actual token usage from the LLM provider after a loop completes.

        This is the primary feedback mechanism from runtime -> SR2. It:
        1. Records ground-truth token counts for future budget decisions.
        2. Calibrates the estimate drift (SR2 estimate vs actual).
        3. Updates the post-processor's budget info with actual numbers.

        Must be called AFTER process() and BEFORE post_process() so that
        compaction/summarization in post_process use calibrated numbers.
        """
        self._last_actual_usage[session_id] = usage

        # Track actual input tokens for rolling average (per session)
        history = self._actual_input_tokens_history.setdefault(session_id, [])
        history.append(usage.input_tokens)
        if len(history) > self._max_usage_history:
            self._actual_input_tokens_history[session_id] = history[-self._max_usage_history:]

        # Calibrate estimate drift: compare SR2's compiled.tokens estimate
        # against the actual input_tokens from the provider
        compiled_tokens = self._last_compiled_tokens.get(session_id)
        if compiled_tokens is not None and usage.input_tokens > 0:
            drift = (usage.input_tokens - compiled_tokens) / usage.input_tokens
            drift_history = self._estimate_drift_history.setdefault(session_id, [])
            drift_history.append(drift)
            if len(drift_history) > self._max_usage_history:
                self._estimate_drift_history[session_id] = drift_history[-self._max_usage_history:]

            if abs(drift) > 0.15:
                logger.warning(
                    "Token estimate drift %.1f%%: SR2 estimated %d, provider reported %d input tokens",
                    drift * 100,
                    compiled_tokens,
                    usage.input_tokens,
                )

        # Update post-processor budget info with actual numbers so that
        # compaction decisions in post_process() use ground truth
        post_processor = self._sources.post_processor
        if post_processor:
            post_processor.set_budget_info(
                token_budget=self._token_budget,
                current_tokens=usage.input_tokens,
            )
            post_processor.set_actual_usage(usage)

    def estimate_drift(self, session_id: str) -> float | None:
        """Rolling average of (actual - estimated) / actual for the given session.

        Positive means SR2 underestimates. Negative means SR2 overestimates.
        Returns None if no history for this session.
        """
        history = self._estimate_drift_history.get(session_id)
        if not history:
            return None
        return sum(history) / len(history)

    def export_metrics(self) -> str:
        """Export collected metrics in Prometheus text exposition format.

        Uses the pull exporter configured via observability.pull_exporter
        (defaults to 'prometheus'). Requires the exporter plugin to be installed.
        """
        if not hasattr(self, "_exporter"):
            from sr2.metrics.registry import get_pull_exporter

            name = self._pull_exporter_name or "prometheus"
            exporter_cls = get_pull_exporter(name)
            self._exporter = exporter_cls(self._collector)
        return self._exporter.export()

    async def collect(
        self,
        pipeline_result: PipelineResult,
        interface: str,
        loop_iterations: int,
        loop_total_tokens: int,
        loop_tool_calls: int,
        loop_cache_hit_rate: float,
        cache_report: Any = None,
        session_id: str = "default",
        session_messages: list[dict] | None = None,
        session_turn_count: int | None = None,
        session_created_at: float | None = None,
        tool_state_machine: ToolStateMachine | None = None,
    ) -> None:
        """Collect metrics from a loop execution.

        Args:
            session_turn_count: Actual turn count from the session (user messages).
                If provided, used directly instead of an internal counter.
            session_created_at: Session creation time as a Unix timestamp.
                If provided, used for accurate session duration.
        """
        conversation = self._sources.conversation
        engine = self._sources.engine
        post_processor = self._sources.post_processor
        retriever = self._sources.retriever
        memory_store = self._sources.memory_store
        trace = self._sources.trace

        extra: dict[str, float] = {
            "sr2_loop_iterations": loop_iterations,
            "sr2_loop_total_tokens": loop_total_tokens,
            "sr2_loop_tool_calls": loop_tool_calls,
            "sr2_cache_hit_rate": loop_cache_hit_rate,
        }
        if cache_report is not None:
            extra["sr2_context_prefix_stable"] = 1.0 if cache_report.prefix_stable else 0.0
            extra["sr2_cache_efficiency"] = cache_report.cache_efficiency

        # --- Conversation lifecycle ---
        # Turn count: prefer the real session value, fall back to internal counter
        if session_turn_count is not None:
            extra[MetricNames.CONVERSATION_TURN_COUNT] = float(session_turn_count)
        else:
            count = self._session_turn_counts.get(session_id, 0) + 1
            self._session_turn_counts[session_id] = count
            extra[MetricNames.CONVERSATION_TURN_COUNT] = float(count)

        # Session duration: prefer real session creation time
        if session_created_at is not None:
            extra[MetricNames.SESSION_DURATION_SECONDS] = time.time() - session_created_at
        else:
            if session_id not in self._session_start_times:
                self._session_start_times[session_id] = time.time()
            extra[MetricNames.SESSION_DURATION_SECONDS] = (
                time.time() - self._session_start_times[session_id]
            )

        # Session message count
        if session_messages is not None:
            extra[MetricNames.SESSION_MESSAGE_COUNT] = float(len(session_messages))

        # --- Naive vs managed comparison ---
        actual_tokens = pipeline_result.total_tokens
        if session_messages is not None:
            naive_tokens = sum(len(str(m.get("content", ""))) // 4 for m in session_messages)
            extra[MetricNames.NAIVE_TOKEN_ESTIMATE] = float(naive_tokens)

            # Token savings cumulative
            savings_this_turn = max(0.0, naive_tokens - actual_tokens)
            cumulative = self._token_savings_cumulative.get(session_id, 0.0) + savings_this_turn
            self._token_savings_cumulative[session_id] = cumulative
            extra[MetricNames.TOKEN_SAVINGS_CUMULATIVE] = cumulative

            # Cost savings ratio
            if naive_tokens > 0:
                extra[MetricNames.COST_SAVINGS_RATIO] = 1.0 - (actual_tokens / naive_tokens)

        # --- Budget & utilization ---
        headroom = max(0, self._token_budget - actual_tokens)
        extra[MetricNames.BUDGET_HEADROOM_TOKENS] = float(headroom)
        extra[MetricNames.BUDGET_HEADROOM_RATIO] = (
            headroom / self._token_budget if self._token_budget > 0 else 0.0
        )
        extra[MetricNames.BUDGET_UTILIZATION] = (
            actual_tokens / self._token_budget if self._token_budget > 0 else 0.0
        )
        extra[MetricNames.TOKEN_EFFICIENCY] = (
            actual_tokens / self._token_budget if self._token_budget > 0 else 0.0
        )

        # Truncation events
        extra[MetricNames.TRUNCATION_EVENTS] = float(engine.truncation_events)

        # --- Zone dynamics ---
        extra[MetricNames.RAW_WINDOW_UTILIZATION] = conversation.get_raw_window_utilization(
            session_id
        )

        # Zone token counts
        zones = conversation.zones(session_id)
        extra[MetricNames.ZONE_RAW_TOKENS] = float(sum(len(t.content) // 4 for t in zones.raw))
        extra[MetricNames.ZONE_COMPACTED_TOKENS] = float(
            sum(len(t.content) // 4 for t in zones.compacted)
        )
        extra[MetricNames.ZONE_SUMMARIZED_TOKENS] = float(
            sum(len(s) // 4 for s in zones.summarized)
        )

        # Zone transition events
        transitions = conversation.get_zone_transitions(session_id)

        # --- Circuit breaker ---
        cb_status = engine._circuit_breaker.status()
        cb_activations = sum(1 for s in cb_status.values() if s.get("is_open"))
        extra[MetricNames.CIRCUIT_BREAKER_ACTIVATIONS] = float(cb_activations)

        # --- Tool state machine ---
        extra[MetricNames.STATE_TRANSITION_RATE] = float(
            max(0, len(tool_state_machine.state_history) - 1) if tool_state_machine else 0.0
        )
        extra[MetricNames.DENIED_TOOL_ATTEMPTS] = float(
            tool_state_machine.denied_tool_attempts if tool_state_machine else 0.0
        )

        # --- Memory system ---
        extra[MetricNames.MEMORIES_EXTRACTED] = float(post_processor.last_memories_extracted)
        extra[MetricNames.MEMORY_CONFLICTS_DETECTED] = float(
            post_processor.last_conflicts_detected
        )
        try:
            store_size = await memory_store.count()
            extra[MetricNames.MEMORY_STORE_SIZE] = float(store_size)
        except Exception:
            logger.error("Failed to collect memory store size metric", exc_info=True)

        # --- Retrieval metrics ---
        if retriever._total_retrievals > 0:
            extra[MetricNames.RETRIEVAL_LATENCY_MS] = retriever.last_latency_ms
            extra[MetricNames.RETRIEVAL_PRECISION] = retriever.last_avg_precision
            extra[MetricNames.RETRIEVAL_EMPTY_RATE] = retriever.empty_rate

        # --- Compaction metrics (from last post-processing run) ---
        cr = post_processor.last_compaction_result
        if cr is not None and cr.original_tokens > 0:
            extra[MetricNames.COMPACTION_RATIO] = cr.compacted_tokens / cr.original_tokens
            total_turns = cr.turns_compacted + len(cr.turns)
            extra[MetricNames.COMPACTION_COVERAGE] = (
                cr.turns_compacted / total_turns if total_turns > 0 else 0.0
            )

        # --- Summarization metrics (from last post-processing run) ---
        sr = post_processor.last_summarization_result
        if sr is not None and sr.original_tokens > 0:
            extra[MetricNames.SUMMARIZATION_RATIO] = sr.summary_tokens / sr.original_tokens
            # Fidelity: for structured summaries, measure field completeness
            if hasattr(sr.summary, "key_decisions"):
                filled = sum(
                    1
                    for f in [
                        sr.summary.key_decisions,
                        sr.summary.unresolved,
                        sr.summary.facts,
                        sr.summary.user_preferences,
                        sr.summary.errors_encountered,
                    ]
                    if f
                )
                extra[MetricNames.SUMMARIZATION_FIDELITY] = filled / 5.0
            else:
                extra[MetricNames.SUMMARIZATION_FIDELITY] = 1.0 if sr.summary else 0.0

        # --- Actual usage (ground truth from provider), scoped to this session ---
        actual_usage = self._last_actual_usage.get(session_id)
        if actual_usage is not None:
            extra[MetricNames.ACTUAL_INPUT_TOKENS] = float(actual_usage.input_tokens)
            extra[MetricNames.ACTUAL_OUTPUT_TOKENS] = float(actual_usage.output_tokens)
            extra[MetricNames.ACTUAL_CACHED_TOKENS] = float(actual_usage.cached_tokens)

        drift = self.estimate_drift(session_id)
        if drift is not None:
            extra[MetricNames.TOKEN_ESTIMATE_DRIFT] = drift

        snapshot = self._collector.collect(pipeline_result, interface, extra_metrics=extra)

        # Add labeled zone transition events to the snapshot
        for transition_type, count in transitions.items():
            snapshot.add(
                MetricNames.ZONE_TRANSITION_EVENTS,
                float(count),
                "event",
                zone_transition=transition_type,
            )

        # Trace: zones event
        if trace:
            raw_util = conversation.get_raw_window_utilization(session_id)
            trace.emit("zones", {
                "summarized_turns": len(zones.summarized),
                "summarized_tokens": int(extra.get(MetricNames.ZONE_SUMMARIZED_TOKENS, 0)),
                "compacted_turns": len(zones.compacted),
                "compacted_tokens": int(extra.get(MetricNames.ZONE_COMPACTED_TOKENS, 0)),
                "raw_turns": len(zones.raw),
                "raw_tokens": int(extra.get(MetricNames.ZONE_RAW_TOKENS, 0)),
                "raw_window_utilization": raw_util,
                "zone_transitions": transitions,
            }, session_id=session_id)

        # Trace: metrics event
        if trace:
            trace.emit("metrics", {
                "invocation_id": snapshot.invocation_id,
                "budget_utilization": extra.get(MetricNames.BUDGET_UTILIZATION, 0.0),
                "budget_headroom_tokens": int(extra.get(MetricNames.BUDGET_HEADROOM_TOKENS, 0)),
                "cache_hit_rate": loop_cache_hit_rate,
                "loop_iterations": loop_iterations,
                "loop_total_tokens": loop_total_tokens,
                "loop_tool_calls": loop_tool_calls,
                "truncation_events": int(extra.get(MetricNames.TRUNCATION_EVENTS, 0)),
                "circuit_breaker_activations": int(extra.get(
                    MetricNames.CIRCUIT_BREAKER_ACTIVATIONS, 0
                )),
                "memories_extracted": int(extra.get(MetricNames.MEMORIES_EXTRACTED, 0)),
                "compaction_ratio": extra.get(MetricNames.COMPACTION_RATIO, 0.0)
                if MetricNames.COMPACTION_RATIO in extra else None,
                "summarization_ratio": extra.get(MetricNames.SUMMARIZATION_RATIO, 0.0)
                if MetricNames.SUMMARIZATION_RATIO in extra else None,
            }, session_id=session_id)

        # Fire alert checks (via plugin)
        if self._alerts is not None:
            await self._alerts.evaluate(snapshot)
