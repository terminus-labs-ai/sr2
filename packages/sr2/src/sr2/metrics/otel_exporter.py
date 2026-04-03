"""OpenTelemetry exporter — maps MetricSnapshot data to OTel instruments."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2.metrics.collector import MetricCollector

from sr2.metrics.definitions import MetricNames, MetricSnapshot

logger = logging.getLogger(__name__)

# OTel instrument type per metric name
_HISTOGRAMS = frozenset(
    {
        MetricNames.PIPELINE_TOTAL_TOKENS,
        MetricNames.PIPELINE_TOTAL_DURATION_MS,
        MetricNames.STAGE_DURATION_MS,
        MetricNames.STAGE_TOKENS,
        MetricNames.RETRIEVAL_LATENCY_MS,
    }
)

_COUNTERS = frozenset(
    {
        MetricNames.CACHE_INVALIDATION_EVENTS,
        MetricNames.FALLBACK_RATE,
        MetricNames.CIRCUIT_BREAKER_ACTIVATIONS,
        MetricNames.FULL_DEGRADATION_EVENTS,
        MetricNames.DENIED_TOOL_ATTEMPTS,
        MetricNames.TOKEN_SAVINGS_CUMULATIVE,
        MetricNames.TRUNCATION_EVENTS,
        MetricNames.ZONE_TRANSITION_EVENTS,
        MetricNames.MEMORIES_EXTRACTED,
        MetricNames.MEMORY_CONFLICTS_DETECTED,
    }
)

_GAUGES = frozenset(
    {
        MetricNames.CACHE_HIT_RATE,
        MetricNames.CONTEXT_PREFIX_STABLE,
        MetricNames.CACHE_EFFICIENCY,
        MetricNames.COST_SAVINGS_RATIO,
        MetricNames.COMPACTION_RATIO,
        MetricNames.COMPACTION_RECOVERY_RATE,
        MetricNames.COMPACTION_COVERAGE,
        MetricNames.SUMMARIZATION_FIDELITY,
        MetricNames.SUMMARIZATION_RATIO,
        MetricNames.SUMMARIZATION_FREQUENCY,
        MetricNames.RETRIEVAL_PRECISION,
        MetricNames.RETRIEVAL_EMPTY_RATE,
        MetricNames.STATE_TRANSITION_RATE,
        MetricNames.TASK_COMPLETION_RATE,
        MetricNames.TOKEN_EFFICIENCY,
        MetricNames.RESPONSE_QUALITY,
        MetricNames.BUDGET_UTILIZATION,
        MetricNames.ZONE_RAW_TOKENS,
        MetricNames.ZONE_COMPACTED_TOKENS,
        MetricNames.ZONE_SUMMARIZED_TOKENS,
        MetricNames.CONVERSATION_TURN_COUNT,
        MetricNames.SESSION_DURATION_SECONDS,
        MetricNames.SESSION_MESSAGE_COUNT,
        MetricNames.NAIVE_TOKEN_ESTIMATE,
        MetricNames.BUDGET_HEADROOM_TOKENS,
        MetricNames.BUDGET_HEADROOM_RATIO,
        MetricNames.RAW_WINDOW_UTILIZATION,
        MetricNames.MEMORY_STORE_SIZE,
    }
)

# Units for OTel instrument descriptions
_UNITS: dict[str, str] = {
    MetricNames.PIPELINE_TOTAL_TOKENS: "tokens",
    MetricNames.PIPELINE_TOTAL_DURATION_MS: "ms",
    MetricNames.STAGE_DURATION_MS: "ms",
    MetricNames.STAGE_TOKENS: "tokens",
    MetricNames.RETRIEVAL_LATENCY_MS: "ms",
    MetricNames.ZONE_RAW_TOKENS: "tokens",
    MetricNames.ZONE_COMPACTED_TOKENS: "tokens",
    MetricNames.ZONE_SUMMARIZED_TOKENS: "tokens",
    MetricNames.CONVERSATION_TURN_COUNT: "turns",
    MetricNames.SESSION_DURATION_SECONDS: "seconds",
    MetricNames.SESSION_MESSAGE_COUNT: "messages",
    MetricNames.NAIVE_TOKEN_ESTIMATE: "tokens",
    MetricNames.TOKEN_SAVINGS_CUMULATIVE: "tokens",
    MetricNames.BUDGET_HEADROOM_TOKENS: "tokens",
    MetricNames.MEMORIES_EXTRACTED: "memories",
    MetricNames.MEMORY_CONFLICTS_DETECTED: "conflicts",
    MetricNames.MEMORY_STORE_SIZE: "memories",
}


class OTelExporter:
    """Pushes MetricSnapshot data to OpenTelemetry instruments in real-time.

    Registers a callback on MetricCollector that fires on every collect() call.
    Requires opentelemetry-api and opentelemetry-sdk to be installed.
    """

    def __init__(self, collector: MetricCollector, meter_name: str = "sr2") -> None:
        from opentelemetry import metrics as otel_metrics

        self._meter = otel_metrics.get_meter(meter_name)
        self._instruments: dict[str, object] = {}

        # Create instruments
        for name in _HISTOGRAMS:
            self._instruments[name] = self._meter.create_histogram(
                name=name,
                unit=_UNITS.get(name, ""),
                description=name.replace("sr2_", "").replace("_", " "),
            )
        for name in _COUNTERS:
            self._instruments[name] = self._meter.create_counter(
                name=name,
                description=name.replace("sr2_", "").replace("_", " "),
            )
        for name in _GAUGES:
            self._instruments[name] = self._meter.create_gauge(
                name=name,
                unit=_UNITS.get(name, ""),
                description=name.replace("sr2_", "").replace("_", " "),
            )

        # Register callback
        collector.on_collect(self._on_snapshot)
        logger.info("OTelExporter registered on MetricCollector (meter=%s)", meter_name)

    def _on_snapshot(self, snapshot: MetricSnapshot) -> None:
        """Push a snapshot's metrics to OTel instruments."""
        for mv in snapshot.metrics:
            instrument = self._instruments.get(mv.name)
            if instrument is None:
                continue

            attrs = dict(mv.labels) if mv.labels else {}

            if mv.name in _HISTOGRAMS:
                instrument.record(mv.value, attributes=attrs)
            elif mv.name in _COUNTERS:
                instrument.add(mv.value, attributes=attrs)
            elif mv.name in _GAUGES:
                instrument.set(mv.value, attributes=attrs)
