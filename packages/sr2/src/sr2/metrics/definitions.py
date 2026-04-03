"""Metric definitions — types, constants, and thresholds."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class MetricValue:
    """A single metric measurement."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class MetricSnapshot:
    """A collection of metrics from a single pipeline invocation."""

    invocation_id: str
    agent_name: str
    interface_type: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: list[MetricValue] = field(default_factory=list)

    def add(self, name: str, value: float, unit: str = "", **labels) -> None:
        """Add a metric to this snapshot."""
        self.metrics.append(
            MetricValue(
                name=name,
                value=value,
                unit=unit,
                labels={
                    "agent": self.agent_name,
                    "interface": self.interface_type,
                    **labels,
                },
                timestamp=self.timestamp,
            )
        )

    def get(self, name: str) -> MetricValue | None:
        """Get first metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None


class MetricNames:
    """Constants for all metric names used in the system."""

    # KV-Cache
    CACHE_HIT_RATE = "sr2_cache_hit_rate"
    CACHE_INVALIDATION_EVENTS = "sr2_cache_invalidation_events"
    COST_SAVINGS_RATIO = "sr2_cost_savings_ratio"
    CONTEXT_PREFIX_STABLE = "sr2_context_prefix_stable"
    CACHE_EFFICIENCY = "sr2_cache_efficiency"

    # Retrieval
    RETRIEVAL_PRECISION = "sr2_retrieval_precision"
    RETRIEVAL_LATENCY_MS = "sr2_retrieval_latency_ms"
    RETRIEVAL_EMPTY_RATE = "sr2_retrieval_empty_rate"

    # Compaction
    COMPACTION_RATIO = "sr2_compaction_ratio"
    COMPACTION_RECOVERY_RATE = "sr2_compaction_recovery_rate"
    COMPACTION_COVERAGE = "sr2_compaction_coverage"

    # Summarization
    SUMMARIZATION_FIDELITY = "sr2_summarization_fidelity"
    SUMMARIZATION_RATIO = "sr2_summarization_ratio"
    SUMMARIZATION_FREQUENCY = "sr2_summarization_frequency"

    # Tool masking
    DENIED_TOOL_ATTEMPTS = "sr2_denied_tool_attempts"
    STATE_TRANSITION_RATE = "sr2_state_transition_rate"

    # Degradation
    FALLBACK_RATE = "sr2_fallback_rate"
    CIRCUIT_BREAKER_ACTIVATIONS = "sr2_circuit_breaker_activations"
    FULL_DEGRADATION_EVENTS = "sr2_full_degradation_events"

    # Agent-level
    TASK_COMPLETION_RATE = "sr2_task_completion_rate"
    TOKEN_EFFICIENCY = "sr2_token_efficiency"
    RESPONSE_QUALITY = "sr2_response_quality"

    # Pipeline
    PIPELINE_TOTAL_TOKENS = "sr2_pipeline_total_tokens"
    PIPELINE_TOTAL_DURATION_MS = "sr2_pipeline_total_duration_ms"
    STAGE_DURATION_MS = "sr2_stage_duration_ms"
    STAGE_TOKENS = "sr2_stage_tokens"

    # Budget & zones
    BUDGET_UTILIZATION = "sr2_budget_utilization"
    ZONE_RAW_TOKENS = "sr2_zone_raw_tokens"
    ZONE_COMPACTED_TOKENS = "sr2_zone_compacted_tokens"
    ZONE_SUMMARIZED_TOKENS = "sr2_zone_summarized_tokens"

    # Conversation lifecycle
    CONVERSATION_TURN_COUNT = "sr2_conversation_turn_count"
    SESSION_DURATION_SECONDS = "sr2_session_duration_seconds"
    SESSION_MESSAGE_COUNT = "sr2_session_message_count"

    # Naive vs managed comparison
    NAIVE_TOKEN_ESTIMATE = "sr2_naive_token_estimate"
    TOKEN_SAVINGS_CUMULATIVE = "sr2_token_savings_cumulative"

    # Context health
    BUDGET_HEADROOM_TOKENS = "sr2_budget_headroom_tokens"
    BUDGET_HEADROOM_RATIO = "sr2_budget_headroom_ratio"
    TRUNCATION_EVENTS = "sr2_truncation_events"

    # Zone dynamics
    ZONE_TRANSITION_EVENTS = "sr2_zone_transition_events"
    RAW_WINDOW_UTILIZATION = "sr2_raw_window_utilization"

    # Memory system
    MEMORIES_EXTRACTED = "sr2_memories_extracted"
    MEMORY_CONFLICTS_DETECTED = "sr2_memory_conflicts_detected"
    MEMORY_STORE_SIZE = "sr2_memory_store_size"


@dataclass
class MetricThreshold:
    """Alert threshold for a metric."""

    metric_name: str
    condition: Literal["<", ">", "<=", ">=", "=="]
    value: float
    severity: Literal["info", "warning", "critical"] = "warning"

    def is_triggered(self, actual: float) -> bool:
        """Check if the threshold is breached."""
        ops = {
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
        }
        return ops[self.condition](actual, self.value)


DEFAULT_THRESHOLDS: list[MetricThreshold] = [
    MetricThreshold(MetricNames.CACHE_HIT_RATE, "<", 0.50, "warning"),
    MetricThreshold(MetricNames.FULL_DEGRADATION_EVENTS, ">", 0, "critical"),
    MetricThreshold(MetricNames.CIRCUIT_BREAKER_ACTIVATIONS, ">", 0, "warning"),
    MetricThreshold(MetricNames.RETRIEVAL_LATENCY_MS, ">", 500, "warning"),
    MetricThreshold(MetricNames.FALLBACK_RATE, ">", 0.10, "warning"),
]
