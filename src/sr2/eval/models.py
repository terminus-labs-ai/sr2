"""Data models for eval harness."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class EvalCase:
    """A single evaluation case with expected outputs."""

    id: str
    name: str
    description: str
    # Input
    system_prompt: str
    conversation_turns: list[tuple[str, str]]  # List of (user, assistant) pairs
    # Expected outputs
    expected_key_facts: list[str]  # Facts that should be remembered
    expected_decisions: list[str]  # Decisions that should be preserved
    expected_tokens: int  # Expected token count range (±10%)
    # Config
    config_overrides: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class EvalMetrics:
    """Quantitative metrics from a single eval case."""

    # Context quality metrics
    coherence_score: float  # 0-1: Does agent remember key facts?
    decision_preservation: float  # 0-1: Are decisions kept?
    token_efficiency: float  # 0-1: Actual tokens vs expected
    # Timing
    compilation_time_ms: float
    total_time_ms: float
    # Cache efficiency
    prefix_hit_rate: float  # 0-1: KV-cache prefix reuse
    layer_cache_hit_rate: float  # 0-1: Layer-level cache hits
    # Degradation
    circuit_breaker_activations: int
    layers_skipped: int


@dataclass
class EvalResult:
    """Result from running a single eval case."""

    case_id: str
    case_name: str
    run_id: str
    timestamp: datetime
    # Outputs
    compiled_context: str
    final_response: str | None
    metrics: EvalMetrics
    # Metadata
    config_used: str
    version: str
    error: str | None = None

    def passed(self, thresholds: dict[str, float] | None = None) -> bool:
        """Check if eval passed quality thresholds.

        Default thresholds:
        - coherence_score >= 0.8
        - decision_preservation >= 0.75
        - token_efficiency >= 0.7
        - prefix_hit_rate >= 0.6
        """
        if thresholds is None:
            thresholds = {
                "coherence_score": 0.8,
                "decision_preservation": 0.75,
                "token_efficiency": 0.7,
                "prefix_hit_rate": 0.6,
            }

        return all(
            getattr(self.metrics, key, 0) >= threshold for key, threshold in thresholds.items()
        )


@dataclass
class ComparisonResult:
    """Result from A/B testing two configurations."""

    test_id: str
    config_a: str
    config_b: str
    timestamp: datetime
    # Results per config
    cases_run: int
    config_a_results: list[EvalResult]
    config_b_results: list[EvalResult]
    # Aggregated metrics
    config_a_avg_coherence: float
    config_b_avg_coherence: float
    config_a_avg_efficiency: float
    config_b_avg_efficiency: float
    # Statistical significance
    coherence_improvement: float  # Can be negative
    efficiency_improvement: float
    p_value: float | None = None  # Statistical significance
    winner: Literal["A", "B", "tie"] = "tie"

    def summary(self) -> str:
        """Human-readable summary of A/B test results."""
        lines = [
            f"A/B Test: {self.config_a} vs {self.config_b}",
            f"Cases run: {self.cases_run}",
            "",
            "Coherence (context quality):",
            f"  Config A: {self.config_a_avg_coherence:.2%}",
            f"  Config B: {self.config_b_avg_coherence:.2%}",
            f"  Improvement: {self.coherence_improvement:+.2%}",
            "",
            "Token Efficiency:",
            f"  Config A: {self.config_a_avg_efficiency:.2%}",
            f"  Config B: {self.config_b_avg_efficiency:.2%}",
            f"  Improvement: {self.efficiency_improvement:+.2%}",
            "",
            f"Winner: {self.winner}",
        ]
        if self.p_value is not None:
            lines.append(f"Statistical significance (p-value): {self.p_value:.4f}")

        return "\n".join(lines)


@dataclass
class RegressionAlert:
    """Alert for detected performance regression."""

    alert_id: str
    timestamp: datetime
    metric_name: str
    previous_baseline: float
    current_value: float
    degradation_percent: float
    severity: Literal["info", "warning", "critical"]
    affected_cases: list[str]

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.metric_name} degraded by "
            f"{self.degradation_percent:.1%} (was {self.previous_baseline:.2f}, "
            f"now {self.current_value:.2f})"
        )
