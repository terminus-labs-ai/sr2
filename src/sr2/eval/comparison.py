"""A/B testing and regression detection for eval harness."""

import logging
import uuid
from datetime import datetime
from typing import Literal

from sr2.config.models import PipelineConfig
from sr2.eval.models import (
    ComparisonResult,
    EvalCase,
    EvalResult,
    RegressionAlert,
)
from sr2.eval.runner import EvalRunner

logger = logging.getLogger(__name__)


class ABTestRunner:
    """Run A/B tests comparing two pipeline configurations."""

    def __init__(self, eval_runner: EvalRunner) -> None:
        self._runner = eval_runner

    async def compare(
        self,
        config_a: PipelineConfig,
        config_b: PipelineConfig,
        cases: list[EvalCase],
        config_a_name: str = "A",
        config_b_name: str = "B",
    ) -> ComparisonResult:
        """Run eval cases against both configs and compare.

        Args:
            config_a: First pipeline configuration
            config_b: Second pipeline configuration
            cases: Eval cases to run
            config_a_name: Display name for config A
            config_b_name: Display name for config B

        Returns:
            ComparisonResult with metrics and winner
        """
        logger.info(f"A/B test: {config_a_name} vs {config_b_name}, {len(cases)} cases")

        # Run both configs
        results_a = await self._runner.run_suite(cases, config_a)
        results_b = await self._runner.run_suite(cases, config_b)

        # Compute averages
        a_coherence = _mean([r.metrics.coherence_score for r in results_a])
        b_coherence = _mean([r.metrics.coherence_score for r in results_b])
        a_efficiency = _mean([r.metrics.token_efficiency for r in results_a])
        b_efficiency = _mean([r.metrics.token_efficiency for r in results_b])

        coherence_improvement = b_coherence - a_coherence
        efficiency_improvement = b_efficiency - a_efficiency

        # Determine winner
        winner = _determine_winner(coherence_improvement, efficiency_improvement)

        # Compute p-value (simple t-test approximation)
        p_value = _approximate_p_value(
            [r.metrics.coherence_score for r in results_a],
            [r.metrics.coherence_score for r in results_b],
        )

        return ComparisonResult(
            test_id=str(uuid.uuid4())[:8],
            config_a=config_a_name,
            config_b=config_b_name,
            timestamp=datetime.now(),
            cases_run=len(cases),
            config_a_results=results_a,
            config_b_results=results_b,
            config_a_avg_coherence=a_coherence,
            config_b_avg_coherence=b_coherence,
            config_a_avg_efficiency=a_efficiency,
            config_b_avg_efficiency=b_efficiency,
            coherence_improvement=coherence_improvement,
            efficiency_improvement=efficiency_improvement,
            p_value=p_value,
            winner=winner,
        )


class RegressionDetector:
    """Detect regressions by comparing eval results against baselines."""

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize regression detector.

        Args:
            thresholds: Minimum acceptable degradation before alerting.
                Keys are metric names, values are max acceptable degradation (0-1).
                Default: 5% degradation triggers warning, 15% triggers critical.
        """
        self._thresholds = thresholds or {
            "coherence_score": 0.05,
            "decision_preservation": 0.05,
            "token_efficiency": 0.10,
            "prefix_hit_rate": 0.10,
        }
        self._baselines: dict[str, dict[str, float]] = {}

    def set_baseline(self, results: list[EvalResult]) -> None:
        """Store current results as baseline for regression detection.

        Args:
            results: List of eval results to use as baseline
        """
        for result in results:
            self._baselines[result.case_id] = {
                "coherence_score": result.metrics.coherence_score,
                "decision_preservation": result.metrics.decision_preservation,
                "token_efficiency": result.metrics.token_efficiency,
                "prefix_hit_rate": result.metrics.prefix_hit_rate,
            }

        logger.info(f"Baseline set with {len(results)} cases")

    def check(self, results: list[EvalResult]) -> list[RegressionAlert]:
        """Check current results against baselines for regressions.

        Args:
            results: Current eval results to check

        Returns:
            List of regression alerts (empty if no regressions)
        """
        if not self._baselines:
            logger.warning("No baseline set, skipping regression check")
            return []

        alerts: list[RegressionAlert] = []

        # Aggregate metrics across cases
        current_metrics: dict[str, list[float]] = {}
        baseline_metrics: dict[str, list[float]] = {}

        for result in results:
            baseline = self._baselines.get(result.case_id)
            if baseline is None:
                continue

            for metric_name in self._thresholds:
                current_val = getattr(result.metrics, metric_name, None)
                baseline_val = baseline.get(metric_name)

                if current_val is not None and baseline_val is not None:
                    if metric_name not in current_metrics:
                        current_metrics[metric_name] = []
                        baseline_metrics[metric_name] = []
                    current_metrics[metric_name].append(current_val)
                    baseline_metrics[metric_name].append(baseline_val)

        # Check each metric for regression
        for metric_name, threshold in self._thresholds.items():
            if metric_name not in current_metrics:
                continue

            current_avg = _mean(current_metrics[metric_name])
            baseline_avg = _mean(baseline_metrics[metric_name])

            if baseline_avg == 0:
                continue

            degradation = (baseline_avg - current_avg) / baseline_avg

            if degradation > threshold:
                severity = _severity_for_degradation(degradation, threshold)
                affected_cases = [r.case_id for r in results if r.case_id in self._baselines]

                alert = RegressionAlert(
                    alert_id=str(uuid.uuid4())[:8],
                    timestamp=datetime.now(),
                    metric_name=metric_name,
                    previous_baseline=baseline_avg,
                    current_value=current_avg,
                    degradation_percent=degradation,
                    severity=severity,
                    affected_cases=affected_cases,
                )
                alerts.append(alert)
                logger.warning(str(alert))

        return alerts


def _mean(values: list[float]) -> float:
    """Compute mean, returning 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _determine_winner(
    coherence_diff: float,
    efficiency_diff: float,
) -> Literal["A", "B", "tie"]:
    """Determine A/B test winner based on improvements.

    B wins if it's better on coherence (primary metric).
    If coherence is a tie (within 2%), use efficiency as tiebreaker.
    """
    if coherence_diff > 0.02:
        return "B"
    elif coherence_diff < -0.02:
        return "A"
    elif efficiency_diff > 0.02:
        return "B"
    elif efficiency_diff < -0.02:
        return "A"
    return "tie"


def _approximate_p_value(group_a: list[float], group_b: list[float]) -> float | None:
    """Approximate p-value using Welch's t-test.

    Returns None if sample size is too small (< 3).
    """
    if len(group_a) < 3 or len(group_b) < 3:
        return None

    n_a = len(group_a)
    n_b = len(group_b)
    mean_a = _mean(group_a)
    mean_b = _mean(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)

    se = (var_a / n_a + var_b / n_b) ** 0.5
    if se == 0:
        return 1.0

    t_stat = abs(mean_a - mean_b) / se

    # Approximate p-value using normal distribution
    # For large samples, t-distribution ≈ normal
    # Using simple approximation: p ≈ 2 * exp(-0.5 * t^2)
    import math

    p_value = 2.0 * math.exp(-0.5 * t_stat * t_stat)
    return min(1.0, p_value)


def _severity_for_degradation(
    degradation: float,
    threshold: float,
) -> Literal["info", "warning", "critical"]:
    """Determine alert severity based on degradation amount."""
    if degradation >= threshold * 3:
        return "critical"
    elif degradation >= threshold * 1.5:
        return "warning"
    return "info"
