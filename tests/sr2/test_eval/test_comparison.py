"""Tests for A/B testing and regression detection."""

import pytest
from datetime import datetime

from sr2.eval.comparison import (
    _mean,
    _determine_winner,
    _approximate_p_value,
    _severity_for_degradation,
    RegressionDetector,
)
from sr2.eval.models import EvalCase, EvalMetrics, EvalResult


def test_mean() -> None:
    """Test mean calculation."""
    assert _mean([1.0, 2.0, 3.0]) == 2.0
    assert _mean([0.5, 0.5]) == 0.5
    assert _mean([0.0]) == 0.0
    assert _mean([]) == 0.0


def test_determine_winner() -> None:
    """Test A/B test winner determination."""
    # B wins with significant coherence improvement
    assert _determine_winner(0.1, 0.0) == "B"

    # A wins with significant coherence improvement
    assert _determine_winner(-0.1, 0.0) == "A"

    # Coherence tie, B wins on efficiency
    assert _determine_winner(0.01, 0.05) == "B"

    # Coherence tie, A wins on efficiency
    assert _determine_winner(0.01, -0.05) == "A"

    # Both metrics tie
    assert _determine_winner(0.01, 0.01) == "tie"


def test_approximate_p_value() -> None:
    """Test p-value approximation."""
    # Identical groups should have p-value near 1.0
    group = [0.5, 0.5, 0.5]
    p = _approximate_p_value(group, group)
    assert p is not None
    assert p > 0.9

    # Very different groups should have low p-value
    # Use groups with enough variance to produce meaningful difference
    group_a = [0.1, 0.15, 0.12, 0.14, 0.13]
    group_b = [0.8, 0.85, 0.82, 0.84, 0.83]
    p = _approximate_p_value(group_a, group_b)
    assert p is not None
    assert p < 0.05  # Relaxed threshold for approximation

    # Small sample size returns None
    p = _approximate_p_value([0.5], [0.5])
    assert p is None


def test_severity_for_degradation() -> None:
    """Test severity classification for degradation."""
    threshold = 0.05

    # Below threshold: info
    assert _severity_for_degradation(0.02, threshold) == "info"

    # At threshold: info
    assert _severity_for_degradation(0.05, threshold) == "info"

    # 2x threshold: warning (>= 1.5x)
    assert _severity_for_degradation(0.10, threshold) == "warning"

    # 4x threshold: critical (>= 3x)
    assert _severity_for_degradation(0.20, threshold) == "critical"


def test_regression_detector_baseline() -> None:
    """Test setting baseline for regression detection."""
    detector = RegressionDetector()

    metrics = EvalMetrics(
        coherence_score=0.85,
        decision_preservation=0.80,
        token_efficiency=0.90,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.75,
        layer_cache_hit_rate=0.70,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    result = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_001",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=metrics,
        config_used="default",
        version="1.0.0",
    )

    detector.set_baseline([result])
    assert "test_001" in detector._baselines
    assert detector._baselines["test_001"]["coherence_score"] == 0.85


def test_regression_detector_no_regression() -> None:
    """Test regression detector with no actual regression."""
    detector = RegressionDetector()

    baseline_metrics = EvalMetrics(
        coherence_score=0.85,
        decision_preservation=0.80,
        token_efficiency=0.90,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.75,
        layer_cache_hit_rate=0.70,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    baseline_result = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_001",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=baseline_metrics,
        config_used="default",
        version="1.0.0",
    )

    detector.set_baseline([baseline_result])

    # Current result is similar (small improvement)
    current_metrics = EvalMetrics(
        coherence_score=0.86,
        decision_preservation=0.81,
        token_efficiency=0.91,
        compilation_time_ms=99.0,
        total_time_ms=149.0,
        prefix_hit_rate=0.76,
        layer_cache_hit_rate=0.71,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    current_result = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_002",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=current_metrics,
        config_used="default",
        version="1.0.0",
    )

    alerts = detector.check([current_result])
    assert len(alerts) == 0


def test_regression_detector_detects_regression() -> None:
    """Test regression detector detecting actual regression."""
    detector = RegressionDetector(
        thresholds={
            "coherence_score": 0.05,
            "decision_preservation": 0.05,
            "token_efficiency": 0.10,
            "prefix_hit_rate": 0.10,
        }
    )

    baseline_metrics = EvalMetrics(
        coherence_score=0.85,
        decision_preservation=0.80,
        token_efficiency=0.90,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.75,
        layer_cache_hit_rate=0.70,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    baseline_result = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_001",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=baseline_metrics,
        config_used="default",
        version="1.0.0",
    )

    detector.set_baseline([baseline_result])

    # Current result has significant degradation
    current_metrics = EvalMetrics(
        coherence_score=0.70,  # 17% degradation
        decision_preservation=0.80,
        token_efficiency=0.90,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.75,
        layer_cache_hit_rate=0.70,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    current_result = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_002",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=current_metrics,
        config_used="default",
        version="1.0.0",
    )

    alerts = detector.check([current_result])
    assert len(alerts) > 0
    assert alerts[0].metric_name == "coherence_score"
    assert alerts[0].severity in ["warning", "critical"]
