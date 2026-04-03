"""Tests for eval models."""

import pytest
from datetime import datetime

from sr2.eval.models import (
    EvalCase,
    EvalMetrics,
    EvalResult,
    ComparisonResult,
    RegressionAlert,
)


def test_eval_case_creation() -> None:
    """Test creating an eval case."""
    case = EvalCase(
        id="test_001",
        name="Test Case",
        description="A test case",
        system_prompt="You are helpful",
        conversation_turns=[("Hello", "Hi there")],
        expected_key_facts=["hello"],
        expected_decisions=[],
        expected_tokens=100,
    )

    assert case.id == "test_001"
    assert case.name == "Test Case"
    assert len(case.conversation_turns) == 1


def test_eval_metrics_creation() -> None:
    """Test creating eval metrics."""
    metrics = EvalMetrics(
        coherence_score=0.85,
        decision_preservation=0.90,
        token_efficiency=0.95,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.80,
        layer_cache_hit_rate=0.75,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    assert metrics.coherence_score == 0.85
    assert metrics.token_efficiency == 0.95


def test_eval_result_passed_with_defaults() -> None:
    """Test eval result pass/fail with default thresholds."""
    metrics_pass = EvalMetrics(
        coherence_score=0.9,
        decision_preservation=0.8,
        token_efficiency=0.8,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.7,
        layer_cache_hit_rate=0.6,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    result_pass = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_001",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=metrics_pass,
        config_used="default",
        version="1.0.0",
    )

    assert result_pass.passed() is True

    # Test with failing metrics
    metrics_fail = EvalMetrics(
        coherence_score=0.5,
        decision_preservation=0.3,
        token_efficiency=0.4,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.2,
        layer_cache_hit_rate=0.1,
        circuit_breaker_activations=2,
        layers_skipped=1,
    )

    result_fail = EvalResult(
        case_id="test_002",
        case_name="Test",
        run_id="run_002",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=metrics_fail,
        config_used="default",
        version="1.0.0",
    )

    assert result_fail.passed() is False


def test_eval_result_passed_with_custom_thresholds() -> None:
    """Test eval result with custom thresholds."""
    metrics = EvalMetrics(
        coherence_score=0.6,
        decision_preservation=0.6,
        token_efficiency=0.6,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.6,
        layer_cache_hit_rate=0.5,
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

    # With strict thresholds, should fail
    assert result.passed({"coherence_score": 0.8}) is False

    # With loose thresholds, should pass
    assert result.passed({"coherence_score": 0.5}) is True


def test_comparison_result_summary() -> None:
    """Test comparison result summary formatting."""
    metrics_a = EvalMetrics(
        coherence_score=0.8,
        decision_preservation=0.75,
        token_efficiency=0.7,
        compilation_time_ms=100.0,
        total_time_ms=150.0,
        prefix_hit_rate=0.6,
        layer_cache_hit_rate=0.5,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    result_a = EvalResult(
        case_id="test_001",
        case_name="Test",
        run_id="run_001",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=metrics_a,
        config_used="config_a",
        version="1.0.0",
    )

    metrics_b = EvalMetrics(
        coherence_score=0.85,
        decision_preservation=0.80,
        token_efficiency=0.75,
        compilation_time_ms=95.0,
        total_time_ms=140.0,
        prefix_hit_rate=0.65,
        layer_cache_hit_rate=0.55,
        circuit_breaker_activations=0,
        layers_skipped=0,
    )

    result_b = EvalResult(
        case_id="test_002",
        case_name="Test",
        run_id="run_002",
        timestamp=datetime.now(),
        compiled_context="test",
        final_response=None,
        metrics=metrics_b,
        config_used="config_b",
        version="1.0.0",
    )

    comparison = ComparisonResult(
        test_id="test_001",
        config_a="Config A",
        config_b="Config B",
        timestamp=datetime.now(),
        cases_run=2,
        config_a_results=[result_a],
        config_b_results=[result_b],
        config_a_avg_coherence=0.8,
        config_b_avg_coherence=0.85,
        config_a_avg_efficiency=0.7,
        config_b_avg_efficiency=0.75,
        coherence_improvement=0.05,
        efficiency_improvement=0.05,
        p_value=0.15,
        winner="B",
    )

    summary = comparison.summary()
    assert "Config A vs Config B" in summary
    assert "Winner: B" in summary
    assert "80.00%" in summary  # Config A coherence (formatted as percentage)
    assert "85.00%" in summary  # Config B coherence (formatted as percentage)


def test_regression_alert_string() -> None:
    """Test regression alert string representation."""
    alert = RegressionAlert(
        alert_id="alert_001",
        timestamp=datetime.now(),
        metric_name="coherence_score",
        previous_baseline=0.85,
        current_value=0.72,
        degradation_percent=0.15,
        severity="warning",
        affected_cases=["test_001", "test_002"],
    )

    alert_str = str(alert)
    assert "WARNING" in alert_str
    assert "coherence_score" in alert_str
    assert "15.0%" in alert_str
    assert "0.85" in alert_str
    assert "0.72" in alert_str
