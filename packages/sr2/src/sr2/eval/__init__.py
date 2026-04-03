"""Eval harness for context engineering quality assessment."""

from sr2.eval.models import (
    EvalCase,
    EvalResult,
    EvalMetrics,
    ComparisonResult,
    RegressionAlert,
)
from sr2.eval.runner import EvalRunner
from sr2.eval.comparison import ABTestRunner, RegressionDetector
from sr2.eval.sample_suites import (
    create_coherence_suite,
    create_compaction_suite,
    create_summarization_suite,
)

__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalMetrics",
    "ComparisonResult",
    "RegressionAlert",
    "EvalRunner",
    "ABTestRunner",
    "RegressionDetector",
    "create_coherence_suite",
    "create_compaction_suite",
    "create_summarization_suite",
]
