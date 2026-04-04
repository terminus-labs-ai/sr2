"""Tests for dimensional memory matching."""

import pytest

from sr2.memory.dimensions import DimensionalMatcher
from sr2.memory.schema import Memory, MemorySearchResult


def _make_result(value: str, score: float, dims: dict[str, str] | None = None) -> MemorySearchResult:
    """Helper to create a MemorySearchResult."""
    mem = Memory(key="test", value=value, dimensions=dims or {})
    return MemorySearchResult(memory=mem, relevance_score=score)


class TestDimensionalMatcher:
    """Tests for DimensionalMatcher."""

    def test_no_current_dimensions(self):
        """No current dimensions → all results returned unchanged."""
        matcher = DimensionalMatcher()
        results = [_make_result("a", 0.8), _make_result("b", 0.6)]
        filtered = matcher.filter(results, {})
        assert len(filtered) == 2

    def test_best_fit_matching_dimension_boosts(self):
        """best_fit: matching dimension boosts score."""
        matcher = DimensionalMatcher(matching_strategy="best_fit")
        matched = _make_result("matched", 0.5, dims={"channel": "slack"})
        unscoped = _make_result("unscoped", 0.5)

        results = matcher.filter([matched, unscoped], {"channel": "slack"})

        # Matched should be boosted: 0.5 * 1.2 = 0.6
        # Unscoped should be penalized: 0.5 * 0.9 = 0.45
        assert results[0].memory.value == "matched"
        assert results[0].relevance_score > results[1].relevance_score

    def test_best_fit_mismatching_dimension_penalizes(self):
        """best_fit: mismatching dimension penalizes score."""
        matcher = DimensionalMatcher(matching_strategy="best_fit")
        mismatched = _make_result("mismatched", 0.8, dims={"channel": "email"})
        unscoped = _make_result("unscoped", 0.5)

        results = matcher.filter([mismatched, unscoped], {"channel": "slack"})

        # Mismatched: 0.8 * 0.3 = 0.24
        # Unscoped: 0.5 * 0.9 = 0.45
        assert results[0].memory.value == "unscoped"
        assert results[0].relevance_score > results[1].relevance_score

    def test_best_fit_unscoped_small_penalty(self):
        """best_fit: unscoped memory gets small penalty but still included."""
        matcher = DimensionalMatcher(matching_strategy="best_fit")
        unscoped = _make_result("unscoped", 0.8)

        results = matcher.filter([unscoped], {"channel": "slack"})

        assert len(results) == 1
        assert results[0].relevance_score == pytest.approx(0.72)  # 0.8 * 0.9

    def test_exact_only_matching(self):
        """exact: only returns memories matching all dimensions."""
        matcher = DimensionalMatcher(matching_strategy="exact")
        matched = _make_result("matched", 0.5, dims={"channel": "slack", "project": "sr2"})
        partial = _make_result("partial", 0.5, dims={"channel": "slack"})
        unscoped = _make_result("unscoped", 0.5)

        results = matcher.filter(
            [matched, partial, unscoped],
            {"channel": "slack", "project": "sr2"},
        )

        assert len(results) == 1
        assert results[0].memory.value == "matched"

    def test_exact_returns_empty(self):
        """exact: returns empty if no exact matches."""
        matcher = DimensionalMatcher(matching_strategy="exact")
        unscoped = _make_result("unscoped", 0.8)

        results = matcher.filter([unscoped], {"channel": "slack"})
        assert len(results) == 0

    def test_fallback_to_generic(self):
        """fallback_to_generic: tries exact, falls back to unscoped."""
        matcher = DimensionalMatcher(matching_strategy="fallback_to_generic")
        unscoped = _make_result("unscoped", 0.8)
        mismatched = _make_result("wrong", 0.9, dims={"channel": "email"})

        results = matcher.filter([unscoped, mismatched], {"channel": "slack"})

        assert len(results) == 1
        assert results[0].memory.value == "unscoped"

    def test_results_resorted_after_matching(self):
        """Results are re-sorted by score after dimensional matching."""
        matcher = DimensionalMatcher(matching_strategy="best_fit")
        low = _make_result("low", 0.3, dims={"channel": "slack"})
        high = _make_result("high", 0.9, dims={"channel": "slack"})

        results = matcher.filter([low, high], {"channel": "slack"})

        assert results[0].memory.value == "high"
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_fit_exact_multiplier_values(self):
        """best_fit applies documented multipliers: 1.2x match, 0.3x mismatch, 0.9x unscoped."""
        matcher = DimensionalMatcher(matching_strategy="best_fit")
        base_score = 0.5

        matched = _make_result("matched", base_score, dims={"channel": "slack"})
        mismatched = _make_result("mismatched", base_score, dims={"channel": "email"})
        unscoped = _make_result("unscoped", base_score)

        matcher.filter([matched, mismatched, unscoped], {"channel": "slack"})

        assert matched.relevance_score == pytest.approx(base_score * 1.2)
        assert mismatched.relevance_score == pytest.approx(base_score * 0.3)
        assert unscoped.relevance_score == pytest.approx(base_score * 0.9)
