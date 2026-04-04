"""CI-friendly tests for the token_savings benchmark.

Runs the benchmark programmatically with a small number of turns to validate:
- Managed tokens < naive tokens (SR2 saves tokens)
- KV-cache prefix hit rate > 0 (prefix stability works)
- No budget violations for managed track
- JSON output is valid and contains expected fields

No API keys required — token_savings is pure computation.
"""

import asyncio
import json
import sys
import os

import pytest

# Add benchmarks/ and its _shared to sys.path so we can import directly
_benchmarks_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "benchmarks"
)
_benchmarks_dir = os.path.abspath(_benchmarks_dir)
if _benchmarks_dir not in sys.path:
    sys.path.insert(0, _benchmarks_dir)

import _shared  # noqa: F401 — sets up sr2 on sys.path

from token_savings import BenchmarkResult, run_benchmark


@pytest.fixture(scope="module")
def benchmark_result() -> BenchmarkResult:
    """Run the benchmark once with a small turn count and share across tests."""
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        run_benchmark(num_turns=10, raw_window=3, token_budget=32000)
    )


class TestTokenSavingsBenchmark:
    """Validates core invariants of the token savings benchmark."""

    def test_managed_tokens_less_than_naive(self, benchmark_result: BenchmarkResult):
        """SR2 pipeline should use fewer total tokens than naive concatenation."""
        assert benchmark_result.managed_total_tokens < benchmark_result.naive_total_tokens, (
            f"Managed ({benchmark_result.managed_total_tokens}) should be less than "
            f"naive ({benchmark_result.naive_total_tokens})"
        )

    def test_savings_percent_positive(self, benchmark_result: BenchmarkResult):
        assert benchmark_result.savings_percent > 0

    def test_cache_hit_rate_positive(self, benchmark_result: BenchmarkResult):
        """KV-cache prefix should be stable for at least some turns."""
        assert benchmark_result.cache_hit_rate > 0, (
            "Expected at least some prefix cache hits"
        )

    def test_no_managed_budget_violations(self, benchmark_result: BenchmarkResult):
        """Managed pipeline should never exceed the token budget."""
        assert benchmark_result.budget_violations_managed == 0

    def test_correct_turn_count(self, benchmark_result: BenchmarkResult):
        assert benchmark_result.turns == 10

    def test_scenario_name(self, benchmark_result: BenchmarkResult):
        assert benchmark_result.scenario == "token_savings"

    def test_per_turn_data_present(self, benchmark_result: BenchmarkResult):
        """Per-turn breakdown should have an entry for each turn."""
        assert len(benchmark_result.per_turn) == benchmark_result.turns

    def test_per_turn_fields(self, benchmark_result: BenchmarkResult):
        """Each per-turn entry should have the expected keys."""
        for entry in benchmark_result.per_turn:
            assert "turn" in entry
            assert "naive_tokens" in entry
            assert "managed_tokens" in entry

    def test_json_output_valid(self, benchmark_result: BenchmarkResult):
        """to_dict() produces valid JSON with all expected top-level fields."""
        data = benchmark_result.to_dict()
        # Roundtrip through JSON to prove it's serializable
        raw = json.dumps(data)
        parsed = json.loads(raw)

        expected_fields = {
            "scenario",
            "turns",
            "naive_total_tokens",
            "managed_total_tokens",
            "savings_percent",
            "cache_hit_rate",
            "budget_violations_naive",
            "budget_violations_managed",
            "per_turn",
        }
        assert expected_fields.issubset(parsed.keys()), (
            f"Missing fields: {expected_fields - parsed.keys()}"
        )

    def test_json_excludes_internal_fields(self, benchmark_result: BenchmarkResult):
        """Internal _turn_metrics should not leak into JSON output."""
        data = benchmark_result.to_dict()
        assert "_turn_metrics" not in data
