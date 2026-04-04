#!/usr/bin/env python3
"""Benchmark 1: Token Savings

Measures cumulative token savings of SR2's managed pipeline vs naive
concatenation. Uses real library components — zero LLM dependencies.

Usage:
    python benchmarks/token_savings.py [--turns N] [--raw-window N] [--budget N] [--json]
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field

# Ensure benchmarks/ is on sys.path so _shared can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _shared  # noqa: F401  — sets up sys.path for sr2

from _shared.conversation_data import (
    RETRIEVED_MEMORIES,
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    generate_conversation,
)
from _shared.pipeline_factory import create_benchmark_pipeline
from _shared.reporting import (
    TurnMetrics,
    print_budget_compliance,
    print_compaction_events,
    print_compile_overhead,
    print_growth_chart,
    print_header,
    print_headline_metrics,
    print_layer_distribution,
    print_section,
    print_three_zone_breakdown,
    print_token_table,
)

from sr2.resolvers.registry import ResolverContext


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class BenchmarkResult:
    """Machine-readable results from a token savings benchmark run."""

    scenario: str
    turns: int
    naive_total_tokens: int
    managed_total_tokens: int
    savings_percent: float
    cache_hit_rate: float
    budget_violations_naive: int
    budget_violations_managed: int
    per_turn: list[dict[str, int]] = field(default_factory=list)
    # Detailed per-turn metrics for the full report (not serialized to JSON)
    _turn_metrics: list[TurnMetrics] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (excludes internal _turn_metrics)."""
        d = asdict(self)
        d.pop("_turn_metrics", None)
        return d


async def run_benchmark(
    num_turns: int = 30,
    raw_window: int = 5,
    token_budget: int = 32000,
) -> BenchmarkResult:
    """Run the token savings benchmark and return structured results.

    Returns a BenchmarkResult with all metrics. Call print_report()
    for the human-readable output.
    """
    conversation = generate_conversation(num_turns)
    pipeline = create_benchmark_pipeline(
        raw_window=raw_window,
        token_budget=token_budget,
    )

    turns: list[TurnMetrics] = []
    prev_prefix = ""
    prefix_hits = 0
    naive_history: list[str] = []

    for turn in conversation:
        m = TurnMetrics(turn_number=turn.turn_number, role=turn.role)

        # ── Naive track: concatenate everything ──
        naive_history.append(f"{turn.role}: {turn.content}")
        naive_ctx = SYSTEM_PROMPT + "\n\n" + TOOL_DEFINITIONS + "\n\n" + "\n".join(naive_history)
        m.naive_tokens = _estimate_tokens(naive_ctx)

        # ── Managed track: real SR2 pipeline ──
        t0 = time.perf_counter_ns()

        pipeline.conversation_manager.add_turn(turn)
        cr = pipeline.conversation_manager.run_compaction()

        if cr is not None:
            m.compaction_turns_compacted = cr.turns_compacted
            m.compaction_original_tokens = cr.original_tokens
            m.compaction_compacted_tokens = cr.compacted_tokens

        # Build session history from zones
        zones = pipeline.conversation_manager.zones()
        session_parts: list[dict[str, str]] = []
        for s in zones.summarized:
            session_parts.append({"role": "system", "content": f"[Summary] {s}"})
        for t_turn in zones.compacted:
            session_parts.append({"role": t_turn.role, "content": t_turn.content})
        for t_turn in zones.raw:
            session_parts.append({"role": t_turn.role, "content": t_turn.content})

        ctx = ResolverContext(
            agent_config={
                "system_prompt": SYSTEM_PROMPT,
                "tool_definitions": TOOL_DEFINITIONS,
                "retrieved_memories": RETRIEVED_MEMORIES,
                "session_history": session_parts,
            },
            trigger_input=turn.content,
            interface_type="user_message",
        )

        compiled = await pipeline.engine.compile(pipeline.config, ctx)

        m.compile_us = (time.perf_counter_ns() - t0) / 1000
        m.managed_tokens = compiled.tokens

        # Layer-level tokens
        for layer_name, contents in compiled.layers.items():
            m.layer_tokens[layer_name] = sum(c.tokens for c in contents)

        # Zone breakdown
        m.zone_raw = sum(_estimate_tokens(t_turn.content) for t_turn in zones.raw)
        m.zone_compacted = sum(_estimate_tokens(t_turn.content) for t_turn in zones.compacted)
        m.zone_summarized = sum(_estimate_tokens(s) for s in zones.summarized)

        # Prefix stability
        prefix_content = ""
        for layer_name in ["core", "memory"]:
            if layer_name in compiled.layers:
                prefix_content += "".join(c.content for c in compiled.layers[layer_name])
        prefix = hashlib.sha256(prefix_content.encode()).hexdigest()[:16]
        m.prefix_hash = prefix
        if prev_prefix and prefix == prev_prefix:
            m.prefix_stable = True
            prefix_hits += 1
        prev_prefix = prefix

        turns.append(m)

    # ── Aggregates ──
    total_naive = sum(t.naive_tokens for t in turns)
    total_managed = sum(t.managed_tokens for t in turns)
    prefix_stability = prefix_hits / (len(turns) - 1) if len(turns) > 1 else 0
    pct = ((total_naive - total_managed) / total_naive * 100) if total_naive else 0

    return BenchmarkResult(
        scenario="token_savings",
        turns=len(turns),
        naive_total_tokens=total_naive,
        managed_total_tokens=total_managed,
        savings_percent=round(pct, 2),
        cache_hit_rate=round(prefix_stability, 4),
        budget_violations_naive=sum(1 for t in turns if t.naive_tokens > token_budget),
        budget_violations_managed=sum(1 for t in turns if t.managed_tokens > token_budget),
        per_turn=[
            {
                "turn": t.turn_number,
                "naive_tokens": t.naive_tokens,
                "managed_tokens": t.managed_tokens,
            }
            for t in turns
        ],
        _turn_metrics=turns,
    )


def print_report(
    result: BenchmarkResult,
    turns: list[TurnMetrics] | None = None,
    raw_window: int = 5,
    token_budget: int = 32000,
) -> None:
    """Print the human-readable benchmark report.

    If turns (detailed per-turn metrics) are not provided, only prints
    the summary section.
    """
    print_header("SR2 — TOKEN SAVINGS BENCHMARK")
    print(f"  Simulated turns: {result.turns}")
    print(f"  Raw window: {raw_window}  |  Token budget: {token_budget:,}")
    print()

    savings = result.naive_total_tokens - result.managed_total_tokens
    print_section("CUMULATIVE TOKEN USAGE (sum of all turns' context windows)")
    print(f"  Naive (concatenate all history):  {result.naive_total_tokens:>12,} tokens")
    print(f"  Managed (SR2 pipeline):      {result.managed_total_tokens:>12,} tokens")
    print(
        f"  Tokens saved:                     {savings:>12,} tokens  ({result.savings_percent:.1f}%)"
    )
    print()

    if turns:
        print_budget_compliance(turns, token_budget)

        print_section("KV-CACHE PREFIX STABILITY")
        print(f"  Core + memory prefix unchanged:  {result.cache_hit_rate * 100:.0f}% of turns")
        print("  -> Provider can reuse KV cache for the stable prefix portion")
        print()

        print_compile_overhead(turns)
        print_token_table(turns)
        print_compaction_events(turns)

        last = turns[-1] if turns else None
        if last:
            print_layer_distribution(last)
            print_three_zone_breakdown(last)

        print_growth_chart(turns)
        print_headline_metrics(
            turns,
            result.naive_total_tokens,
            result.managed_total_tokens,
            result.cache_hit_rate,
            token_budget,
        )


async def run_and_print(
    num_turns: int = 30,
    raw_window: int = 5,
    token_budget: int = 32000,
    output_json: bool = False,
) -> None:
    """Run benchmark and output results (JSON or pretty-printed)."""
    result = await run_benchmark(num_turns, raw_window, token_budget)

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_report(
            result, turns=result._turn_metrics, raw_window=raw_window, token_budget=token_budget
        )


def main() -> None:
    logging.getLogger("sr2.pipeline.engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="SR2 Token Savings Benchmark")
    parser.add_argument("--turns", type=int, default=30, help="Number of conversation turns")
    parser.add_argument("--raw-window", type=int, default=5, help="Raw window size")
    parser.add_argument("--budget", type=int, default=32000, help="Token budget")
    parser.add_argument(
        "--json", action="store_true", dest="output_json", help="Output results as JSON"
    )
    args = parser.parse_args()

    asyncio.run(run_and_print(args.turns, args.raw_window, args.budget, args.output_json))


if __name__ == "__main__":
    main()
