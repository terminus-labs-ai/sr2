#!/usr/bin/env python3
"""Benchmark 1: Token Savings

Measures cumulative token savings of SR2's managed pipeline vs naive
concatenation. Uses real library components — zero LLM dependencies.

Usage:
    python benchmarks/token_savings.py [--turns N] [--raw-window N] [--budget N]
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time

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


async def run_benchmark(
    num_turns: int = 30,
    raw_window: int = 5,
    token_budget: int = 32000,
) -> None:
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
    avg_compile = sum(t.compile_us for t in turns) / len(turns) if turns else 0

    # ── Report ──
    print_header("SR2 — TOKEN SAVINGS BENCHMARK")
    print(f"  Simulated turns: {len(turns)}")
    print(f"  Raw window: {raw_window}  |  Token budget: {token_budget:,}")
    print()

    savings = total_naive - total_managed
    pct = (savings / total_naive * 100) if total_naive else 0

    print_section("CUMULATIVE TOKEN USAGE (sum of all turns' context windows)")
    print(f"  Naive (concatenate all history):  {total_naive:>12,} tokens")
    print(f"  Managed (SR2 pipeline):      {total_managed:>12,} tokens")
    print(f"  Tokens saved:                     {savings:>12,} tokens  ({pct:.1f}%)")
    print()

    print_budget_compliance(turns, token_budget)

    print_section("KV-CACHE PREFIX STABILITY")
    print(f"  Core + memory prefix unchanged:  {prefix_stability * 100:.0f}% of turns")
    print(f"  -> Provider can reuse KV cache for the stable prefix portion")
    print()

    print_compile_overhead(turns)

    print_token_table(turns)
    print_compaction_events(turns)

    last = turns[-1] if turns else None
    if last:
        print_layer_distribution(last)
        print_three_zone_breakdown(last)

    print_growth_chart(turns)
    print_headline_metrics(turns, total_naive, total_managed, prefix_stability, token_budget)


def main() -> None:
    logging.getLogger("sr2.pipeline.engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="SR2 Token Savings Benchmark")
    parser.add_argument("--turns", type=int, default=30, help="Number of conversation turns")
    parser.add_argument("--raw-window", type=int, default=5, help="Raw window size")
    parser.add_argument("--budget", type=int, default=32000, help="Token budget")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.turns, args.raw_window, args.budget))


if __name__ == "__main__":
    main()
