#!/usr/bin/env python3
"""Benchmark 3: Cost

Shows actual billed prompt_tokens from real APIs by sending naive vs managed
context per turn with max_tokens=1.

Usage:
    python benchmarks/cost.py [--models MODEL1,MODEL2] [--turns N]

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
Without a key, prints a skip message and exits 0.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys

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
from _shared.reporting import W, print_header

from sr2.resolvers.registry import ResolverContext


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# Input pricing per 1M tokens (USD)
PRICING = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "claude-sonnet-4-20250514": 3.00,
    "claude-haiku-3-5-20241022": 0.80,
}


def _get_available_models() -> list[tuple[str, str]]:
    """Return list of (model_id, provider) that have valid API keys."""
    models = []
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401
            models.append(("gpt-4o-mini", "openai"))
        except ImportError:
            pass
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            models.append(("claude-haiku-3-5-20241022", "anthropic"))
        except ImportError:
            pass
    return models


def _send_and_measure_openai(model: str, context: str) -> int:
    """Send context to OpenAI with max_tokens=1, return prompt_tokens."""
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=1,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": "ok"},
        ],
    )
    return resp.usage.prompt_tokens


def _send_and_measure_anthropic(model: str, context: str) -> int:
    """Send context to Anthropic with max_tokens=1, return input_tokens."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=1,
        system=context,
        messages=[{"role": "user", "content": "ok"}],
    )
    return resp.usage.input_tokens


def _measure(model: str, provider: str, context: str) -> int:
    if provider == "openai":
        return _send_and_measure_openai(model, context)
    else:
        return _send_and_measure_anthropic(model, context)


async def run_benchmark(
    model_filter: str | None = None,
    num_turns: int = 30,
) -> None:
    available = _get_available_models()
    if not available:
        print()
        print("  COST BENCHMARK — SKIPPED")
        print("  No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        print("  Example: OPENAI_API_KEY=sk-... python benchmarks/cost.py")
        print()
        return

    # Filter models if specified, or use custom models directly
    if model_filter:
        requested = [m.strip() for m in model_filter.split(",")]
        # First try matching against known defaults
        filtered = [(m, p) for m, p in available if m in requested]
        if filtered:
            available = filtered
        else:
            # Custom model names (e.g., Ollama models) — infer provider from API key
            available = []
            for m in requested:
                if m.startswith("claude"):
                    available.append((m, "anthropic"))
                elif os.environ.get("OPENAI_API_KEY"):
                    available.append((m, "openai"))
                elif os.environ.get("ANTHROPIC_API_KEY"):
                    available.append((m, "anthropic"))
            if not available:
                print(f"  No matching models found for: {model_filter}")
                return

    print_header(f"SR2 — COST BENCHMARK ({num_turns} turns)")
    print()

    conversation = generate_conversation(num_turns)

    for model_id, provider in available:
        print(f"  Testing {model_id} ({provider})...")

        pipeline = create_benchmark_pipeline(raw_window=5, token_budget=32000)
        naive_history: list[str] = []
        total_naive_tokens = 0
        total_managed_tokens = 0
        prev_prefix = ""
        prefix_hits = 0

        for turn in conversation:
            # ── Naive context ──
            naive_history.append(f"{turn.role}: {turn.content}")
            naive_ctx = (
                SYSTEM_PROMPT + "\n\n" + TOOL_DEFINITIONS + "\n\n" + "\n".join(naive_history)
            )

            # ── Managed context ──
            pipeline.conversation_manager.add_turn(turn)
            pipeline.conversation_manager.run_compaction()

            zones = pipeline.conversation_manager.zones()
            session_parts: list[dict[str, str]] = []
            for s in zones.summarized:
                session_parts.append({"role": "system", "content": f"[Summary] {s}"})
            for t in zones.compacted:
                session_parts.append({"role": t.role, "content": t.content})
            for t in zones.raw:
                session_parts.append({"role": t.role, "content": t.content})

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

            # Track prefix stability across core + memory layers
            prefix_content = ""
            for layer_name in ["core", "memory"]:
                if layer_name in compiled.layers:
                    prefix_content += "".join(c.content for c in compiled.layers[layer_name])
            prefix = hashlib.sha256(prefix_content.encode()).hexdigest()[:16]
            if prev_prefix and prefix == prev_prefix:
                prefix_hits += 1
            prev_prefix = prefix

            # Measure actual billed tokens
            naive_billed = _measure(model_id, provider, naive_ctx)
            managed_billed = _measure(model_id, provider, compiled.content)
            total_naive_tokens += naive_billed
            total_managed_tokens += managed_billed

        # Results
        saved_pct = (
            (total_naive_tokens - total_managed_tokens) / total_naive_tokens * 100
            if total_naive_tokens
            else 0
        )
        prefix_hit_rate = prefix_hits / (len(conversation) - 1) if len(conversation) > 1 else 0
        price = PRICING.get(model_id, 1.0)
        naive_cost = total_naive_tokens / 1_000_000 * price
        managed_cost = total_managed_tokens / 1_000_000 * price

        print()
        print("─" * W)
        print(
            f"  {'Provider':<20} {'Naive Input':>12} {'Managed':>12} "
            f"{'Saved':>8} {'Cost Naive':>12} {'Cost Managed':>12}"
        )
        print(
            f"  {'─' * 20} {'─' * 12} {'─' * 12} "
            f"{'─' * 8} {'─' * 12} {'─' * 12}"
        )
        print(
            f"  {model_id:<20} {total_naive_tokens:>12,} {total_managed_tokens:>12,} "
            f"{saved_pct:>7.1f}% ${naive_cost:>11.3f} ${managed_cost:>11.3f}"
        )
        print()
        print(f"  Saved: ${naive_cost - managed_cost:.3f} ({saved_pct:.1f}% reduction)")
        print(f"  KV-cache prefix hit rate: {prefix_hit_rate * 100:.0f}%")
        print()


def main() -> None:
    logging.getLogger("sr2.pipeline.engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="SR2 Cost Benchmark")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model IDs (e.g., gpt-4o-mini,claude-haiku-3-5-20241022)",
    )
    parser.add_argument("--turns", type=int, default=30, help="Number of conversation turns")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.models, args.turns))


if __name__ == "__main__":
    main()
