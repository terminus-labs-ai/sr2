#!/usr/bin/env python3
"""Benchmark 2: Coherence

Proves that SR2's managed context retains anchor decisions that naive
truncation loses. Requires an LLM API key.

Usage:
    python benchmarks/coherence.py [--model MODEL] [--turns N] [--budget N]

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
Without a key, prints a skip message and exits 0.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
from dataclasses import dataclass

# Ensure benchmarks/ is on sys.path so _shared can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _shared  # noqa: F401  — sets up sys.path for sr2

from _shared.conversation_data import (
    RECALL_QUESTIONS,
    RETRIEVED_MEMORIES,
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    generate_long_conversation,
)
from _shared.pipeline_factory import create_benchmark_pipeline
from _shared.reporting import W, print_header, print_section

from sr2.resolvers.registry import ResolverContext


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _get_llm_client(model: str):
    """Return (ask_fn, provider_name) or None if no API key."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if model.startswith("claude") or (anthropic_key and not openai_key):
        if not anthropic_key:
            return None
        try:
            import anthropic
        except ImportError:
            print("  anthropic package not installed. pip install anthropic")
            return None

        client = anthropic.Anthropic(api_key=anthropic_key)
        resolved_model = model if model.startswith("claude") else "claude-sonnet-4-20250514"

        async def ask_anthropic(system: str, prompt: str) -> str:
            resp = client.messages.create(
                model=resolved_model,
                max_tokens=200,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

        return ask_anthropic, "Anthropic"

    if openai_key:
        try:
            import openai
        except ImportError:
            print("  openai package not installed. pip install openai")
            return None

        client = openai.OpenAI(api_key=openai_key)
        resolved_model = model if model != "default" else "gpt-4o-mini"

        async def ask_openai(system: str, prompt: str) -> str:
            resp = client.chat.completions.create(
                model=resolved_model,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content

        return ask_openai, "OpenAI"

    return None


def _build_naive_context(turns, token_budget: int) -> str:
    """Build naive context: system prompt + truncated history (drop oldest)."""
    system = SYSTEM_PROMPT + "\n\n" + TOOL_DEFINITIONS
    system_tokens = _estimate_tokens(system)
    remaining_budget = token_budget - system_tokens

    # Build history lines from newest to oldest, keep what fits
    history_lines: list[str] = []
    for turn in reversed(turns):
        line = f"{turn.role}: {turn.content}"
        line_tokens = _estimate_tokens(line)
        if remaining_budget - line_tokens < 0:
            break
        history_lines.insert(0, line)
        remaining_budget -= line_tokens

    return system + "\n\n" + "\n".join(history_lines)


@dataclass
class ManagedContextResult:
    content: str
    prefix_hit_rate: float
    zone_raw_tokens: int
    zone_compacted_tokens: int
    zone_summarized_tokens: int


async def _build_managed_context(turns, token_budget: int, llm_callable) -> ManagedContextResult:
    """Build managed context using real SR2 pipeline with compaction + summarization.

    Compiles per-turn so we can track KV-cache prefix stability and zone breakdown.
    """
    pipeline = create_benchmark_pipeline(
        token_budget=token_budget,
        llm_callable=llm_callable,
    )

    prev_prefix = ""
    prefix_hits = 0

    for turn in turns:
        pipeline.conversation_manager.add_turn(turn)
        pipeline.conversation_manager.run_compaction()
        await pipeline.conversation_manager.run_summarization()

        # Compile each turn to track prefix stability
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

    # Final zone breakdown
    zones = pipeline.conversation_manager.zones()
    prefix_hit_rate = prefix_hits / (len(turns) - 1) if len(turns) > 1 else 0
    return ManagedContextResult(
        content=compiled.content,
        prefix_hit_rate=prefix_hit_rate,
        zone_raw_tokens=sum(_estimate_tokens(t.content) for t in zones.raw),
        zone_compacted_tokens=sum(_estimate_tokens(t.content) for t in zones.compacted),
        zone_summarized_tokens=sum(_estimate_tokens(s) for s in zones.summarized),
    )


async def run_benchmark(
    model: str = "default",
    num_turns: int = 50,
    token_budget: int = 8000,
) -> None:
    result = _get_llm_client(model)
    if result is None:
        print()
        print("  COHERENCE BENCHMARK — SKIPPED")
        print("  No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")
        print("  Example: OPENAI_API_KEY=sk-... python benchmarks/coherence.py")
        print()
        return

    ask_fn, provider = result

    print_header("SR2 — COHERENCE BENCHMARK")
    print(f"  Provider: {provider}  |  Model: {model}")
    print(f"  Turns: {num_turns}  |  Token budget: {token_budget:,}")
    print()

    # Generate conversation with anchor decisions
    conversation = generate_long_conversation(num_turns)

    print("  Building naive context (truncated)...")
    naive_ctx = _build_naive_context(conversation, token_budget)
    naive_tokens = _estimate_tokens(naive_ctx)

    print("  Building managed context (SR2 pipeline)...")
    managed = await _build_managed_context(conversation, token_budget, ask_fn)
    managed_tokens = _estimate_tokens(managed.content)

    print(f"  Naive context: {naive_tokens:,} tokens")
    print(f"  Managed context: {managed_tokens:,} tokens")
    print(f"  KV-cache prefix hit rate: {managed.prefix_hit_rate * 100:.0f}%")
    print()

    # Zone breakdown — shows summarization doing work
    zt = managed.zone_raw_tokens + managed.zone_compacted_tokens + managed.zone_summarized_tokens
    if zt:
        print_section("THREE-ZONE BREAKDOWN (final managed context)")
        for label, val in [
            ("Raw (recent, verbatim)", managed.zone_raw_tokens),
            ("Compacted (tool refs)", managed.zone_compacted_tokens),
            ("Summarized (LLM digest)", managed.zone_summarized_tokens),
        ]:
            print(f"  {label:<28} {val:>6,} tokens  ({val / zt * 100:>5.1f}%)")
        print()

    # Run recall questions
    system_prompt = "Answer based only on the conversation context provided. Be concise."

    naive_hits = 0
    managed_hits = 0

    print("─" * W)
    print(f"  {'Question':<40} {'Naive':>8} {'Managed':>8}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8}")

    # Only test questions for anchors within our turn range
    active_questions = [(t, q, k) for t, q, k in RECALL_QUESTIONS if t < num_turns]

    for turn_num, question, keyword in active_questions:
        # Test naive
        naive_answer = await ask_fn(
            system_prompt,
            f"Context:\n{naive_ctx}\n\nQuestion: {question}",
        )
        naive_hit = keyword.lower() in naive_answer.lower()
        if naive_hit:
            naive_hits += 1

        # Test managed
        managed_answer = await ask_fn(
            system_prompt,
            f"Context:\n{managed.content}\n\nQuestion: {question}",
        )
        managed_hit = keyword.lower() in managed_answer.lower()
        if managed_hit:
            managed_hits += 1

        naive_label = "HIT" if naive_hit else "MISS"
        managed_label = "HIT" if managed_hit else "MISS"
        print(f"  {question:<40} {naive_label:>8} {managed_label:>8}")

    total_q = len(active_questions)
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8}")
    print(f"  {'Score:':<40} {naive_hits}/{total_q:>5} {managed_hits}/{total_q:>5}")
    print()

    # Information density: recall per 1k tokens
    naive_density = (naive_hits / naive_tokens * 1000) if naive_tokens else 0
    managed_density = (managed_hits / managed_tokens * 1000) if managed_tokens else 0

    print_section("INFORMATION DENSITY")
    print(
        f"  Naive:    {naive_hits} recalls in {naive_tokens:,} tokens  "
        f"= {naive_density:.2f} recalls/1k tokens"
    )
    print(
        f"  Managed:  {managed_hits} recalls in {managed_tokens:,} tokens  "
        f"= {managed_density:.2f} recalls/1k tokens"
    )
    if managed_density > naive_density and naive_density > 0:
        print(f"  -> {managed_density / naive_density:.1f}x more information per token")
    elif managed_density > 0 and naive_density == 0:
        print("  -> Managed retains information that naive lost entirely")
    print()

    print("=" * W)
    print("  SUMMARY")
    print("=" * W)
    print(f"  Naive recall:   {naive_hits}/{total_q}")
    print(f"  Managed recall: {managed_hits}/{total_q}")
    print(f"  KV-cache prefix hit rate: {managed.prefix_hit_rate * 100:.0f}%")
    if managed_density > 0:
        print(f"  Information density: {managed_density:.2f} vs {naive_density:.2f} recalls/1k tok")
    if managed_hits > naive_hits:
        print(f"  -> SR2 retained {managed_hits - naive_hits} more decisions than naive truncation")
    print()


def main() -> None:
    # Suppress noisy pipeline warnings during benchmark compilation loops.
    # The append_only cache policy warns when content changes, which is expected
    # since we recompile every turn to track prefix stability.
    logging.getLogger("sr2.pipeline.engine").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="SR2 Coherence Benchmark")
    parser.add_argument("--model", default="default", help="LLM model name")
    parser.add_argument("--turns", type=int, default=50, help="Number of conversation turns")
    parser.add_argument("--budget", type=int, default=8000, help="Token budget")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.model, args.turns, args.budget))


if __name__ == "__main__":
    main()
