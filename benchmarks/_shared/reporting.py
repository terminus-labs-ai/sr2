"""Terminal formatting utilities for benchmark reports."""

from __future__ import annotations

from dataclasses import dataclass, field

W = 74  # Standard report width


@dataclass
class TurnMetrics:
    """Per-turn metrics collected during a benchmark run."""

    turn_number: int
    role: str
    naive_tokens: int = 0
    managed_tokens: int = 0
    prefix_hash: str = ""
    prefix_stable: bool = False
    compaction_turns_compacted: int = 0
    compaction_original_tokens: int = 0
    compaction_compacted_tokens: int = 0
    zone_raw: int = 0
    zone_compacted: int = 0
    zone_summarized: int = 0
    layer_tokens: dict[str, int] = field(default_factory=dict)
    compile_us: float = 0.0


def print_header(title: str) -> None:
    print()
    print("=" * W)
    print(f"  {title}")
    print("=" * W)


def print_section(title: str) -> None:
    print("─" * W)
    print(f"  {title}")
    print("─" * W)


def print_token_table(turns: list[TurnMetrics]) -> None:
    """Print the per-turn naive vs managed table."""
    print_section("PER-TURN DETAIL")
    print(
        f"  {'#':>3}  {'Role':<12}  {'Naive':>7}  {'Mgd':>7}  "
        f"{'Saved':>6}  {'Cache':>5}  {'Raw':>6}  {'Cmpct':>6}"
    )
    print(
        f"  {'─' * 3}  {'─' * 12}  {'─' * 7}  {'─' * 7}  {'─' * 6}  {'─' * 5}  {'─' * 6}  {'─' * 6}"
    )
    print("  (Early turns: managed is larger because it includes retrieved memories)")
    print()

    for t in turns:
        saved = t.naive_tokens - t.managed_tokens
        saved_pct = (saved / t.naive_tokens * 100) if t.naive_tokens else 0
        cache = "✓" if t.prefix_stable else "·"
        print(
            f"  {t.turn_number:>3}  {t.role:<12}  "
            f"{t.naive_tokens:>7,}  {t.managed_tokens:>7,}  "
            f"{saved_pct:>5.1f}%  {cache:>5}  "
            f"{t.zone_raw:>6,}  {t.zone_compacted:>6,}"
        )

    print()


def print_compaction_events(turns: list[TurnMetrics]) -> None:
    events = [t for t in turns if t.compaction_turns_compacted > 0]
    if not events:
        return
    print_section("COMPACTION EVENTS")
    for t in events:
        orig = t.compaction_original_tokens
        comp = t.compaction_compacted_tokens
        ratio = (1 - comp / orig) * 100 if orig else 0
        print(
            f"  Turn {t.turn_number:>2}: {t.compaction_turns_compacted} tool outputs compacted  "
            f"{orig:>6,} -> {comp:>6,} tok  "
            f"({ratio:.0f}% reduction)"
        )
    print()


def print_layer_distribution(last: TurnMetrics) -> None:
    print_section("CONTEXT WINDOW LAYOUT (final turn)")
    total = sum(last.layer_tokens.values()) or 1
    labels = {
        "core": "Core (system+tools)",
        "memory": "Memory (retrieved)",
        "conversation": "Conversation (3-zone)",
    }
    for name, tok in last.layer_tokens.items():
        layer_pct = tok / total * 100
        bar_len = int(layer_pct / 2)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {labels.get(name, name):<24} {tok:>6,} tok  {bar} {layer_pct:>5.1f}%")
    print()


def print_three_zone_breakdown(last: TurnMetrics) -> None:
    zt = last.zone_raw + last.zone_compacted + last.zone_summarized
    print_section("THREE-ZONE BREAKDOWN (final conversation state)")
    if zt:
        for label, val in [
            ("Raw (recent, verbatim)", last.zone_raw),
            ("Compacted (tool refs)", last.zone_compacted),
            ("Summarized (LLM digest)", last.zone_summarized),
        ]:
            print(f"  {label:<28} {val:>6,} tokens  ({val / zt * 100:>5.1f}%)")
    print()


def print_growth_chart(turns: list[TurnMetrics]) -> None:
    print_section("TOKEN GROWTH OVER TIME")
    print(f"  {'Turn':>4}  {'Naive ctx':>10}  {'Managed ctx':>12}  {'Ratio':>6}  Chart")
    for t in turns:
        if t.turn_number % 5 == 0 or t.turn_number == len(turns) - 1:
            ratio = t.managed_tokens / t.naive_tokens if t.naive_tokens else 1
            n_bar = min(40, t.naive_tokens // 200)
            m_bar = min(40, t.managed_tokens // 200)
            print(
                f"  {t.turn_number:>4}  {t.naive_tokens:>10,}  "
                f"{t.managed_tokens:>12,}  {ratio:>5.2f}x  "
                f"{'▓' * n_bar}{'░' * (40 - n_bar)}"
            )
            print(f"  {'':>4}  {'':>10}  {'':>12}  {'':>6}  {'█' * m_bar}{'░' * (40 - m_bar)}")
    print()


def print_budget_compliance(
    turns: list[TurnMetrics],
    token_budget: int,
) -> None:
    """Show how many turns exceed the token budget: naive vs managed."""
    naive_over = [t for t in turns if t.naive_tokens > token_budget]
    managed_over = [t for t in turns if t.managed_tokens > token_budget]

    print_section(f"BUDGET COMPLIANCE (budget: {token_budget:,} tokens)")

    naive_pct = len(naive_over) / len(turns) * 100 if turns else 0
    managed_pct = len(managed_over) / len(turns) * 100 if turns else 0

    print(
        f"  Naive exceeded budget:    {len(naive_over):>3}/{len(turns)} turns  ({naive_pct:.0f}%)"
    )
    print(
        f"  Managed exceeded budget:  {len(managed_over):>3}/{len(turns)} turns  ({managed_pct:.0f}%)"
    )

    if naive_over:
        worst_naive = max(t.naive_tokens for t in turns)
        print(
            f"  Naive peak:               {worst_naive:>12,} tokens  ({worst_naive / token_budget:.1f}x budget)"
        )

    worst_managed = max(t.managed_tokens for t in turns) if turns else 0
    utilization = worst_managed / token_budget * 100 if token_budget else 0
    print(
        f"  Managed peak:             {worst_managed:>12,} tokens  ({utilization:.0f}% of budget)"
    )

    if not managed_over:
        print("  -> Managed context never exceeded the token budget")
    print()


def print_compile_overhead(turns: list[TurnMetrics]) -> None:
    """Show pipeline compile latency, framed relative to LLM round-trip."""
    if not turns:
        return
    times = [t.compile_us for t in turns]
    avg_us = sum(times) / len(times)
    max_us = max(times)
    min_us = min(times)

    print_section("PIPELINE OVERHEAD")
    print(f"  Avg compile time:  {avg_us:>8.0f} us/turn")
    print(f"  Min / Max:         {min_us:>8.0f} / {max_us:.0f} us")
    # Typical LLM round-trip: 500-2000ms = 500_000-2_000_000 us
    overhead_pct = avg_us / 500_000 * 100  # vs a fast 500ms LLM call
    print(f"  vs 500ms LLM call: {overhead_pct:.3f}% overhead")
    print("  -> Pipeline adds negligible latency to each turn")
    print()


def print_cost_projection(total_naive: int, total_managed: int) -> None:
    savings = total_naive - total_managed
    for name, rate in [("Claude Sonnet", 3.0), ("GPT-4o", 2.5), ("Ollama (local)", 0.0)]:
        if rate > 0:
            naive_cost = total_naive / 1_000_000 * rate
            mgd_cost = total_managed / 1_000_000 * rate
            print(
                f"  * {name}: ${naive_cost:.4f} -> ${mgd_cost:.4f} "
                f"(${naive_cost - mgd_cost:.4f} saved per session)"
            )
        else:
            print(f"  * {name}: {savings:,} fewer tokens = faster inference")


def print_headline_metrics(
    turns: list[TurnMetrics],
    total_naive: int,
    total_managed: int,
    prefix_stability: float,
    token_budget: int = 0,
) -> None:
    pct = (total_naive - total_managed) / total_naive * 100 if total_naive else 0
    last = turns[-1] if turns else None

    print("=" * W)
    print("  HEADLINE METRICS (for README / blog post)")
    print("=" * W)
    print(f"  * {pct:.0f}% cumulative token reduction over {len(turns)} turns")
    if last:
        final_pct = (
            ((last.naive_tokens - last.managed_tokens) / last.naive_tokens * 100)
            if last.naive_tokens
            else 0
        )
        print(f"  * {final_pct:.0f}% context window reduction by final turn")
    print(f"  * {prefix_stability * 100:.0f}% KV-cache prefix hit rate (stable across turns)")

    if token_budget:
        naive_over = sum(1 for t in turns if t.naive_tokens > token_budget)
        managed_over = sum(1 for t in turns if t.managed_tokens > token_budget)
        print(
            f"  * Budget compliance: naive {len(turns) - naive_over}/{len(turns)}, "
            f"managed {len(turns) - managed_over}/{len(turns)} turns within budget"
        )

    events = [t for t in turns if t.compaction_turns_compacted > 0]
    if events:
        avg_ratio = sum(
            (1 - t.compaction_compacted_tokens / t.compaction_original_tokens) * 100
            for t in events
            if t.compaction_original_tokens
        ) / len(events)
        print(f"  * {avg_ratio:.0f}% avg compaction ratio on tool outputs")

    if turns:
        avg_us = sum(t.compile_us for t in turns) / len(turns)
        print(
            f"  * {avg_us:.0f}us avg compile overhead ({avg_us / 500_000 * 100:.3f}% of a 500ms LLM call)"
        )

    print_cost_projection(total_naive, total_managed)

    if last:
        ratio = last.managed_tokens / last.naive_tokens if last.naive_tokens else 1
        print(
            f"  * Final turn: {ratio:.2f}x context size vs naive "
            f"({last.managed_tokens:,} vs {last.naive_tokens:,} tok)"
        )
    print()
