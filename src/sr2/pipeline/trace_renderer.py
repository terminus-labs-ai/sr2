"""CLI trace renderers for the SR2 Pipeline Inspector.

Renders TurnTrace objects as formatted strings for terminal output.
Three verbosity levels: brief (one-line), default (multi-section), full (with details).
"""

from __future__ import annotations

from sr2.pipeline.trace import TurnTrace


def render_default(trace: TurnTrace) -> str:
    """Render a multi-line trace summary with conditional sections."""
    lines: list[str] = []
    lines.append(f"━━━ Turn {trace.turn_number} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    # INPUT section
    inp = trace.get("input")
    if inp:
        d = inp.data
        lines.append("► INPUT")
        trigger = d.get("trigger_input", "")
        if len(trigger) > 80:
            trigger = trigger[:77] + "..."
        lines.append(f'  User: "{trigger}"')
        lines.append(f"  Session: {d.get('session_turns', 0)} turns, session_id={d.get('session_id', '')}")
        lines.append(f"  Interface: {d.get('interface_name', '')}")
        lines.append("")

    # PIPELINE section (from "resolve" event)
    resolve = trace.get("resolve")
    if resolve:
        d = resolve.data
        lines.append(f"▼ PIPELINE ({resolve.duration_ms:.0f}ms)")
        # Layer table
        lines.append("  ┌─────────────────────────────────────────────────────────────┐")
        lines.append("  │ Layer          Tokens   Cache     Source                    │")
        lines.append("  │─────────────────────────────────────────────────────────────│")
        for layer in d.get("layers", []):
            name = layer.get("name", "?")
            tokens = layer.get("tokens", 0)
            cache = layer.get("cache_status", "?").upper()
            cb = layer.get("circuit_breaker", "closed")
            items = layer.get("items", 0)
            if cb == "open":
                cache_str = "⚡ CIRCUIT OPEN"
                source_str = ""
            else:
                cache_str = f"{cache} {'✓' if cache == 'HIT' else ''}"
                source_str = f"{items} items"
            lines.append(f"  │ {name:<14} {tokens:>6,}   {cache_str:<9} {source_str:<24} │")
        lines.append("  │─────────────────────────────────────────────────────────────│")
        total = d.get("total_tokens", 0)
        budget = d.get("budget", 0)
        pct = d.get("utilization", 0) * 100
        tight = " ⚠ TIGHT" if pct > 90 else ""
        lines.append(f"  │ Total         {total:>7,} / {budget:,} ({pct:.0f}%){tight:<16} │")
        lines.append("  └─────────────────────────────────────────────────────────────┘")
        lines.append("")

    # MEMORY RETRIEVAL section
    retrieve = trace.get("retrieve")
    if retrieve:
        d = retrieve.data
        lines.append(f"  Memory retrieval ({d.get('latency_ms', 0):.0f}ms):")
        lines.append(
            f'    "{d.get("query", "")[:60]}" → '
            f'{d.get("results_returned", 0)} results / {d.get("candidates_scored", 0)} candidates'
        )
        selected = [r for r in d.get("results", []) if r.get("selected")]
        for i, r in enumerate(selected):
            prefix = "└─" if i == len(selected) - 1 else "├─"
            lines.append(f"    {prefix} {r.get('key', '?'):<20} {r.get('relevance_score', 0):.2f}")
        lines.append("")

    # ZONES section
    zones = trace.get("zones")
    if zones:
        d = zones.data
        s = d.get("summarized", {})
        c = d.get("compacted", {})
        r = d.get("raw", {})
        lines.append("  Conversation zones:")
        lines.append(
            f"    summarized [{s.get('turns', 0)} turns, {s.get('tokens', 0)} tk] → "
            f"compacted [{c.get('turns', 0)} turns, {c.get('tokens', 0)} tk] → "
            f"raw [{r.get('turns', 0)} turns, {r.get('tokens', 0)} tk]"
        )
        lines.append("")

    # TOOL STATE section
    ts = trace.get("tool_state")
    if ts:
        d = ts.data
        allowed = len(d.get("allowed_tools", []))
        denied = len(d.get("denied_tools", []))
        lines.append(f"  Tool state: {d.get('current_state', '?')} → {allowed} tools allowed, {denied} masked")
        lines.append("")

    # LLM REQUEST section
    llm_req = trace.get("llm_request")
    if llm_req:
        d = llm_req.data
        lines.append("▼ LLM REQUEST")
        if d.get("provider"):
            lines.append(f"  Provider: {d['provider']}")
        lines.append(f"  Messages: {d.get('message_count', 0)}")
        lines.append(f"  Tool schemas: {d.get('tool_count', 0)} tools")
        lines.append("")

    # LLM RESPONSE section
    llm_resp = trace.get("llm_response")
    if llm_resp:
        d = llm_resp.data
        latency = d.get("latency_ms", llm_resp.duration_ms)
        lines.append(f"▲ LLM RESPONSE ({latency:.0f}ms)")
        preview = d.get("content_preview", "")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        tokens = d.get("content_tokens", 0)
        lines.append(f'  Content: "{preview}" ({tokens} tokens)')
        tool_calls = d.get("tool_calls", [])
        if tool_calls:
            lines.append(f"  Tool calls: {len(tool_calls)}")
            for tc in tool_calls:
                if isinstance(tc, str):
                    lines.append(f"    └─ {tc}()")
                else:
                    lines.append(f"    └─ {tc.get('name', '?')}({tc.get('arguments_preview', '')})")
        lines.append("")

    # POST-PROCESS section
    pp = trace.get("post_process")
    if pp:
        d = pp.data
        lines.append("▼ POST-PROCESS")
        me = d.get("memory_extraction")
        if me:
            lines.append(
                f"  Memory extraction: {me.get('memories_extracted', 0)} extracted, "
                f"{me.get('conflicts_detected', 0)} conflicts"
            )
        comp = d.get("compaction")
        if comp:
            saved = comp.get("tokens_saved", 0)
            lines.append(f"  Compaction: {comp.get('turns_compacted', 0)} turns, {saved:,} tk saved")
        summ = d.get("summarization")
        if summ and summ.get("triggered"):
            lines.append(f"  Summarization: {summ.get('original_tokens', 0)} → {summ.get('summary_tokens', 0)} tk")
        lines.append("")

    # WARNINGS section
    warnings = trace.warnings
    if warnings:
        lines.append("  ⚠ WARNINGS")
        for i, w in enumerate(warnings):
            prefix = "└─" if i == len(warnings) - 1 else "├─"
            lines.append(f"    {prefix} {w}")
        lines.append("")

    # METRICS section
    metrics = trace.get("metrics")
    if metrics:
        d = metrics.data
        savings = d.get("token_savings_this_turn", 0)
        cache = d.get("cache_efficiency", 0) * 100
        deg = d.get("degradation_level", 0)
        lines.append("▼ METRICS")
        lines.append(f"  Token savings: {savings:,} tk | Cache: {cache:.0f}% | Degradation: level {deg}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def render_full(trace: TurnTrace) -> str:
    """Render a full trace with extra detail (compaction breakdown)."""
    base = render_default(trace)
    extra: list[str] = []

    pp = trace.get("post_process")
    if pp:
        comp = pp.data.get("compaction")
        if comp and comp.get("details"):
            extra.append("")
            extra.append("  Compaction detail:")
            for detail in comp["details"]:
                rule = detail.get("rule", "none")
                orig = detail.get("original_tokens", 0)
                comp_tk = detail.get("compacted_tokens", 0)
                extra.append(
                    f"    Turn {detail.get('turn_number', '?')} "
                    f"({detail.get('content_type', '?')}, {orig} → {comp_tk} tk)"
                )
                extra.append(f"      Rule: {rule}")

    if extra:
        # Insert before the final separator line
        parts = base.rsplit("━━━", 1)
        return parts[0] + "\n".join(extra) + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    return base


def render_brief(trace: TurnTrace) -> str:
    """Render a single-line trace summary."""
    resolve = trace.get("resolve")
    retrieve = trace.get("retrieve")
    pp = trace.get("post_process")
    llm_resp = trace.get("llm_response")

    total = resolve.data.get("total_tokens", 0) if resolve else 0
    budget = resolve.data.get("budget", 0) if resolve else 0
    pct = (total / budget * 100) if budget > 0 else 0

    mem_count = retrieve.data.get("results_returned", 0) if retrieve else 0

    compacted = 0
    saved = 0
    if pp and pp.data.get("compaction"):
        compacted = pp.data["compaction"].get("turns_compacted", 0)
        saved = pp.data["compaction"].get("tokens_saved", 0)

    cache_pct = ((resolve.data.get("cache_efficiency") or 0) * 100) if resolve else 0

    pipeline_ms = resolve.duration_ms if resolve else 0
    llm_ms = llm_resp.duration_ms if llm_resp else 0

    saved_str = f" (-{saved:,} tk)" if saved > 0 else ""
    line = (
        f"T{trace.turn_number} │ {total:,}/{budget:,} tk ({pct:.0f}%) │ "
        f"{mem_count} mem │ {compacted} compacted{saved_str} │ "
        f"cache {cache_pct:.0f}% │ {pipeline_ms:.0f}ms+{llm_ms:.0f}ms"
    )

    warnings = trace.warnings
    if warnings:
        first = warnings[0].split("—")[0].strip() if "—" in warnings[0] else warnings[0][:30]
        line += f" ⚠ {first}"

    return line
