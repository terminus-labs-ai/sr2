"""Tracing model for SR2 pipeline firings.

Provides FiringRecord, Tracer protocol, CollectingTracer, and render_trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class FiringRecord:
    """Immutable record of a single component firing in the pipeline."""

    turn_seq: int
    firing_seq: int
    kind: Literal["resolver", "transformer", "tool_provider"]
    component_name: str
    layer: str
    trigger_events: list[str]
    content_before: list
    content_after: list
    tokens_before: int
    tokens_after: int
    tokens_delta: int
    duration_ms: float
    status: Literal["ok", "failed"] = "ok"
    error: str | None = None


@runtime_checkable
class Tracer(Protocol):
    """Protocol for objects that receive pipeline firing records."""

    def on_firing(self, record: FiringRecord) -> None: ...


class CollectingTracer:
    """In-memory tracer that collects all firing records in insertion order."""

    def __init__(self) -> None:
        self._buffer: list[FiringRecord] = []

    def on_firing(self, record: FiringRecord) -> None:
        self._buffer.append(record)

    def get_trace(self) -> list[FiringRecord]:
        """Return an independent copy of the collected records."""
        return list(self._buffer)

    def clear(self) -> None:
        """Empty the buffer."""
        self._buffer.clear()


def render_trace(records: list[FiringRecord]) -> str:
    """Render a human-readable timeline of pipeline firings.

    Groups records by turn_seq (sorted), then by firing_seq within each turn.
    """
    if not records:
        return "(no records)\n"

    lines: list[str] = []
    width = 57

    sorted_records = sorted(records, key=lambda r: (r.turn_seq, r.firing_seq))

    for turn_seq, turn_records in groupby(sorted_records, key=lambda r: r.turn_seq):
        # Turn header
        header = f"── Turn {turn_seq} "
        header = header + "─" * max(0, width - len(header))
        lines.append(header)

        for rec in turn_records:
            # Token delta with mandatory sign
            delta = rec.tokens_delta
            delta_str = f"+{delta}" if delta >= 0 else f"{delta}"

            # Status suffix
            status_suffix = " [FAILED]" if rec.status == "failed" else ""

            # Main firing line: #<seq>  [<layer>]  <kind>/<name>  <delta> tok  <dur>ms
            firing_line = (
                f"#{ rec.firing_seq}  [{rec.layer}]  {rec.component_name}"
                f"  {delta_str} tok  {rec.duration_ms:.3g}ms{status_suffix}"
            )
            lines.append(firing_line)

            # Error line (only when failed)
            if rec.status == "failed" and rec.error:
                lines.append(f"      error: {rec.error}")

            # Before line (only shown when empty)
            if not rec.content_before:
                lines.append("      before: (empty)")

            # After line — always shown
            if not rec.content_after:
                lines.append(f"      after:  [] (0 items)")
            else:
                items_repr = ", ".join(repr(item) for item in rec.content_after)
                lines.append(f"      after:  {items_repr}")

        lines.append("─" * width)

    return "\n".join(lines) + "\n"
