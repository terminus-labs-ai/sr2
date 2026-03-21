"""Detect tool call loops — same tool called repeatedly with identical or near-identical args."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.llm.loop import ToolCallRecord

logger = logging.getLogger(__name__)


@dataclass
class LoopDetection:
    """Result of loop detection analysis."""

    detected: bool
    tool_name: str = ""
    pattern: str = ""  # "identical_args" | "same_tool_dominant"
    count: int = 0


def _hash_args(arguments: dict) -> str:
    """Deterministic hash of tool arguments for comparison."""
    return hashlib.md5(
        json.dumps(arguments, sort_keys=True, default=str).encode()
    ).hexdigest()


def detect_loop(
    tool_calls: list[ToolCallRecord],
    window: int = 6,
    threshold: int = 3,
) -> LoopDetection:
    """Analyze recent tool call history for repetitive patterns.

    Args:
        tool_calls: Full history of tool calls so far.
        window: Number of recent calls to examine.
        threshold: Minimum repetitions to trigger detection.

    Returns:
        LoopDetection with detected=True if a loop is found.
    """
    if len(tool_calls) < threshold:
        return LoopDetection(detected=False)

    recent = tool_calls[-window:]

    # Rule 1: Identical (tool_name, args_hash) repeated threshold+ times
    sig_counts: dict[tuple[str, str], int] = {}
    for tc in recent:
        sig = (tc.tool_name, _hash_args(tc.arguments))
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

    for (tool_name, _), count in sig_counts.items():
        if count >= threshold:
            return LoopDetection(
                detected=True,
                tool_name=tool_name,
                pattern="identical_args",
                count=count,
            )

    # Rule 2: Same tool dominates the window AND results are similar
    name_counts: dict[str, int] = {}
    for tc in recent:
        name_counts[tc.tool_name] = name_counts.get(tc.tool_name, 0) + 1

    for tool_name, count in name_counts.items():
        if count >= threshold:
            # Check if results are all similar (same first 200 chars)
            tool_results = [tc.result[:200] for tc in recent if tc.tool_name == tool_name]
            unique_results = set(tool_results)
            if len(unique_results) <= 1:
                return LoopDetection(
                    detected=True,
                    tool_name=tool_name,
                    pattern="same_tool_dominant",
                    count=count,
                )

    return LoopDetection(detected=False)
