"""State types for SR2-powered LangGraph graphs."""

from __future__ import annotations

from typing import Any, TypedDict


class SR2GraphState(TypedDict, total=False):
    """Base state for SR2-powered LangGraph graphs.

    LangGraph owns inter-agent data flow.
    The SR2 runtime owns intra-agent data (context, memory, sessions).
    """

    current_task: str
    prior_output: str | None
    outputs: dict[str, Any]
    metrics: dict[str, Any]
    iteration: int
    metadata: dict[str, Any]
