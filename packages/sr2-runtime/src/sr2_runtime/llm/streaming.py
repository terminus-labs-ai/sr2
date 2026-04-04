"""Stream event types for LLM streaming through to interface plugins."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class TextDeltaEvent:
    """A chunk of text from the LLM stream."""

    content: str


@dataclass
class ToolStartEvent:
    """Emitted when the LLM invokes a tool."""

    tool_name: str
    tool_call_id: str
    arguments: dict = field(default_factory=dict)


@dataclass
class ToolResultEvent:
    """Emitted after a tool finishes execution."""

    tool_name: str
    tool_call_id: str
    result: str
    success: bool


@dataclass
class StreamRetractEvent:
    """Emitted when streamed text turns out to be a tool call, not user-facing content.

    Interface plugins should delete or replace the previously streamed text
    with appropriate tool-status messaging.
    """

    retracted_text: str


@dataclass
class StreamEndEvent:
    """Emitted when the full streaming response is complete."""

    full_text: str


StreamEvent = (
    TextDeltaEvent | ToolStartEvent | ToolResultEvent | StreamRetractEvent | StreamEndEvent
)
StreamCallback = Callable[[StreamEvent], Awaitable[None]]
