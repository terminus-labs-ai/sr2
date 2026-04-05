"""LLM subpackage — client, loop, context bridge, and streaming."""

from sr2_runtime.llm.client import LLMClient, LLMResponse
from sr2_runtime.llm.context_bridge import ContextBridge
from sr2_runtime.llm.loop import LLMLoop, LoopResult, ToolCallRecord
from sr2_runtime.llm.streaming import (
    StreamCallback,
    StreamEndEvent,
    StreamEvent,
    StreamRetractEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ContextBridge",
    "LLMLoop",
    "LoopResult",
    "ToolCallRecord",
    "StreamCallback",
    "StreamEndEvent",
    "StreamEvent",
    "StreamRetractEvent",
    "TextDeltaEvent",
    "ToolResultEvent",
    "ToolStartEvent",
]
