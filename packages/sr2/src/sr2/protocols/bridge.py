"""Protocol for context-to-messages bridge."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sr2.pipeline.engine import CompiledContext


@runtime_checkable
class ContextBridgeProtocol(Protocol):
    """Converts compiled context into LLM message arrays."""

    def build_messages(
        self,
        compiled: CompiledContext,
        session_turns: list[dict],
        current_input: str | None = None,
    ) -> list[dict]: ...

    def append_tool_result(
        self,
        messages: list[dict],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict]: ...

    def append_assistant_tool_calls(
        self,
        messages: list[dict],
        content: str | None,
        tool_calls: list[dict],
        raw_tool_call_text: str = "",
    ) -> list[dict]: ...
