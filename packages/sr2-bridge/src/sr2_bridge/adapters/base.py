"""Protocol definition for bridge adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sr2.compaction.engine import ConversationTurn


@runtime_checkable
class BridgeAdapter(Protocol):
    """Translates between a wire format (Anthropic, OpenAI) and SR2 internals."""

    def extract_messages(self, body: dict) -> tuple[str | None, list[dict]]:
        """Extract (system_prompt, messages) from the request body.

        Returns:
            Tuple of (system_prompt_text_or_None, list_of_message_dicts).
            Each message dict has at minimum {"role": str, "content": ...}.
        """
        ...

    def rebuild_body(
        self,
        original_body: dict,
        optimized_messages: list[dict],
        system_injection: str | None,
    ) -> dict:
        """Rebuild the request body with optimized messages.

        Args:
            original_body: The original request body (to preserve non-message fields).
            optimized_messages: The optimized message list.
            system_injection: Optional text to prepend to the system prompt
                (e.g. summarization context).

        Returns:
            A new request body dict ready to forward upstream.
        """
        ...

    def parse_sse_text(self, chunk: bytes) -> str | None:
        """Extract assistant text from a raw SSE chunk.

        Args:
            chunk: A single SSE data line (bytes, as received from upstream).

        Returns:
            The extracted text fragment, or None if this chunk has no text.
        """
        ...

    def messages_to_turns(
        self,
        messages: list[dict],
        turn_counter_start: int,
    ) -> list[ConversationTurn]:
        """Convert wire-format messages to SR2 ConversationTurns.

        Args:
            messages: List of message dicts in the adapter's wire format.
            turn_counter_start: Starting turn number for numbering.

        Returns:
            List of ConversationTurns with metadata["_original_message"] preserved.
        """
        ...

    def turns_to_messages(
        self,
        turns: list[ConversationTurn],
        original_messages: list[dict],
    ) -> list[dict]:
        """Convert SR2 ConversationTurns back to wire-format messages.

        For compacted turns, emits plain text content. For raw turns that
        still have their original structure, preserves it.

        Args:
            turns: The ConversationTurns to convert.
            original_messages: The original messages for fallback structure.

        Returns:
            List of wire-format message dicts.
        """
        ...
