"""LLMCompactionStrategy: use an LLM to compact a list of Messages."""

from __future__ import annotations

from typing import Any, Callable

from sr2.models import ContentBlock, Message, TextBlock


class LLMCompactionStrategy:
    """Compact a conversation history using an LLM callable.

    Parameters
    ----------
    llm:
        An async LLM callable with a ``complete(request)`` coroutine.
        The ``complete`` method must accept a request-like object and
        return a response with a ``content`` attribute.
    rules:
        Optional list of rule callables applied before LLM compaction
        (pre-filtering).  Defaults to no rules.
    """

    def __init__(
        self,
        llm: Any,
        rules: list[Callable[[ContentBlock], ContentBlock | None]] | None = None,
    ) -> None:
        self._llm = llm
        self._rules = rules or []

    async def compact(self, messages: list[Message]) -> list[Message]:
        """Compact *messages* using the LLM.

        - An empty list is returned immediately without calling the LLM.
        - Non-empty lists are sent to the LLM for compaction; the result
          is returned as a list of Messages.
        """
        if not messages:
            return []

        # Build a plain-text representation of the conversation for the LLM
        conversation_text = self._render(messages)

        # Build a minimal request object accepted by the LLM callable
        request = _CompletionRequest(conversation_text)
        response = await self._llm.complete(request)

        # Extract the summary text from the LLM response
        summary_text = ""
        for block in response.content:
            if isinstance(block, TextBlock):
                summary_text = block.text
                break

        if not summary_text:
            return messages

        compacted_message = Message(
            role="user",
            content=[TextBlock(text=summary_text)],
        )
        return [compacted_message]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render(messages: list[Message]) -> str:
        """Render messages as a plain-text conversation."""
        parts: list[str] = []
        for msg in messages:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(f"{msg.role}: {block.text}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal minimal request type — avoids importing CompletionRequest from
# sr2.protocols.llm (which may pull in heavy dependencies) while still
# giving the LLM callable something it can inspect.
# ---------------------------------------------------------------------------


class _CompletionRequest:
    """Minimal completion request wrapper."""

    def __init__(self, text: str) -> None:
        self.messages = [
            Message(role="user", content=[TextBlock(text=text)])
        ]
        self.system: list[ContentBlock] = [
            TextBlock(text="Summarize the following conversation concisely.")
        ]
