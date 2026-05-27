"""CompactionEngine: applies a pipeline of compaction rules to Messages."""

from __future__ import annotations

from typing import Callable

from sr2.models import ContentBlock, Message, ToolResultBlock


class CompactionEngine:
    """Apply a list of block-level compaction rules to a list of Messages.

    Rules are callables with the signature::

        rule(block: ContentBlock) -> ContentBlock | None

    A rule returns a replacement block when it applies, or None when it
    does not.  Rules are tried in order; the first rule that returns a
    non-None value wins for that block.

    Parameters
    ----------
    rules:
        Ordered list of rule callables to apply.
    """

    def __init__(self, rules: list[Callable[[ContentBlock], ContentBlock | None]]) -> None:
        self._rules = rules

    def apply(self, messages: list[Message]) -> list[Message]:
        """Apply all rules to each block in every message.

        Returns the original list unchanged if no rule produced any
        transformation.  Otherwise returns a new list of Messages with
        compacted blocks in place.
        """
        if not self._rules:
            return messages

        result: list[Message] = []
        any_changed = False

        for msg in messages:
            new_content: list[ContentBlock] = []
            msg_changed = False

            for block in msg.content:
                replacement = self._apply_rules(block)
                if replacement is not None:
                    new_content.append(replacement)
                    msg_changed = True
                    any_changed = True
                else:
                    new_content.append(block)

            if msg_changed:
                result.append(msg.model_copy(update={"content": new_content}))
            else:
                result.append(msg)

        return result if any_changed else messages

    def _apply_rules(self, block: ContentBlock) -> ContentBlock | None:
        """Return the first non-None result from the rule list, or None."""
        for rule in self._rules:
            replacement = rule(block)
            if replacement is not None:
                return replacement
        return None
