"""Content resolver that returns compacted conversation history."""

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.resolvers.registry import ResolvedContent, ResolverContext


class CompactionResolver:
    """Content resolver that returns compacted conversation history."""

    def __init__(self, engine: CompactionEngine):
        self._engine = engine

    async def resolve(
        self,
        key: str,
        config: dict,
        context: ResolverContext,
    ) -> ResolvedContent:
        """Resolve by running compaction on session history.

        1. Get conversation turns from context.agent_config["session_history"]
        2. Convert to ConversationTurn objects
        3. Run compaction engine
        4. Format compacted turns as string
        5. Return ResolvedContent
        """
        raw_history = context.agent_config.get("session_history", [])
        turns = self._to_turns(raw_history)

        result = self._engine.compact(turns)

        raw_window = self._engine._config.raw_window
        compacted_zone = (
            result.turns[:-raw_window] if len(result.turns) > raw_window else []
        )

        content = self._format_turns(compacted_zone)
        tokens = (
            result.compacted_tokens
            - sum(len(t.content) // 4 for t in result.turns[-raw_window:])
            if len(result.turns) > raw_window
            else 0
        )

        return ResolvedContent(
            key=key,
            content=content,
            tokens=max(0, tokens),
            metadata={
                "turns_compacted": result.turns_compacted,
                "original_tokens": result.original_tokens,
                "compacted_tokens": result.compacted_tokens,
            },
        )

    def _to_turns(self, history: list[dict]) -> list[ConversationTurn]:
        """Convert raw session history dicts to ConversationTurn objects."""
        turns = []
        for i, entry in enumerate(history):
            turns.append(
                ConversationTurn(
                    turn_number=i,
                    role=entry.get("role", "assistant"),
                    content=entry.get("content", ""),
                    content_type=entry.get("content_type"),
                    metadata=entry.get("metadata"),
                    compacted=entry.get("compacted", False),
                )
            )
        return turns

    def _format_turns(self, turns: list[ConversationTurn]) -> str:
        """Format turns as a string for context injection."""
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"[Turn {t.turn_number}] {t.role}: {t.content}")
        return "\n".join(lines)
