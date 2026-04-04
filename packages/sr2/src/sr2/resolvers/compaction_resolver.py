"""Content resolver that returns compacted conversation history."""

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.resolvers.registry import ResolvedContent, ResolverContext, estimate_tokens


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

        raw_window = self._engine.raw_window
        compacted_zone = result.turns[:-raw_window] if len(result.turns) > raw_window else []

        content = self._format_turns(compacted_zone)
        tokens = (
            result.compacted_tokens
            - sum(estimate_tokens(t.content) for t in result.turns[-raw_window:])
            if len(result.turns) > raw_window
            else 0
        )

        # If LLM compaction produced analysis, prepend it before the compacted turns
        if result.analysis:
            analysis_block = self._format_analysis(result.analysis)
            if analysis_block and content:
                content = f"{analysis_block}\n\n{content}"
            elif analysis_block:
                content = analysis_block
            tokens = estimate_tokens(content)

        return ResolvedContent(
            key=key,
            content=content,
            tokens=max(0, tokens),
            metadata={
                "turns_compacted": result.turns_compacted,
                "original_tokens": result.original_tokens,
                "compacted_tokens": result.compacted_tokens,
                "has_analysis": result.analysis is not None,
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

    @staticmethod
    def _format_analysis(analysis: dict) -> str:
        """Format structured analysis from LLM compaction as context block."""
        parts = ["<compaction_analysis>"]
        if analysis.get("decisions"):
            parts.append("Decisions:")
            for d in analysis["decisions"]:
                parts.append(f"  - {d}")
        if analysis.get("current_state"):
            parts.append(f"Current state: {analysis['current_state']}")
        if analysis.get("open_questions"):
            parts.append("Open questions:")
            for q in analysis["open_questions"]:
                parts.append(f"  - {q}")
        if analysis.get("key_context"):
            parts.append("Key context:")
            for c in analysis["key_context"]:
                parts.append(f"  - {c}")
        parts.append("</compaction_analysis>")
        return "\n".join(parts) if len(parts) > 2 else ""
