"""Resolver that injects compaction-immune session notes into compiled context."""

from sr2.resolvers.registry import ResolvedContent, ResolverContext, estimate_tokens


class SessionNotesResolver:
    """Injects session notes from ConversationZones into the pipeline.

    Session notes are agent-writable, compaction-immune strings that persist
    across compaction and summarization cycles. They appear in the compiled
    context as a distinct <session_notes> block.

    The resolver reads notes from context.agent_config["session_notes"],
    which should be populated by the runtime before pipeline compilation.
    """

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        notes: list[str] = context.agent_config.get("session_notes", [])
        if not notes:
            return ResolvedContent(key=key, content="", tokens=0)

        max_tokens = config.get("max_tokens", 2000)
        formatted = self._format_notes(notes, max_tokens)
        return ResolvedContent(key=key, content=formatted, tokens=estimate_tokens(formatted))

    @staticmethod
    def _format_notes(notes: list[str], max_tokens: int) -> str:
        """Format notes as XML block, respecting token cap.

        Drops oldest notes first when the cap is exceeded.
        """
        # Build from newest to oldest, stop when we'd exceed cap
        kept: list[str] = []
        tokens_used = 0
        # Overhead for the wrapping tags
        wrapper_overhead = estimate_tokens("<session_notes>\n</session_notes>")

        for note in reversed(notes):
            entry = f"- {note}"
            entry_tokens = estimate_tokens(entry)
            if tokens_used + entry_tokens + wrapper_overhead > max_tokens:
                break
            kept.append(entry)
            tokens_used += entry_tokens

        if not kept:
            return ""

        kept.reverse()
        body = "\n".join(kept)
        return f"<session_notes>\n{body}\n</session_notes>"
