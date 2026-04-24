"""Content resolver that returns previously generated summaries."""

import logging

from sr2.resolvers.registry import ResolvedContent, ResolverContext, estimate_tokens
from sr2.summarization.engine import StructuredSummary

logger = logging.getLogger(__name__)


class SummarizationResolver:
    """Content resolver that returns previously generated summaries."""

    def __init__(self) -> None:
        self._summaries: list[str] = []

    def add_summary(self, summary_text: str) -> None:
        """Add a completed summary to the store.

        Called by the post-LLM processor when summarization completes.
        """
        self._summaries.append(summary_text)

    async def resolve(
        self,
        key: str,
        config: dict,
        context: ResolverContext,
    ) -> ResolvedContent:
        """Return all stored summaries concatenated.

        Uses flat injection (all summaries concatenated in order).
        """
        if not self._summaries:
            return ResolvedContent(key=key, content="", tokens=0)

        summaries = list(self._summaries)
        content = "\n---\n".join(summaries)
        tokens = estimate_tokens(content)
        max_tokens = config.get("max_tokens")

        if max_tokens and tokens > max_tokens:
            original_count = len(summaries)
            while tokens > max_tokens and len(summaries) > 1:
                summaries.pop(0)
                content = "\n---\n".join(summaries)
                tokens = estimate_tokens(content)
            dropped = original_count - len(summaries)
            if dropped:
                logger.warning(
                    "Summary token budget exceeded, dropped %d/%d oldest summaries to fit (%d tokens)",
                    dropped,
                    original_count,
                    tokens,
                )

        return ResolvedContent(
            key=key,
            content=content,
            tokens=tokens,
            metadata={"summary_count": len(summaries)},
        )

    @staticmethod
    def format_structured_summary(summary: StructuredSummary) -> str:
        """Format a StructuredSummary as a string for storage."""
        parts = [f"[Summary of turns {summary.summary_of_turns}]"]
        if summary.key_decisions:
            parts.append("Decisions: " + "; ".join(summary.key_decisions))
        if summary.unresolved:
            parts.append("Unresolved: " + "; ".join(summary.unresolved))
        if summary.facts:
            parts.append("Facts: " + "; ".join(summary.facts))
        if summary.user_preferences:
            parts.append("Preferences: " + "; ".join(summary.user_preferences))
        if summary.errors_encountered:
            parts.append("Errors: " + "; ".join(summary.errors_encountered))
        return "\n".join(parts)
