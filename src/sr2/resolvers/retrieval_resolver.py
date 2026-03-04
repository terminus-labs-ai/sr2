"""Content resolver that retrieves memories from long-term memory."""

from sr2.memory.dimensions import DimensionalMatcher
from sr2.memory.retrieval import HybridRetriever
from sr2.resolvers.registry import ResolvedContent, ResolverContext, estimate_tokens


class RetrievalResolver:
    """Content resolver that retrieves memories from LTM."""

    def __init__(
        self,
        retriever: HybridRetriever,
        matcher: DimensionalMatcher | None = None,
    ):
        self._retriever = retriever
        self._matcher = matcher

    async def resolve(
        self,
        key: str,
        config: dict,
        context: ResolverContext,
    ) -> ResolvedContent:
        """Resolve by retrieving relevant memories.

        1. Extract query from context (trigger_input or a specific field)
        2. Call retriever.retrieve()
        3. Apply dimensional matching if matcher is set
        4. Format results as a string for context injection
        5. Return ResolvedContent
        """
        query = self._extract_query(context)

        top_k = config.get("top_k", 10)
        max_tokens = config.get("max_tokens", 4000)
        results = await self._retriever.retrieve(query, top_k=top_k, max_tokens=max_tokens)

        if self._matcher:
            current_dims = self._extract_dimensions(context)
            results = self._matcher.filter(results, current_dims)

        content = self._format_memories(results)
        tokens = estimate_tokens(content)

        return ResolvedContent(
            key=key,
            content=content,
            tokens=tokens,
            metadata={"memory_count": len(results)},
        )

    def _extract_query(self, context: ResolverContext) -> str:
        """Extract a search query from the resolver context."""
        if isinstance(context.trigger_input, str):
            return context.trigger_input
        if isinstance(context.trigger_input, dict):
            return context.trigger_input.get("message", str(context.trigger_input))
        return str(context.trigger_input)

    def _extract_dimensions(self, context: ResolverContext) -> dict[str, str]:
        """Extract current dimensions from context."""
        dims = {}
        if context.interface_type:
            channel_map = {
                "user_message": "chat",
                "webhook": "webhook",
                "a2a_inbound": "agent",
                "heartbeat": "system",
            }
            dims["channel"] = channel_map.get(context.interface_type, context.interface_type)
        return dims

    def _format_memories(self, results: list) -> str:
        """Format retrieved memories as a string for context injection."""
        if not results:
            return ""
        lines = ["<retrieved_memories>"]
        for r in results:
            lines.append(f"  [{r.memory.key}] {r.memory.value}")
        lines.append("</retrieved_memories>")
        return "\n".join(lines)
