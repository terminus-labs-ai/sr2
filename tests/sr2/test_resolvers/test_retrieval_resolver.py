"""Tests for retrieval resolver."""

import pytest

from sr2.memory.dimensions import DimensionalMatcher
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore
from sr2.resolvers.registry import ResolverContext
from sr2.resolvers.retrieval_resolver import RetrievalResolver


@pytest.fixture
def store():
    return InMemoryMemoryStore()


@pytest.fixture
def retriever(store):
    return HybridRetriever(store=store, strategy="keyword")


@pytest.fixture
def context_str():
    return ResolverContext(
        agent_config={},
        trigger_input="Alice",
    )


@pytest.fixture
def context_dict():
    return ResolverContext(
        agent_config={},
        trigger_input={"message": "Alice"},
    )


class TestRetrievalResolver:
    """Tests for RetrievalResolver."""

    @pytest.mark.asyncio
    async def test_resolve_string_trigger(self, store, retriever, context_str):
        """Resolve with string trigger_input uses it as query."""
        await store.save(Memory(key="user.name", value="Alice"))

        resolver = RetrievalResolver(retriever=retriever)
        result = await resolver.resolve(
            "memories", {"top_k": 10, "max_tokens": 4000}, context_str
        )

        assert result.key == "memories"
        assert "Alice" in result.content

    @pytest.mark.asyncio
    async def test_resolve_dict_trigger(self, store, retriever, context_dict):
        """Resolve with dict trigger_input extracts message field."""
        await store.save(Memory(key="user.name", value="Alice"))

        resolver = RetrievalResolver(retriever=retriever)
        result = await resolver.resolve("memories", {}, context_dict)

        assert "Alice" in result.content

    @pytest.mark.asyncio
    async def test_empty_results(self, store, retriever, context_str):
        """Empty retrieval results returns empty content string."""
        resolver = RetrievalResolver(retriever=retriever)
        result = await resolver.resolve("memories", {}, context_str)

        assert result.content == ""
        assert result.metadata == {"memory_count": 0}

    @pytest.mark.asyncio
    async def test_formatted_with_xml_tags(self, store, retriever, context_str):
        """Multiple results formatted with XML tags and key-value pairs."""
        await store.save(Memory(key="user.name", value="Alice"))
        await store.save(Memory(key="user.employer", value="Anthropic"))

        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="user")
        result = await resolver.resolve("memories", {}, context)

        assert "<retrieved_memories>" in result.content
        assert "</retrieved_memories>" in result.content
        assert "[user.name] Alice" in result.content

    @pytest.mark.asyncio
    async def test_max_tokens_passed_through(self, store, retriever, context_str):
        """max_tokens from config is passed through to retriever."""
        for i in range(20):
            await store.save(Memory(key=f"k{i}", value=f"alice info {i}"))

        resolver = RetrievalResolver(retriever=retriever)
        config = {"max_tokens": 5}  # Very small
        result = await resolver.resolve("memories", config, context_str)

        assert result.metadata["memory_count"] < 20

    @pytest.mark.asyncio
    async def test_dimensional_matcher_applied(self, store, retriever):
        """Dimensional matcher applied when provided."""
        await store.save(Memory(key="k1", value="slack info", dimensions={"channel": "slack"}))
        await store.save(Memory(key="k2", value="email info", dimensions={"channel": "email"}))

        matcher = DimensionalMatcher(matching_strategy="exact")
        resolver = RetrievalResolver(retriever=retriever, matcher=matcher)

        context = ResolverContext(
            agent_config={},
            trigger_input="info",
            interface_type="user_message",
        )
        result = await resolver.resolve("memories", {}, context)

        # interface_type "user_message" maps to channel "chat"
        # Neither "slack" nor "email" matches "chat" exactly
        assert result.metadata["memory_count"] == 0

    @pytest.mark.asyncio
    async def test_resolve_skipped_when_disabled(self, store, retriever, context_str):
        """Resolver returns empty content when enabled=False."""
        await store.save(Memory(key="user.name", value="Alice"))

        resolver = RetrievalResolver(retriever=retriever, enabled=False)
        result = await resolver.resolve("memories", {}, context_str)

        assert result.content == ""
        assert result.tokens == 0
        assert result.metadata["memory_count"] == 0
        assert result.metadata["skipped"] == "retrieval_disabled"

    @pytest.mark.asyncio
    async def test_resolve_runs_when_enabled(self, store, retriever, context_str):
        """Resolver retrieves memories when enabled=True (default)."""
        await store.save(Memory(key="user.name", value="Alice"))

        resolver = RetrievalResolver(retriever=retriever, enabled=True)
        result = await resolver.resolve("memories", {}, context_str)

        assert "Alice" in result.content
        assert result.metadata["memory_count"] > 0

    @pytest.mark.asyncio
    async def test_top_k_config_limits_results(self, store):
        """Config top_k limits number of returned memories."""
        retriever = HybridRetriever(store=store, strategy="keyword")
        for i in range(10):
            await store.save(Memory(key=f"fact.{i}", value=f"alice fact number {i}"))

        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="alice fact")

        result_limited = await resolver.resolve("memories", {"top_k": 3}, context)
        result_all = await resolver.resolve("memories", {"top_k": 20}, context)

        assert result_limited.metadata["memory_count"] <= 3
        assert result_all.metadata["memory_count"] > result_limited.metadata["memory_count"]

    @pytest.mark.asyncio
    async def test_empty_config_uses_defaults(self, store):
        """Empty config uses default top_k=10 and max_tokens=4000."""
        for i in range(15):
            await store.save(Memory(key=f"fact.{i}", value=f"alice info {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="alice info")

        result_default = await resolver.resolve("memories", {}, context)
        result_explicit = await resolver.resolve(
            "memories", {"top_k": 10, "max_tokens": 4000}, context
        )

        # Both should return same count since {} defaults to top_k=10, max_tokens=4000
        assert result_default.metadata["memory_count"] == result_explicit.metadata["memory_count"]

    @pytest.mark.asyncio
    async def test_top_k_one_returns_single_memory(self, store):
        """top_k=1 returns at most one memory."""
        for i in range(5):
            await store.save(Memory(key=f"fact.{i}", value=f"alice fact {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="alice fact")

        result = await resolver.resolve("memories", {"top_k": 1}, context)

        assert result.metadata["memory_count"] == 1


class TestRetrievalResolverHybridStrategy:
    """Tests for RetrievalResolver with hybrid strategy (keyword + semantic)."""

    @pytest.mark.asyncio
    async def test_hybrid_with_embedding_callable(self):
        """Hybrid strategy uses both keyword and semantic search when embedding callable is provided."""
        store = InMemoryMemoryStore()
        await store.save(Memory(key="user.name", value="Alice"))
        await store.save(Memory(key="user.hobby", value="painting"))

        async def mock_embed(query: str) -> list[float]:
            """Return a dummy embedding vector."""
            return [0.1, 0.2, 0.3]

        retriever = HybridRetriever(
            store=store,
            strategy="hybrid",
            embedding_callable=mock_embed,
        )
        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="Alice")
        result = await resolver.resolve("memories", {"top_k": 10}, context)

        # Hybrid merges keyword + semantic results. "Alice" matches keyword
        # on user.name, and semantic returns all (InMemoryMemoryStore returns
        # all memories with score 0.5 for vector search).
        assert result.metadata["memory_count"] >= 1
        assert result.content != ""

    @pytest.mark.asyncio
    async def test_hybrid_without_embedding_falls_back_to_keyword(self):
        """Hybrid strategy without embedding callable falls back to keyword-only.

        From the source: semantic branch runs only when self._embed is truthy.
        So hybrid without an embedding callable is effectively keyword-only.
        """
        store = InMemoryMemoryStore()
        await store.save(Memory(key="user.name", value="Alice"))

        retriever = HybridRetriever(
            store=store,
            strategy="hybrid",
            embedding_callable=None,  # No embedding function
        )
        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="Alice")
        result = await resolver.resolve("memories", {}, context)

        # Should still find Alice via keyword search
        assert "Alice" in result.content
        assert result.metadata["memory_count"] == 1

    @pytest.mark.asyncio
    async def test_semantic_only_strategy(self):
        """Semantic-only strategy skips keyword search entirely."""
        store = InMemoryMemoryStore()
        await store.save(Memory(key="user.name", value="Alice"))

        async def mock_embed(query: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        retriever = HybridRetriever(
            store=store,
            strategy="semantic",
            embedding_callable=mock_embed,
        )
        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="Alice")
        result = await resolver.resolve("memories", {"top_k": 10}, context)

        # semantic strategy + InMemoryMemoryStore returns all with score 0.5
        assert result.metadata["memory_count"] >= 1

    @pytest.mark.asyncio
    async def test_semantic_without_embedding_returns_empty(self):
        """Semantic strategy without embedding callable returns no results.

        Neither keyword nor semantic branch runs: keyword is excluded by
        strategy='semantic', and semantic is excluded by missing _embed.
        """
        store = InMemoryMemoryStore()
        await store.save(Memory(key="user.name", value="Alice"))

        retriever = HybridRetriever(
            store=store,
            strategy="semantic",
            embedding_callable=None,
        )
        resolver = RetrievalResolver(retriever=retriever)
        context = ResolverContext(agent_config={}, trigger_input="Alice")
        result = await resolver.resolve("memories", {}, context)

        assert result.content == ""
        assert result.metadata["memory_count"] == 0
