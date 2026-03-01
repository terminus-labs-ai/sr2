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
        result = await resolver.resolve("memories", {}, context_str)

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
