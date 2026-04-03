"""Tests for hybrid retrieval engine."""

import time
from datetime import UTC, datetime, timedelta

import pytest

from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.mark.asyncio
    async def test_empty_store(self, store):
        """Empty store returns empty list."""
        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_only_strategy(self, store):
        """Keyword-only strategy returns keyword matches."""
        await store.save(Memory(key="user.employer", value="Anthropic"))
        await store.save(Memory(key="user.language", value="Python"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("anthropic")

        assert len(results) == 1
        assert results[0].memory.value == "Anthropic"

    @pytest.mark.asyncio
    async def test_hybrid_strategy_merges(self, store):
        """Hybrid strategy merges keyword and semantic results (deduplicated)."""
        await store.save(Memory(key="user.employer", value="Anthropic"))

        retriever = HybridRetriever(store=store, strategy="hybrid")
        # No embedding callable → only keyword results
        results = await retriever.retrieve("anthropic")

        assert len(results) == 1
        # No duplicates
        ids = [r.memory.id for r in results]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_recency_boost(self, store):
        """Recently accessed memory scores higher."""
        old = Memory(
            key="user.fact1", value="old fact",
            last_accessed=datetime.now(UTC) - timedelta(days=60),
        )
        recent = Memory(
            key="user.fact2", value="recent fact",
            last_accessed=datetime.now(UTC),
        )
        await store.save(old)
        await store.save(recent)

        retriever = HybridRetriever(store=store, strategy="keyword", recency_weight=0.5)
        results = await retriever.retrieve("fact")

        assert len(results) == 2
        # Recent should score higher due to recency boost
        assert results[0].memory.value == "recent fact"

    @pytest.mark.asyncio
    async def test_frequency_boost(self, store):
        """Frequently accessed memory scores higher."""
        low_freq = Memory(key="user.fact1", value="low freq fact", access_count=0)
        high_freq = Memory(key="user.fact2", value="high freq fact", access_count=100)
        await store.save(low_freq)
        await store.save(high_freq)

        retriever = HybridRetriever(store=store, strategy="keyword", frequency_weight=0.5)
        results = await retriever.retrieve("fact")

        assert len(results) == 2
        assert results[0].memory.value == "high freq fact"

    @pytest.mark.asyncio
    async def test_top_k_cap(self, store):
        """top_k caps results correctly."""
        for i in range(10):
            await store.save(Memory(key=f"k{i}", value=f"value {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("value", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_max_tokens_cap(self, store):
        """max_tokens caps results by estimated token count."""
        for i in range(10):
            await store.save(Memory(key=f"k{i}", value=f"value {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        # Very small token budget should limit results
        results = await retriever.retrieve("value", top_k=10, max_tokens=5)

        assert len(results) < 10

    @pytest.mark.asyncio
    async def test_results_sorted_descending(self, store):
        """Results sorted by relevance score descending."""
        await store.save(Memory(key="user.a", value="keyword match"))
        await store.save(Memory(key="user.b", value="another keyword match"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("keyword")

        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_stable_sort_deterministic_tiebreaker(self, store):
        """Equal scores produce deterministic order (sorted by memory ID)."""
        # Create memories with identical metadata so scores match after boosts
        now = datetime.now(UTC)
        mem_b = Memory(
            key="user.b", value="beta fact", id="mem_bbb",
            access_count=5, last_accessed=now,
        )
        mem_a = Memory(
            key="user.a", value="alpha fact", id="mem_aaa",
            access_count=5, last_accessed=now,
        )
        await store.save(mem_b)
        await store.save(mem_a)

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("fact")

        assert len(results) == 2
        # With deterministic tiebreaker, same-score items sort by ID ascending
        assert results[0].memory.id == "mem_aaa"
        assert results[1].memory.id == "mem_bbb"

    @pytest.mark.asyncio
    async def test_retrieve_does_not_touch(self, store):
        """retrieve() should NOT touch memories — touch is deferred."""
        mem = Memory(key="user.name", value="Alice", access_count=0)
        await store.save(mem)

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("alice")

        assert len(results) == 1
        # access_count unchanged after retrieve (touch is deferred)
        assert results[0].memory.access_count == 0

    @pytest.mark.asyncio
    async def test_flush_touches(self, store):
        """flush_touches() persists deferred touch updates."""
        mem = Memory(key="user.name", value="Alice", access_count=0)
        await store.save(mem)

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("alice")
        assert results[0].memory.access_count == 0

        await retriever.flush_touches()

        # Re-fetch from store to verify persistence
        updated = await store.get(mem.id)
        assert updated.access_count == 1
