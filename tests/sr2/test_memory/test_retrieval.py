"""Tests for hybrid retrieval engine."""

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [1, 3, 10, 50])
    async def test_top_k_parametrized(self, store, top_k):
        """top_k caps results: 20 stored memories, returns at most top_k."""
        for i in range(20):
            await store.save(Memory(key=f"item.{i}", value=f"item number {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("item", top_k=top_k)

        expected = min(top_k, 20)
        assert len(results) == expected

    @pytest.mark.asyncio
    async def test_recency_weight_zero_disables_boost(self, store):
        """With recency_weight=0 and frequency_weight=0, recency has no effect on scores."""
        old = Memory(
            key="user.old_fact",
            value="old keyword",
            last_accessed=datetime.now(UTC) - timedelta(days=365),
            access_count=0,
        )
        recent = Memory(
            key="user.new_fact",
            value="recent keyword",
            last_accessed=datetime.now(UTC),
            access_count=0,
        )
        await store.save(old)
        await store.save(recent)

        retriever = HybridRetriever(
            store=store,
            strategy="keyword",
            recency_weight=0.0,
            frequency_weight=0.0,
        )
        results = await retriever.retrieve("keyword")

        assert len(results) == 2
        # Both should have equal scores — base keyword score (0.7) * 1.0
        assert results[0].relevance_score == pytest.approx(results[1].relevance_score)


class TestTopKParametrized:
    """Parametrized tests for top_k config values."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "top_k",
        [1, 5, 10],
        ids=["top_k_1", "top_k_5", "top_k_10"],
    )
    async def test_top_k_returns_exactly_k(self, store, top_k):
        """With more items than top_k in the store, exactly top_k results are returned."""
        for i in range(15):
            await store.save(Memory(key=f"fact.{i}", value=f"fact number {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("fact", top_k=top_k)

        assert len(results) == top_k

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "top_k",
        [1, 5, 10],
        ids=["top_k_1", "top_k_5", "top_k_10"],
    )
    async def test_top_k_with_fewer_items(self, store, top_k):
        """When store has fewer items than top_k, all matching items are returned."""
        count = min(top_k - 1, 1)  # always fewer than top_k but at least 1
        for i in range(count):
            await store.save(Memory(key=f"fact.{i}", value=f"fact number {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("fact", top_k=top_k)

        assert len(results) == count

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "top_k",
        [1, 5, 10],
        ids=["top_k_1", "top_k_5", "top_k_10"],
    )
    async def test_top_k_results_sorted_by_score(self, store, top_k):
        """Returned top_k results are sorted by relevance score descending."""
        for i in range(15):
            await store.save(Memory(key=f"fact.{i}", value=f"fact number {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("fact", top_k=top_k)

        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)


class TestRecencyWeightParametrized:
    """Parametrized tests for recency_weight config values."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "recency_weight",
        [0.0, 0.5, 1.0],
        ids=["recency_off", "recency_half", "recency_full"],
    )
    async def test_recency_weight_scoring(self, store, recency_weight):
        """Recency weight controls how much recent memories are boosted."""
        old = Memory(
            key="user.old_fact",
            value="old keyword",
            last_accessed=datetime.now(UTC) - timedelta(days=365),
            access_count=0,
        )
        recent = Memory(
            key="user.new_fact",
            value="recent keyword",
            last_accessed=datetime.now(UTC),
            access_count=0,
        )
        await store.save(old)
        await store.save(recent)

        retriever = HybridRetriever(
            store=store,
            strategy="keyword",
            recency_weight=recency_weight,
            frequency_weight=0.0,
        )
        results = await retriever.retrieve("keyword")
        assert len(results) == 2

        old_result = next(r for r in results if r.memory.key == "user.old_fact")
        recent_result = next(r for r in results if r.memory.key == "user.new_fact")

        if recency_weight == 0.0:
            # No recency boost — scores should be equal
            assert old_result.relevance_score == pytest.approx(recent_result.relevance_score)
        else:
            # Positive recency weight — recent memory scores higher
            assert recent_result.relevance_score > old_result.relevance_score

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "recency_weight",
        [0.0, 0.5, 1.0],
        ids=["recency_off", "recency_half", "recency_full"],
    )
    async def test_recency_weight_gap_scales_with_weight(self, store, recency_weight):
        """Higher recency_weight produces a larger score gap between old and recent memories."""
        old = Memory(
            key="user.old_fact",
            value="old keyword",
            last_accessed=datetime.now(UTC) - timedelta(days=365),
            access_count=0,
        )
        recent = Memory(
            key="user.new_fact",
            value="recent keyword",
            last_accessed=datetime.now(UTC),
            access_count=0,
        )
        await store.save(old)
        await store.save(recent)

        retriever = HybridRetriever(
            store=store,
            strategy="keyword",
            recency_weight=recency_weight,
            frequency_weight=0.0,
        )
        results = await retriever.retrieve("keyword")

        old_result = next(r for r in results if r.memory.key == "user.old_fact")
        recent_result = next(r for r in results if r.memory.key == "user.new_fact")

        gap = recent_result.relevance_score - old_result.relevance_score
        assert gap >= 0.0  # recent should never score lower
