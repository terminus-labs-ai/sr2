"""Tests for trace event emission in HybridRetriever."""

import pytest

from sr2.memory.retrieval import HybridRetriever
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


class MockTraceCollector:
    """Simple mock that records emit() calls."""

    def __init__(self):
        self.calls: list[tuple[str, dict, dict]] = []

    def emit(self, event_name: str, data: dict, **kwargs):
        self.calls.append((event_name, data, kwargs))


@pytest.fixture
def store():
    return InMemoryMemoryStore()


@pytest.fixture
def trace():
    return MockTraceCollector()


class TestRetrievalTrace:
    """Tests for trace_collector integration in HybridRetriever."""

    @pytest.mark.asyncio
    async def test_no_trace_collector_works_as_before(self, store):
        """When trace_collector=None (default), retrieve() works exactly as before."""
        await store.save(Memory(key="user.name", value="Alice"))

        retriever = HybridRetriever(store=store, strategy="keyword")
        results = await retriever.retrieve("alice")

        assert len(results) == 1
        assert results[0].memory.value == "Alice"

    @pytest.mark.asyncio
    async def test_trace_emit_called(self, store, trace):
        """When a trace collector is provided, emit('retrieve', ...) is called."""
        await store.save(Memory(key="user.name", value="Alice"))

        retriever = HybridRetriever(store=store, strategy="keyword", trace_collector=trace)
        await retriever.retrieve("alice")

        assert len(trace.calls) == 1
        event_name, data, kwargs = trace.calls[0]
        assert event_name == "retrieve"
        assert "duration_ms" in kwargs

    @pytest.mark.asyncio
    async def test_trace_data_fields(self, store, trace):
        """Emitted data contains correct top-level fields."""
        await store.save(Memory(key="user.name", value="Alice"))
        await store.save(Memory(key="user.lang", value="Python"))

        retriever = HybridRetriever(store=store, strategy="keyword", trace_collector=trace)
        await retriever.retrieve("alice", top_k=5)

        assert len(trace.calls) == 1
        _, data, _ = trace.calls[0]

        assert data["query"] == "alice"
        assert data["strategy"] == "keyword"
        assert data["top_k"] == 5
        assert data["threshold"] == 0.0
        assert isinstance(data["candidates_scored"], int)
        assert isinstance(data["results_returned"], int)
        assert isinstance(data["results"], list)
        assert isinstance(data["latency_ms"], float)

    @pytest.mark.asyncio
    async def test_trace_result_entry_fields(self, store, trace):
        """Each result in the trace has required fields."""
        await store.save(
            Memory(key="user.name", value="Alice", memory_type="identity", scope="private")
        )

        retriever = HybridRetriever(store=store, strategy="keyword", trace_collector=trace)
        await retriever.retrieve("alice")

        _, data, _ = trace.calls[0]
        assert len(data["results"]) >= 1

        entry = data["results"][0]
        assert entry["key"] == "user.name"
        assert entry["value_preview"] == "Alice"
        assert isinstance(entry["relevance_score"], float)
        assert entry["memory_type"] == "identity"
        assert entry["scope"] == "private"
        assert "selected" in entry
        assert "match_type" in entry

    @pytest.mark.asyncio
    async def test_selected_flag(self, store, trace):
        """The 'selected' flag is True for top_k results and False for others."""
        # Store uses top_k*2 internally for keyword search, so use top_k=3
        # and store enough items to exceed that internal fetch (3*2=6)
        for i in range(10):
            await store.save(Memory(key=f"item.{i}", value=f"item number {i}"))

        retriever = HybridRetriever(store=store, strategy="keyword", trace_collector=trace)
        await retriever.retrieve("item", top_k=3)

        _, data, _ = trace.calls[0]
        # Keyword search fetches top_k*2 = 6 candidates internally
        candidates_scored = data["candidates_scored"]
        assert data["results_returned"] == 3
        assert candidates_scored > 3  # more candidates than returned

        selected = [r for r in data["results"] if r["selected"]]
        not_selected = [r for r in data["results"] if not r["selected"]]
        assert len(selected) == 3
        assert len(not_selected) == candidates_scored - 3

    @pytest.mark.asyncio
    async def test_value_preview_truncated(self, store, trace):
        """value_preview is truncated to 100 characters."""
        long_value = "x" * 200
        await store.save(Memory(key="user.long", value=long_value))

        retriever = HybridRetriever(store=store, strategy="keyword", trace_collector=trace)
        await retriever.retrieve("x")

        _, data, _ = trace.calls[0]
        assert len(data["results"]) == 1
        assert len(data["results"][0]["value_preview"]) == 100

    @pytest.mark.asyncio
    async def test_empty_results_still_emits(self, store, trace):
        """Trace is emitted even when no results are found."""
        retriever = HybridRetriever(store=store, strategy="keyword", trace_collector=trace)
        results = await retriever.retrieve("nothing")

        assert results == []
        assert len(trace.calls) == 1
        _, data, _ = trace.calls[0]
        assert data["candidates_scored"] == 0
        assert data["results_returned"] == 0
        assert data["results"] == []
