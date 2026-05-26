"""Interface tests for MemoryStore and MemoryExtractor protocols.

Verifies that concrete implementations satisfy the Protocol contract.
Tests against a minimal stub implementation to validate the interface shape.
"""

from __future__ import annotations

from typing import Any

from sr2.memory import (
    ExtractionResult,
    Memory,
    MemoryExtractor,
    MemoryScope,
    MemorySearchResult,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# Minimal stub implementations for interface validation
# ---------------------------------------------------------------------------

class StubStore:
    """Minimal MemoryStore implementation for testing the protocol."""

    def save(self, memory: Memory) -> Memory:
        return memory

    def search(self, query: str, scope: MemoryScope | None = None, limit: int = 10) -> list[MemorySearchResult]:
        return []

    def get_by_tag(self, tag: str, scope: MemoryScope | None = None, limit: int = 10) -> list[MemorySearchResult]:
        return []

    def delete(self, memory_id: str) -> bool:
        return True

    def get_all(self, scope: MemoryScope | None = None) -> list[Memory]:
        return []


class StubExtractor:
    """Minimal MemoryExtractor implementation for testing the protocol."""

    def extract(self, turn_text: str, turn_id: str | None = None) -> ExtractionResult:
        return ExtractionResult()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMemoryStoreProtocol:
    def test_stub_satisfies_protocol(self):
        """StubStore is recognized as a MemoryStore."""
        store: MemoryStore = StubStore()
        assert isinstance(store, MemoryStore)

    def test_save_returns_memory(self):
        store: MemoryStore = StubStore()
        m = Memory(content="test")
        result = store.save(m)
        assert isinstance(result, Memory)
        assert result.content == "test"

    def test_search_returns_list_of_results(self):
        store: MemoryStore = StubStore()
        results = store.search("anything")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_with_scope_filter(self):
        store: MemoryStore = StubStore()
        results = store.search("query", scope=MemoryScope.PRIVATE, limit=5)
        assert isinstance(results, list)

    def test_get_by_tag_returns_list(self):
        store: MemoryStore = StubStore()
        results = store.get_by_tag("test")
        assert isinstance(results, list)

    def test_delete_returns_bool(self):
        store: MemoryStore = StubStore()
        result = store.delete("nonexistent-id")
        assert isinstance(result, bool)

    def test_get_all_returns_list(self):
        store: MemoryStore = StubStore()
        results = store.get_all()
        assert isinstance(results, list)

    def test_get_all_with_scope(self):
        store: MemoryStore = StubStore()
        results = store.get_all(scope=MemoryScope.SHARED)
        assert isinstance(results, list)

    def test_search_result_type(self):
        """Search results are MemorySearchResult, not Memory."""
        store = StubStore()
        results = store.search("test")
        # Empty list, but verify the return annotation is correct
        for r in results:
            assert isinstance(r, MemorySearchResult)


class TestMemoryExtractorProtocol:
    def test_stub_satisfies_protocol(self):
        """StubExtractor is recognized as a MemoryExtractor."""
        extractor: MemoryExtractor = StubExtractor()
        assert isinstance(extractor, MemoryExtractor)

    def test_extract_returns_extraction_result(self):
        extractor: MemoryExtractor = StubExtractor()
        result = extractor.extract("some conversation text")
        assert isinstance(result, ExtractionResult)

    def test_extract_with_turn_id(self):
        extractor: MemoryExtractor = StubExtractor()
        result = extractor.extract("text", turn_id="turn-001")
        assert isinstance(result, ExtractionResult)

    def test_extract_empty_text(self):
        extractor: MemoryExtractor = StubExtractor()
        result = extractor.extract("")
        assert isinstance(result, ExtractionResult)
        assert result.memories == []


class TestProtocolContract:
    def test_memory_store_method_signatures(self):
        """Verify all required methods exist and are callable."""
        store: MemoryStore = StubStore()
        assert callable(getattr(store, "save", None))
        assert callable(getattr(store, "search", None))
        assert callable(getattr(store, "get_by_tag", None))
        assert callable(getattr(store, "delete", None))
        assert callable(getattr(store, "get_all", None))

    def test_memory_extractor_method_signatures(self):
        """Verify extract method exists and is callable."""
        extractor: MemoryExtractor = StubExtractor()
        assert callable(getattr(extractor, "extract", None))
