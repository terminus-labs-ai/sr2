"""Tests for memory dataclasses — schema.py.

Covers:
  1. Memory creation with defaults
  2. Memory with explicit values
  3. MemoryScope enum members
  4. MemorySearchResult structure
  5. ExtractionResult defaults
  6. Memory serialization (model_dump / model_validate round-trip)
  7. Memory frequency increment pattern
"""

from __future__ import annotations

import json

from sr2.memory import (
    ExtractionResult,
    Memory,
    MemoryScope,
    MemorySearchResult,
)


class TestMemoryDefaults:
    def test_memory_generates_id(self):
        m = Memory(content="test fact")
        assert m.id is not None
        assert len(m.id) > 0

    def test_memory_defaults(self):
        m = Memory(content="test")
        assert m.scope == MemoryScope.PRIVATE
        assert m.tags == []
        assert m.frequency == 0
        assert m.last_accessed is None
        assert m.created_at is not None

    def test_memory_explicit_values(self):
        m = Memory(
            content="user prefers concise responses",
            scope=MemoryScope.SHARED,
            tags=["preference", "communication"],
            frequency=3,
        )
        assert m.scope == MemoryScope.SHARED
        assert len(m.tags) == 2
        assert m.frequency == 3


class TestMemoryScope:
    def test_scope_enum_members(self):
        assert MemoryScope.PRIVATE == "private"
        assert MemoryScope.PROJECT == "project"
        assert MemoryScope.SHARED == "shared"

    def test_scope_from_string(self):
        assert MemoryScope("private") == MemoryScope.PRIVATE


class TestMemorySearchResult:
    def test_search_result_fields(self):
        r = MemorySearchResult(
            id="abc123",
            content="test memory",
            score=0.95,
            scope=MemoryScope.PRIVATE,
            tags=["test"],
        )
        assert r.id == "abc123"
        assert r.score == 0.95
        assert len(r.tags) == 1


class TestExtractionResult:
    def test_extraction_result_defaults(self):
        e = ExtractionResult()
        assert e.memories == []
        assert e.source_turn_id is None
        assert e.metadata == {}

    def test_extraction_result_with_memories(self):
        m = Memory(content="extracted fact")
        e = ExtractionResult(
            memories=[m],
            source_turn_id="turn-001",
            metadata={"model": "test"},
        )
        assert len(e.memories) == 1
        assert e.source_turn_id == "turn-001"


class TestMemorySerialization:
    def test_memory_round_trip(self):
        """Memory survives model_dump -> model_validate."""
        original = Memory(
            content="round trip test",
            scope=MemoryScope.PROJECT,
            tags=["test", "serialization"],
            frequency=5,
        )

        dumped = original.model_dump()
        restored = Memory.model_validate(dumped)

        assert restored.content == original.content
        assert restored.scope == original.scope
        assert restored.tags == original.tags
        assert restored.frequency == original.frequency

    def test_memory_json_round_trip(self):
        """Memory survives JSON serialization."""
        original = Memory(
            content="json test",
            scope=MemoryScope.SHARED,
            tags=["test"],
        )

        json_str = original.model_dump_json()
        restored = Memory.model_validate_json(json_str)

        assert restored.content == original.content
        assert restored.scope == original.scope
        assert restored.tags == original.tags

    def test_search_result_serialization(self):
        r = MemorySearchResult(
            id="test-id",
            content="search result",
            score=0.75,
            scope=MemoryScope.PRIVATE,
            tags=["result"],
        )
        dumped = r.model_dump()
        assert "score" in dumped
        assert dumped["score"] == 0.75

    def test_extraction_result_serialization(self):
        e = ExtractionResult(
            memories=[Memory(content="fact1"), Memory(content="fact2")],
            source_turn_id="turn-42",
        )
        dumped = e.model_dump()
        assert len(dumped["memories"]) == 2
        assert dumped["source_turn_id"] == "turn-42"

    def test_memory_unique_ids(self):
        """Two Memory instances get different IDs."""
        m1 = Memory(content="first")
        m2 = Memory(content="second")
        assert m1.id != m2.id
