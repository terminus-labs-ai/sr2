"""Tests for memory schema models."""

import time

import pytest

from sr2.memory.schema import (
    CONFIDENCE_SCORES,
    STABILITY_DEFAULTS,
    ExtractionResult,
    Memory,
    MemorySearchResult,
)


class TestMemory:
    """Tests for the Memory model."""

    def test_minimal_args(self):
        """Memory with just key and value produces valid defaults."""
        mem = Memory(key="user.identity.name", value="Alice")
        assert mem.key == "user.identity.name"
        assert mem.value == "Alice"
        assert mem.memory_type == "semi_stable"
        assert mem.stability_score == 0.7
        assert mem.confidence == 0.7
        assert mem.confidence_source == "contextual_mention"
        assert mem.dimensions == {}
        assert mem.archived is False
        assert mem.access_count == 0

    def test_id_auto_generated_unique(self):
        """Memory.id is auto-generated and unique across instances."""
        mem1 = Memory(key="k1", value="v1")
        mem2 = Memory(key="k2", value="v2")
        assert mem1.id.startswith("mem_")
        assert mem2.id.startswith("mem_")
        assert mem1.id != mem2.id

    def test_memory_type_stability_defaults(self):
        """Each memory_type has a corresponding stability default."""
        for mem_type, expected_stability in STABILITY_DEFAULTS.items():
            if mem_type == "ephemeral":
                continue
            mem = Memory(
                key="test",
                value="test",
                memory_type=mem_type,
                stability_score=expected_stability,
            )
            assert mem.stability_score == expected_stability

    def test_touch_updates_access(self):
        """touch() updates last_accessed and increments access_count."""
        mem = Memory(key="k", value="v")
        original_accessed = mem.last_accessed
        original_count = mem.access_count

        time.sleep(0.01)
        mem.touch()

        assert mem.access_count == original_count + 1
        assert mem.last_accessed >= original_accessed

    def test_dimensions_serialization(self):
        """Memory with dimensions dict serializes/deserializes correctly."""
        dims = {"channel": "slack", "project": "sr2"}
        mem = Memory(key="k", value="v", dimensions=dims)
        data = mem.model_dump()
        restored = Memory.model_validate(data)
        assert restored.dimensions == dims

    def test_full_fields(self):
        """Memory with all fields set works correctly."""
        mem = Memory(
            key="user.identity.employer",
            value="Anthropic",
            memory_type="identity",
            stability_score=1.0,
            confidence=0.9,
            confidence_source="direct_answer",
            dimensions={"channel": "chat"},
            scope="project",
            scope_ref="sr2",
            source="session:conv_123",
            archived=False,
            raw_text="I work at Anthropic",
        )
        assert mem.memory_type == "identity"
        assert mem.confidence_source == "direct_answer"
        assert mem.scope == "project"
        assert mem.scope_ref == "sr2"
        assert mem.source == "session:conv_123"


class TestMemorySearchResult:
    """Tests for the MemorySearchResult model."""

    def test_relevance_score_bounds(self):
        """MemorySearchResult requires relevance_score between 0 and 1."""
        mem = Memory(key="k", value="v")
        result = MemorySearchResult(memory=mem, relevance_score=0.85)
        assert result.relevance_score == 0.85

        with pytest.raises(Exception):
            MemorySearchResult(memory=mem, relevance_score=1.5)

        with pytest.raises(Exception):
            MemorySearchResult(memory=mem, relevance_score=-0.1)

    def test_default_match_type(self):
        """Default match_type is 'semantic'."""
        mem = Memory(key="k", value="v")
        result = MemorySearchResult(memory=mem, relevance_score=0.5)
        assert result.match_type == "semantic"


class TestExtractionResult:
    """Tests for the ExtractionResult model."""

    def test_extraction_result(self):
        """ExtractionResult holds a list of memories."""
        mem = Memory(key="k", value="v")
        result = ExtractionResult(memories=[mem], source="session:conv_1")
        assert len(result.memories) == 1
        assert result.source == "session:conv_1"


class TestConstants:
    """Tests for STABILITY_DEFAULTS and CONFIDENCE_SCORES."""

    def test_stability_defaults(self):
        """STABILITY_DEFAULTS has correct values."""
        assert STABILITY_DEFAULTS == {
            "identity": 1.0,
            "semi_stable": 0.7,
            "dynamic": 0.3,
            "ephemeral": 0.0,
        }

    def test_confidence_scores(self):
        """CONFIDENCE_SCORES has correct values."""
        assert CONFIDENCE_SCORES == {
            "explicit_statement": 1.0,
            "direct_answer": 0.9,
            "contextual_mention": 0.7,
            "inferred": 0.5,
            "offhand": 0.3,
        }
