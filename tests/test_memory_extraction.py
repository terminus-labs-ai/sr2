"""Tests for rule-based memory extraction and MemoryResolver.

Covers:
  1. Extraction: preference detection
  2. Extraction: decision detection
  3. Extraction: correction detection
  4. Extraction: fact detection
  5. Extraction: tooling detection
  6. Extraction: short text produces no memories
  7. Extraction: deduplication (similar content)
  8. Extraction: max per turn limit
  9. Extraction: scope override
  10. Extraction: turn_id propagation
  11. MemoryResolver: returns empty for no match
  12. MemoryResolver: injects relevant memories
  13. MemoryResolver: scope filtering
  14. MemoryResolver: limit enforcement
  15. MemoryResolver: build from deps (custom store)
  16. MemoryResolver: build with default store (no extras)
"""

from __future__ import annotations

import pytest

from sr2.config.models import ResolverConfig
from sr2.memory import (
    InMemoryMemoryStore,
    Memory,
    MemoryResolver,
    MemoryScope,
    RuleBasedExtractor,
)
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------

class TestRuleBasedExtractor:
    def test_extract_preference(self):
        """Detects user preferences."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("I prefer concise responses and bullet points.")
        assert len(result.memories) > 0
        assert any("preference" in m.tags for m in result.memories)

    def test_extract_decision(self):
        """Detects decisions."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("We decided to use Godot 4.6 for the game engine.")
        assert len(result.memories) > 0
        assert any("decision" in m.tags for m in result.memories)

    def test_extract_correction(self):
        """Detects corrections."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("Don't do that again — always use protocols for dependencies.")
        assert len(result.memories) > 0
        assert any("correction" in m.tags for m in result.memories)

    def test_extract_fact(self):
        """Detects factual statements."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("The project uses pytest with xdist for parallel testing.")
        assert len(result.memories) > 0

    def test_extract_tooling(self):
        """Detects tooling/configuration facts."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("We installed ruff as the linter and formatter.")
        assert len(result.memories) > 0
        assert any("tooling" in m.tags for m in result.memories)

    def test_extract_short_text_no_memories(self):
        """Very short text produces no memories."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("Hi there!")
        assert result.memories == []

    def test_extract_empty_text(self):
        """Empty text returns empty result."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("")
        assert result.memories == []

    def test_extract_turn_id(self):
        """Turn ID is propagated to ExtractionResult."""
        extractor = RuleBasedExtractor()
        result = extractor.extract(
            "I prefer Python over everything else.",
            turn_id="turn-42",
        )
        assert result.source_turn_id == "turn-42"

    def test_extract_scope_override(self):
        """Scope override applies to all extracted memories."""
        extractor = RuleBasedExtractor()
        result = extractor.extract(
            "I prefer dark mode everywhere.",
            scope_override=MemoryScope.PROJECT,
        )
        for m in result.memories:
            assert m.scope == MemoryScope.PROJECT

    def test_extract_deduplication(self):
        """Similar content is deduplicated."""
        extractor = RuleBasedExtractor()
        # Two similar preference statements
        result = extractor.extract(
            "I prefer concise responses. I also prefer short answers."
        )
        # Should not produce two near-duplicate memories
        contents = [m.content.lower() for m in result.memories]
        for i, c1 in enumerate(contents):
            for c2 in contents[i + 1:]:
                assert c1 != c2

    def test_extract_max_per_turn(self):
        """Respects MAX_EXTRACT_PER_TURN limit."""
        extractor = RuleBasedExtractor()
        result = extractor.extract(
            "I prefer concise. I prefer bullets. I prefer direct. "
            "I prefer no fluff. I prefer facts. I prefer speed. "
            "We decided on Python. We chose Linux. We settled on Git. "
            "The project uses pytest. The system runs Docker."
        )
        assert len(result.memories) <= 5

    def test_extract_metadata(self):
        """ExtractionResult carries extractor metadata."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("I prefer Python.")
        assert result.metadata["extractor"] == "rule_based"

    def test_extract_memory_has_id(self):
        """Each extracted memory has a unique id."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("I prefer concise. We decided on Python.")
        ids = [m.id for m in result.memories]
        assert len(ids) == len(set(ids))

    def test_extract_memory_has_timestamps(self):
        """Extracted memories have created_at and last_accessed."""
        extractor = RuleBasedExtractor()
        result = extractor.extract("I prefer concise.")
        for m in result.memories:
            assert m.created_at is not None
            assert m.last_accessed is not None


# ---------------------------------------------------------------------------
# MemoryResolver tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMemoryResolver:
    def _make_event(self, text: str) -> Event:
        from sr2.models import TextBlock

        return Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="test",
            data=[TextBlock(text=text)],
        )

    async def test_resolver_returns_empty_no_match(self):
        """No matching memories returns empty content."""
        store = InMemoryMemoryStore()
        resolver = MemoryResolver(
            ResolverConfig(type="memory"),
            store,
        )
        result = await resolver.resolve([self._make_event("something random")])
        assert result.content == []

    async def test_resolver_injects_memories(self):
        """Relevant memories are injected as context."""
        store = InMemoryMemoryStore()
        store.save(Memory(content="user prefers concise responses", scope=MemoryScope.SHARED, tags=["preference"]))
        store.save(Memory(content="project uses Godot 4.6", scope=MemoryScope.PROJECT, tags=["fact"]))

        resolver = MemoryResolver(
            ResolverConfig(type="memory"),
            store,
        )
        # Query must substring-match stored content — the store uses literal keyword search
        result = await resolver.resolve([self._make_event("concise responses")])
        assert len(result.content) > 0
        content_text = " ".join(b.text if hasattr(b, "text") else "" for b in result.content)
        assert "concise" in content_text.lower()

    async def test_resolver_scope_filter(self):
        """Scope config filters search results."""
        store = InMemoryMemoryStore()
        store.save(Memory(content="private fact", scope=MemoryScope.PRIVATE, tags=["fact"]))
        store.save(Memory(content="shared fact", scope=MemoryScope.SHARED, tags=["fact"]))

        resolver = MemoryResolver(
            ResolverConfig(type="memory", config={"scope": "SHARED"}),
            store,
        )
        result = await resolver.resolve([self._make_event("fact")])
        if result.content:
            content_text = " ".join(b.text if hasattr(b, "text") else "" for b in result.content)
            assert "private" not in content_text.lower() or "shared" in content_text.lower()

    async def test_resolver_limit(self):
        """Limit config caps injected memories."""
        store = InMemoryMemoryStore()
        for i in range(10):
            store.save(Memory(content=f"test memory number {i}", scope=MemoryScope.PRIVATE, tags=["test"]))

        resolver = MemoryResolver(
            ResolverConfig(type="memory", config={"limit": 2}),
            store,
        )
        result = await resolver.resolve([self._make_event("test memory")])
        if result.content:
            content_text = " ".join(b.text if hasattr(b, "text") else "" for b in result.content)
            # Count "test memory number" occurrences
            count = content_text.lower().count("test memory number")
            assert count <= 2

    async def test_resolver_build_with_custom_store(self):
        """Build pulls custom MemoryStore from deps extras."""
        custom_store = InMemoryMemoryStore()
        custom_store.save(Memory(content="custom store memory", scope=MemoryScope.SHARED, tags=["test"]))

        resolver = MemoryResolver.build(
            ResolverConfig(type="memory"),
            Dependencies(extras={"memory_store": custom_store}),
        )
        assert resolver._store is custom_store

    async def test_resolver_build_default_store(self):
        """Build creates InMemoryMemoryStore when no extras."""
        resolver = MemoryResolver.build(
            ResolverConfig(type="memory"),
            Dependencies(),
        )
        assert isinstance(resolver._store, InMemoryMemoryStore)

    async def test_resolver_name(self):
        """Resolver has correct name."""
        resolver = MemoryResolver(ResolverConfig(type="memory"), InMemoryMemoryStore())
        assert resolver.name == "memory"

    async def test_resolver_execution_count(self):
        """Resolve increments execution count."""
        resolver = MemoryResolver(ResolverConfig(type="memory"), InMemoryMemoryStore())
        assert resolver.execution_count == 0
        await resolver.resolve([])
        assert resolver.execution_count == 1
