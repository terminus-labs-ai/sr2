"""Tests for conflict detection."""

import pytest

from sr2.memory.conflicts import ConflictDetector
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


class TestConflictDetector:
    """Tests for ConflictDetector."""

    @pytest.mark.asyncio
    async def test_no_existing_memories(self, store):
        """No existing memories → no conflicts detected."""
        detector = ConflictDetector(store=store)
        new = Memory(key="user.name", value="Alice")
        conflicts = await detector.detect(new)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_same_key_different_value(self, store):
        """Same key, different value → conflict detected with type key_match."""
        existing = Memory(key="user.employer", value="Google")
        await store.save(existing)

        detector = ConflictDetector(store=store)
        new = Memory(key="user.employer", value="Anthropic")
        conflicts = await detector.detect(new)

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "key_match"
        assert conflicts[0].confidence == 1.0
        assert conflicts[0].existing_memory.value == "Google"
        assert conflicts[0].new_memory.value == "Anthropic"

    @pytest.mark.asyncio
    async def test_same_key_same_value_case_insensitive(self, store):
        """Same key, same value (case-insensitive) → no conflict."""
        existing = Memory(key="user.name", value="Alice")
        await store.save(existing)

        detector = ConflictDetector(store=store)
        new = Memory(key="user.name", value="alice")
        conflicts = await detector.detect(new)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_archived_memory_excluded(self, store):
        """Same key, archived memory → no conflict (archived excluded)."""
        existing = Memory(key="user.employer", value="Google", archived=True)
        await store.save(existing)

        detector = ConflictDetector(store=store)
        new = Memory(key="user.employer", value="Anthropic")
        conflicts = await detector.detect(new)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_different_key_no_conflict(self, store):
        """Different key, no semantic → no conflict."""
        existing = Memory(key="user.name", value="Alice")
        await store.save(existing)

        detector = ConflictDetector(store=store)
        new = Memory(key="user.employer", value="Anthropic")
        conflicts = await detector.detect(new)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_self_conflict_excluded(self, store):
        """Self-conflict excluded (same memory ID)."""
        mem = Memory(key="user.name", value="Alice")
        await store.save(mem)

        detector = ConflictDetector(store=store)
        # Detect against itself
        conflicts = await detector.detect(mem)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_multiple_conflicts(self, store):
        """Multiple existing memories with same key → multiple conflicts."""
        await store.save(Memory(key="user.employer", value="Google"))
        await store.save(Memory(key="user.employer", value="Meta"))

        detector = ConflictDetector(store=store)
        new = Memory(key="user.employer", value="Anthropic")
        conflicts = await detector.detect(new)

        assert len(conflicts) == 2

    @pytest.mark.asyncio
    async def test_new_key_no_conflict(self, store):
        """detect() returns empty list for new key."""
        await store.save(Memory(key="user.name", value="Alice"))

        detector = ConflictDetector(store=store)
        new = Memory(key="user.language", value="Python")
        conflicts = await detector.detect(new)

        assert len(conflicts) == 0
