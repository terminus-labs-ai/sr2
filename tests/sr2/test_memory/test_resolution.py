"""Tests for conflict resolution pipeline."""

import pytest

from sr2.memory.conflicts import Conflict
from sr2.memory.resolution import ConflictResolver
from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


class TestConflictResolver:
    """Tests for ConflictResolver."""

    @pytest.mark.asyncio
    async def test_identity_conflict_archives_old(self, store):
        """Identity memory conflict → old archived, new kept."""
        existing = Memory(key="user.name", value="Alice", memory_type="identity")
        new = Memory(key="user.name", value="Bob", memory_type="identity")
        await store.save(existing)
        await store.save(new)

        resolver = ConflictResolver(store=store)
        conflict = Conflict(new_memory=new, existing_memory=existing, conflict_type="key_match", confidence=1.0)
        result = await resolver.resolve(conflict)

        assert result.action == "archive_old"
        assert result.winner.value == "Bob"
        assert result.loser.value == "Alice"

        old = await store.get(existing.id)
        assert old.archived is True

    @pytest.mark.asyncio
    async def test_dynamic_conflict_deletes_old(self, store):
        """Dynamic memory conflict → old deleted, new kept."""
        existing = Memory(key="user.mood", value="happy", memory_type="dynamic")
        new = Memory(key="user.mood", value="tired", memory_type="dynamic")
        await store.save(existing)
        await store.save(new)

        resolver = ConflictResolver(store=store)
        conflict = Conflict(new_memory=new, existing_memory=existing, conflict_type="key_match", confidence=1.0)
        result = await resolver.resolve(conflict)

        assert result.action == "keep_new"
        assert result.winner.value == "tired"

        old = await store.get(existing.id)
        assert old is None

    @pytest.mark.asyncio
    async def test_keep_both_strategy(self, store):
        """Custom strategy keep_both → both tagged with conflicts_with."""
        existing = Memory(key="user.name", value="Alice", memory_type="identity")
        new = Memory(key="user.name", value="Bob", memory_type="identity")
        await store.save(existing)
        await store.save(new)

        resolver = ConflictResolver(store=store, strategies={"identity": "keep_both"})
        conflict = Conflict(new_memory=new, existing_memory=existing, conflict_type="key_match", confidence=1.0)
        result = await resolver.resolve(conflict)

        assert result.action == "keep_both_tagged"
        assert result.winner.conflicts_with == existing.id

        updated_existing = await store.get(existing.id)
        assert updated_existing.conflicts_with == new.id

    @pytest.mark.asyncio
    async def test_unknown_type_uses_default(self, store):
        """Unknown memory type falls back to default strategy."""
        existing = Memory(key="k", value="old", memory_type="semi_stable")
        new = Memory(key="k", value="new", memory_type="semi_stable")
        await store.save(existing)

        # No strategy defined for "semi_stable" explicitly, but default has it
        resolver = ConflictResolver(store=store)
        conflict = Conflict(new_memory=new, existing_memory=existing, conflict_type="key_match", confidence=1.0)
        result = await resolver.resolve(conflict)

        assert result.action == "archive_old"

    @pytest.mark.asyncio
    async def test_resolve_all(self, store):
        """resolve_all processes multiple conflicts."""
        mem1 = Memory(key="k1", value="old1", memory_type="identity")
        mem2 = Memory(key="k2", value="old2", memory_type="dynamic")
        new1 = Memory(key="k1", value="new1", memory_type="identity")
        new2 = Memory(key="k2", value="new2", memory_type="dynamic")
        await store.save(mem1)
        await store.save(mem2)

        resolver = ConflictResolver(store=store)
        conflicts = [
            Conflict(new_memory=new1, existing_memory=mem1, conflict_type="key_match", confidence=1.0),
            Conflict(new_memory=new2, existing_memory=mem2, conflict_type="key_match", confidence=1.0),
        ]
        results = await resolver.resolve_all(conflicts)

        assert len(results) == 2
        assert results[0].action == "archive_old"
        assert results[1].action == "keep_new"

    @pytest.mark.asyncio
    async def test_archive_resolution_verifiable(self, store):
        """After archive resolution, store.get(old_id).archived is True."""
        existing = Memory(key="k", value="old", memory_type="identity")
        new = Memory(key="k", value="new", memory_type="identity")
        await store.save(existing)

        resolver = ConflictResolver(store=store)
        conflict = Conflict(new_memory=new, existing_memory=existing, conflict_type="key_match", confidence=1.0)
        await resolver.resolve(conflict)

        old = await store.get(existing.id)
        assert old is not None
        assert old.archived is True

    @pytest.mark.asyncio
    async def test_discard_resolution_verifiable(self, store):
        """After discard resolution, store.get(old_id) returns None."""
        existing = Memory(key="k", value="old", memory_type="dynamic")
        new = Memory(key="k", value="new", memory_type="dynamic")
        await store.save(existing)

        resolver = ConflictResolver(store=store)
        conflict = Conflict(new_memory=new, existing_memory=existing, conflict_type="key_match", confidence=1.0)
        await resolver.resolve(conflict)

        old = await store.get(existing.id)
        assert old is None
