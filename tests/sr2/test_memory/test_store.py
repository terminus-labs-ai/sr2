"""Tests for memory store (InMemoryMemoryStore)."""

from datetime import UTC, datetime

import pytest

from sr2.memory.schema import Memory
from sr2.memory.store import InMemoryMemoryStore


@pytest.fixture
def store():
    return InMemoryMemoryStore()


class TestInMemoryMemoryStore:
    """Tests for InMemoryMemoryStore."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        """save() then get() returns the same memory."""
        mem = Memory(key="user.name", value="Alice")
        await store.save(mem)
        result = await store.get(mem.id)
        assert result is not None
        assert result.key == "user.name"
        assert result.value == "Alice"

    @pytest.mark.asyncio
    async def test_get_unknown_id(self, store):
        """get() with unknown ID returns None."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_key_sorted(self, store):
        """get_by_key() returns memories sorted by extracted_at desc."""
        old = Memory(
            key="user.name", value="Alice",
            extracted_at=datetime(2024, 1, 1, tzinfo=UTC),
        )
        new = Memory(
            key="user.name", value="Bob",
            extracted_at=datetime(2024, 6, 1, tzinfo=UTC),
        )
        await store.save(old)
        await store.save(new)
        results = await store.get_by_key("user.name")
        assert len(results) == 2
        assert results[0].value == "Bob"
        assert results[1].value == "Alice"

    @pytest.mark.asyncio
    async def test_get_by_key_excludes_archived(self, store):
        """get_by_key() excludes archived by default."""
        active = Memory(key="user.name", value="Alice")
        archived = Memory(key="user.name", value="Old", archived=True)
        await store.save(active)
        await store.save(archived)

        results = await store.get_by_key("user.name")
        assert len(results) == 1
        assert results[0].value == "Alice"

        results_all = await store.get_by_key("user.name", include_archived=True)
        assert len(results_all) == 2

    @pytest.mark.asyncio
    async def test_search_by_key_prefix(self, store):
        """search_by_key_prefix matches keys with given prefix."""
        await store.save(Memory(key="user.identity.name", value="Alice"))
        await store.save(Memory(key="user.identity.employer", value="Anthropic"))
        await store.save(Memory(key="user.preference.language", value="Python"))

        results = await store.search_by_key_prefix("user.identity")
        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"user.identity.name", "user.identity.employer"}

    @pytest.mark.asyncio
    async def test_archive(self, store):
        """archive() sets archived=True."""
        mem = Memory(key="k", value="v")
        await store.save(mem)
        result = await store.archive(mem.id)
        assert result is True
        updated = await store.get(mem.id)
        assert updated.archived is True

    @pytest.mark.asyncio
    async def test_archive_not_found(self, store):
        """archive() returns False if not found."""
        result = await store.archive("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """delete() removes the memory."""
        mem = Memory(key="k", value="v")
        await store.save(mem)
        result = await store.delete(mem.id)
        assert result is True
        assert await store.get(mem.id) is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, store):
        """delete() returns False if not found."""
        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_search_keyword(self, store):
        """search_keyword finds memory with matching value."""
        await store.save(Memory(key="user.employer", value="Anthropic"))
        await store.save(Memory(key="user.language", value="Python"))

        results = await store.search_keyword("anthropic")
        assert len(results) == 1
        assert results[0].memory.value == "Anthropic"
        assert results[0].match_type == "keyword"

    @pytest.mark.asyncio
    async def test_count(self, store):
        """count() excludes archived by default, includes with flag."""
        await store.save(Memory(key="k1", value="v1"))
        await store.save(Memory(key="k2", value="v2"))
        archived = Memory(key="k3", value="v3", archived=True)
        await store.save(archived)

        assert await store.count() == 2
        assert await store.count(include_archived=True) == 3

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, store):
        """save() with existing ID updates the memory."""
        mem = Memory(key="k", value="original")
        await store.save(mem)

        mem.value = "updated"
        await store.save(mem)

        result = await store.get(mem.id)
        assert result.value == "updated"
        assert await store.count() == 1


class TestListTopKeys:
    """Tests for list_top_keys()."""

    @pytest.mark.asyncio
    async def test_returns_keys_sorted_by_access_count(self, store):
        """Keys returned in descending access_count order."""
        await store.save(Memory(
            key="user.name", value="Alice", scope_ref="agent:test", access_count=5,
        ))
        await store.save(Memory(
            key="user.timezone", value="MST", scope_ref="agent:test", access_count=20,
        ))
        await store.save(Memory(
            key="user.employer", value="Acme", scope_ref="agent:test", access_count=10,
        ))

        keys = await store.list_top_keys("agent:test")
        assert keys == ["user.timezone", "user.employer", "user.name"]

    @pytest.mark.asyncio
    async def test_filters_by_scope_ref(self, store):
        """Only keys from the specified scope_ref are returned."""
        await store.save(Memory(
            key="user.name", value="Alice", scope_ref="agent:miranda", access_count=5,
        ))
        await store.save(Memory(
            key="user.name", value="Bob", scope_ref="agent:other", access_count=10,
        ))

        keys = await store.list_top_keys("agent:miranda")
        assert keys == ["user.name"]

    @pytest.mark.asyncio
    async def test_excludes_archived(self, store):
        """Archived memories are not included."""
        await store.save(Memory(
            key="user.old", value="Old", scope_ref="agent:test",
            access_count=100, archived=True,
        ))
        await store.save(Memory(
            key="user.current", value="Current", scope_ref="agent:test",
            access_count=1,
        ))

        keys = await store.list_top_keys("agent:test")
        assert keys == ["user.current"]

    @pytest.mark.asyncio
    async def test_respects_limit(self, store):
        """Returns at most `limit` keys."""
        for i in range(10):
            await store.save(Memory(
                key=f"key.{i}", value=f"v{i}", scope_ref="agent:test",
                access_count=10 - i,
            ))

        keys = await store.list_top_keys("agent:test", limit=3)
        assert len(keys) == 3
        assert keys == ["key.0", "key.1", "key.2"]

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty(self, store):
        """Empty store returns empty list."""
        keys = await store.list_top_keys("agent:test")
        assert keys == []

    @pytest.mark.asyncio
    async def test_deduplicates_keys(self, store):
        """Multiple memories with same key return single entry with max access."""
        await store.save(Memory(
            key="user.name", value="Alice", scope_ref="agent:test", access_count=5,
        ))
        await store.save(Memory(
            key="user.name", value="Bob", scope_ref="agent:test", access_count=2,
        ))

        keys = await store.list_top_keys("agent:test")
        assert keys.count("user.name") == 1
