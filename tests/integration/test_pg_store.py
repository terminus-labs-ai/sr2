"""Integration tests for PostgresMemoryStore against real PostgreSQL + pgvector."""

import pytest

from sr2.memory.schema import Memory
from tests.integration.conftest import requires_postgres


@requires_postgres
class TestPostgresMemoryStore:
    """Integration tests for PostgresMemoryStore.

    Mirrors all InMemoryMemoryStore unit tests but against real PostgreSQL.
    """

    @pytest.mark.asyncio
    async def test_save_and_get(self, pg_store):
        mem = Memory(key="user.name", value="Alice", memory_type="identity")
        await pg_store.save(mem)
        result = await pg_store.get(mem.id)
        assert result is not None
        assert result.key == "user.name"
        assert result.value == "Alice"

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, pg_store):
        result = await pg_store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_key(self, pg_store):
        await pg_store.save(Memory(key="user.employer", value="Google"))
        await pg_store.save(Memory(key="user.employer", value="Anthropic"))
        results = await pg_store.get_by_key("user.employer")
        assert len(results) == 2
        # Most recent first
        assert results[0].value == "Anthropic"

    @pytest.mark.asyncio
    async def test_get_by_key_excludes_archived(self, pg_store):
        mem = Memory(key="user.employer", value="Old Corp")
        await pg_store.save(mem)
        await pg_store.archive(mem.id)
        results = await pg_store.get_by_key("user.employer")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_by_key_prefix(self, pg_store):
        await pg_store.save(Memory(key="user.identity.name", value="Alice"))
        await pg_store.save(Memory(key="user.identity.employer", value="Acme"))
        await pg_store.save(Memory(key="user.preference.language", value="Python"))
        results = await pg_store.search_by_key_prefix("user.identity")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_keyword_search(self, pg_store):
        await pg_store.save(Memory(key="user.employer", value="Anthropic"))
        await pg_store.save(Memory(key="user.hobby", value="Chess"))
        results = await pg_store.search_keyword("anthropic")
        assert len(results) == 1
        assert results[0].memory.value == "Anthropic"

    @pytest.mark.asyncio
    async def test_archive_and_count(self, pg_store):
        mem = Memory(key="test", value="data")
        await pg_store.save(mem)
        assert await pg_store.count() == 1
        await pg_store.archive(mem.id)
        assert await pg_store.count(include_archived=False) == 0
        assert await pg_store.count(include_archived=True) == 1

    @pytest.mark.asyncio
    async def test_delete(self, pg_store):
        mem = Memory(key="test", value="data")
        await pg_store.save(mem)
        assert await pg_store.delete(mem.id) is True
        assert await pg_store.get(mem.id) is None

    @pytest.mark.asyncio
    async def test_save_updates_existing(self, pg_store):
        mem = Memory(key="user.name", value="Alice")
        await pg_store.save(mem)
        mem.value = "Bob"
        await pg_store.save(mem)
        result = await pg_store.get(mem.id)
        assert result.value == "Bob"
        assert await pg_store.count() == 1

    @pytest.mark.asyncio
    async def test_vector_search(self, pg_store):
        """Test vector search with a dummy embedding.

        Note: This tests the SQL/pgvector mechanics, not semantic quality.
        """
        mem = Memory(key="test", value="data")
        await pg_store.save(mem)
        # pgvector requires embeddings — verify it doesn't crash
        results = await pg_store.search_vector([0.0] * 1536, top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_dimensions_stored(self, pg_store):
        mem = Memory(
            key="user.preference.tone",
            value="formal",
            dimensions={"channel": "email"},
        )
        await pg_store.save(mem)
        result = await pg_store.get(mem.id)
        assert result.dimensions == {"channel": "email"}
