"""Tests for SQLiteMemoryStore backend."""

import pytest
from datetime import datetime

from sr2.memory.schema import Memory
from sr2.memory.store import SQLiteMemoryStore


@pytest.fixture
async def sqlite_store() -> SQLiteMemoryStore:
    """Create an in-memory SQLite store for testing."""
    store = SQLiteMemoryStore(":memory:")
    await store.connect()
    yield store
    await store.disconnect()


@pytest.mark.asyncio
async def test_sqlite_save_and_get(sqlite_store: SQLiteMemoryStore) -> None:
    """Test saving and retrieving a memory."""
    memory = Memory(
        id="test1",
        key="user.preference",
        value="prefers concise responses",
        memory_type="semi_stable",
        stability_score=0.8,
        confidence=0.85,
        confidence_source="explicit_statement",
        dimensions={"topic": "communication_style"},
        scope="project",
        scope_ref="test-project",
        source="session:conv1",
    )
    await sqlite_store.save(memory)

    retrieved = await sqlite_store.get("test1")
    assert retrieved is not None
    assert retrieved.id == "test1"
    assert retrieved.key == "user.preference"
    assert retrieved.value == "prefers concise responses"
    assert retrieved.stability_score == 0.8


@pytest.mark.asyncio
async def test_sqlite_save_with_embedding(sqlite_store: SQLiteMemoryStore) -> None:
    """Test saving a memory with an embedding."""
    memory = Memory(
        id="test_embed",
        key="user.preference",
        value="prefers concise responses",
    )
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    await sqlite_store.save(memory, embedding=embedding)

    retrieved = await sqlite_store.get("test_embed")
    assert retrieved is not None
    assert retrieved.id == "test_embed"


@pytest.mark.asyncio
async def test_sqlite_get_nonexistent(sqlite_store: SQLiteMemoryStore) -> None:
    """Test retrieving a nonexistent memory returns None."""
    result = await sqlite_store.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_sqlite_get_by_key(sqlite_store: SQLiteMemoryStore) -> None:
    """Test retrieving memories by key."""
    memories = [
        Memory(id="m1", key="user.preference", value="concise"),
        Memory(id="m2", key="user.preference", value="detailed"),
        Memory(id="m3", key="user.timezone", value="UTC"),
    ]
    for m in memories:
        await sqlite_store.save(m)

    results = await sqlite_store.get_by_key("user.preference")
    assert len(results) == 2
    assert all(m.key == "user.preference" for m in results)


@pytest.mark.asyncio
async def test_sqlite_get_by_key_archived(sqlite_store: SQLiteMemoryStore) -> None:
    """Test that archived memories are excluded by default."""
    memory = Memory(id="m1", key="user.preference", value="value", archived=False)
    await sqlite_store.save(memory)

    await sqlite_store.archive("m1")

    results = await sqlite_store.get_by_key("user.preference", include_archived=False)
    assert len(results) == 0

    results_with_archived = await sqlite_store.get_by_key("user.preference", include_archived=True)
    assert len(results_with_archived) == 1


@pytest.mark.asyncio
async def test_sqlite_search_by_key_prefix(sqlite_store: SQLiteMemoryStore) -> None:
    """Test prefix search."""
    memories = [
        Memory(id="m1", key="user.preference.style", value="concise"),
        Memory(id="m2", key="user.preference.tone", value="formal"),
        Memory(id="m3", key="user.timezone", value="UTC"),
    ]
    for m in memories:
        await sqlite_store.save(m)

    results = await sqlite_store.search_by_key_prefix("user.preference")
    assert len(results) == 2
    assert all(m.key.startswith("user.preference") for m in results)


@pytest.mark.asyncio
async def test_sqlite_delete(sqlite_store: SQLiteMemoryStore) -> None:
    """Test deleting a memory."""
    memory = Memory(id="to_delete", key="test", value="value")
    await sqlite_store.save(memory)

    result = await sqlite_store.delete("to_delete")
    assert result is True

    retrieved = await sqlite_store.get("to_delete")
    assert retrieved is None


@pytest.mark.asyncio
async def test_sqlite_delete_nonexistent(sqlite_store: SQLiteMemoryStore) -> None:
    """Test deleting a nonexistent memory returns False."""
    result = await sqlite_store.delete("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_sqlite_archive(sqlite_store: SQLiteMemoryStore) -> None:
    """Test archiving a memory."""
    memory = Memory(id="to_archive", key="test", value="value", archived=False)
    await sqlite_store.save(memory)

    result = await sqlite_store.archive("to_archive")
    assert result is True

    retrieved = await sqlite_store.get("to_archive")
    assert retrieved is not None
    assert retrieved.archived is True


@pytest.mark.asyncio
async def test_sqlite_archive_nonexistent(sqlite_store: SQLiteMemoryStore) -> None:
    """Test archiving a nonexistent memory returns False."""
    result = await sqlite_store.archive("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_sqlite_search_keyword(sqlite_store: SQLiteMemoryStore) -> None:
    """Test keyword search."""
    memories = [
        Memory(id="m1", key="task", value="implement authentication system"),
        Memory(id="m2", key="task", value="fix authentication bug"),
        Memory(id="m3", key="task", value="refactor database layer"),
    ]
    for m in memories:
        await sqlite_store.save(m)

    results = await sqlite_store.search_keyword("authentication", top_k=10)
    assert len(results) == 2
    assert all("authentication" in m.memory.value.lower() for m in results)


@pytest.mark.asyncio
async def test_sqlite_count(sqlite_store: SQLiteMemoryStore) -> None:
    """Test counting memories."""
    memories = [
        Memory(id="m1", key="test", value="v1", archived=False),
        Memory(id="m2", key="test", value="v2", archived=False),
        Memory(id="m3", key="test", value="v3", archived=True),
    ]
    for m in memories:
        await sqlite_store.save(m)

    count = await sqlite_store.count(include_archived=False)
    assert count == 2

    count_with_archived = await sqlite_store.count(include_archived=True)
    assert count_with_archived == 3


@pytest.mark.asyncio
async def test_sqlite_update_existing(sqlite_store: SQLiteMemoryStore) -> None:
    """Test updating an existing memory."""
    memory1 = Memory(id="m1", key="user.preference", value="old value")
    await sqlite_store.save(memory1)

    memory2 = Memory(id="m1", key="user.preference", value="new value")
    await sqlite_store.save(memory2)

    retrieved = await sqlite_store.get("m1")
    assert retrieved is not None
    assert retrieved.value == "new value"


@pytest.mark.asyncio
async def test_sqlite_search_vector(sqlite_store: SQLiteMemoryStore) -> None:
    """Test vector search (limited in SQLite)."""
    memories = [
        Memory(id="m1", key="test", value="v1"),
        Memory(id="m2", key="test", value="v2"),
    ]
    for m in memories:
        await sqlite_store.save(m, embedding=[0.1, 0.2, 0.3])

    results = await sqlite_store.search_vector([0.1, 0.2, 0.3], top_k=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_sqlite_not_connected_error(sqlite_store: SQLiteMemoryStore) -> None:
    """Test that operations fail when not connected."""
    new_store = SQLiteMemoryStore(":memory:")
    # Don't call connect()

    with pytest.raises(RuntimeError, match="Not connected"):
        await new_store.get("test")
