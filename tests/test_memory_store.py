"""Tests for InMemoryMemoryStore — full CRUD, search, scope filtering.

Covers:
  1. save: persist and return Memory
  2. save: frequency increment on duplicate id
  3. search: keyword match on content
  4. search: keyword match on tags
  5. search: scope filtering
  6. search: ranking by frequency
  7. search: empty query returns empty
  8. search: limit enforcement
  9. get_by_tag: filter by tag
  10. get_by_tag: scope filtering
  11. get_by_tag: limit enforcement
  12. delete: remove by id
  13. delete: returns False for missing id
  14. get_all: all memories
  15. get_all: scope filtering
  16. Empty state: search/delete/get_all on empty store
  17. Protocol satisfaction
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sr2.memory import (
    InMemoryMemoryStore,
    Memory,
    MemoryScope,
    MemoryStore,
)


def make_memory(content: str, scope: MemoryScope = MemoryScope.PRIVATE, tags: list[str] | None = None, frequency: int = 0, last_accessed: datetime | None = None) -> Memory:
    return Memory(
        content=content,
        scope=scope,
        tags=tags or [],
        frequency=frequency,
        last_accessed=last_accessed or datetime.now(timezone.utc),
    )


class TestSave:
    def test_save_persists_memory(self):
        store = InMemoryMemoryStore()
        m = make_memory("test fact")
        result = store.save(m)
        assert result.content == "test fact"
        assert len(store.get_all()) == 1

    def test_save_returns_memory(self):
        store = InMemoryMemoryStore()
        m = make_memory("test")
        result = store.save(m)
        assert isinstance(result, Memory)
        assert result.id == m.id

    def test_save_sets_last_accessed(self):
        store = InMemoryMemoryStore()
        m = make_memory("test", last_accessed=None)
        result = store.save(m)
        assert result.last_accessed is not None

    def test_save_increments_frequency_on_duplicate(self):
        store = InMemoryMemoryStore()
        m = make_memory("repeated fact", frequency=0)
        store.save(m)
        result = store.save(m)
        assert result.frequency == 1  # 0 + 1

    def test_save_multiple_increment_frequency(self):
        store = InMemoryMemoryStore()
        m = make_memory("very repeated", frequency=0)
        for _ in range(5):
            store.save(m)
        result = store.save(m)
        assert result.frequency == 5

    def test_save_does_not_mutate_original(self):
        store = InMemoryMemoryStore()
        m = make_memory("original", frequency=0)
        original_last_accessed = m.last_accessed
        store.save(m)
        store.save(m)  # second save would increment frequency if mutating
        assert m.frequency == 0  # original object unchanged
        assert m.last_accessed == original_last_accessed  # timestamp unchanged

    def test_save_different_memories_independent(self):
        store = InMemoryMemoryStore()
        m1 = make_memory("fact one")
        m2 = make_memory("fact two")
        store.save(m1)
        store.save(m2)
        assert len(store.get_all()) == 2


class TestSearch:
    def test_search_keyword_in_content(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("user likes Python"))
        store.save(make_memory("user dislikes Rust"))
        results = store.search("Python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_keyword_in_tags(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("some fact", tags=["preference", "tooling"]))
        results = store.search("preference")
        assert len(results) == 1

    def test_search_case_insensitive(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("User likes Coffee"))
        results = store.search("coffee")
        assert len(results) == 1

    def test_search_scope_filter(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("private fact", scope=MemoryScope.PRIVATE))
        store.save(make_memory("shared fact", scope=MemoryScope.SHARED))
        results = store.search("fact", scope=MemoryScope.PRIVATE)
        assert len(results) == 1
        assert results[0].scope == MemoryScope.PRIVATE

    def test_search_no_scope_filter(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("private fact", scope=MemoryScope.PRIVATE))
        store.save(make_memory("shared fact", scope=MemoryScope.SHARED))
        results = store.search("fact")
        assert len(results) == 2

    def test_search_ranks_by_frequency(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("high freq fact", frequency=10))
        store.save(make_memory("low freq fact", frequency=1))
        results = store.search("fact")
        assert results[0].content == "high freq fact"
        assert results[1].content == "low freq fact"

    def test_search_empty_query(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("something"))
        results = store.search("")
        assert results == []

    def test_search_no_match(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("about cats"))
        results = store.search("dogs")
        assert results == []

    def test_search_limit(self):
        store = InMemoryMemoryStore()
        for i in range(20):
            store.save(make_memory(f"test fact number {i}"))
        results = store.search("fact", limit=5)
        assert len(results) == 5

    def test_search_returns_memory_search_result(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("searchable"))
        results = store.search("searchable")
        assert len(results) == 1
        assert hasattr(results[0], "score")
        assert results[0].score > 0


class TestGetByTag:
    def test_get_by_tag(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("fact one", tags=["important"]))
        store.save(make_memory("fact two", tags=["minor"]))
        results = store.get_by_tag("important")
        assert len(results) == 1
        assert "fact one" in results[0].content

    def test_get_by_tag_scope_filter(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("private tagged", scope=MemoryScope.PRIVATE, tags=["todo"]))
        store.save(make_memory("shared tagged", scope=MemoryScope.SHARED, tags=["todo"]))
        results = store.get_by_tag("todo", scope=MemoryScope.SHARED)
        assert len(results) == 1
        assert results[0].scope == MemoryScope.SHARED

    def test_get_by_tag_limit(self):
        store = InMemoryMemoryStore()
        for i in range(15):
            store.save(make_memory(f"tagged {i}", tags=["batch"]))
        results = store.get_by_tag("batch", limit=3)
        assert len(results) == 3

    def test_get_by_tag_case_insensitive(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("fact", tags=["Important"]))
        results = store.get_by_tag("important")
        assert len(results) == 1

    def test_get_by_tag_no_match(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("fact", tags=["alpha"]))
        results = store.get_by_tag("beta")
        assert results == []


class TestDelete:
    def test_delete_existing(self):
        store = InMemoryMemoryStore()
        m = make_memory("to delete")
        store.save(m)
        result = store.delete(m.id)
        assert result is True
        assert len(store.get_all()) == 0

    def test_delete_nonexistent(self):
        store = InMemoryMemoryStore()
        result = store.delete("does-not-exist")
        assert result is False

    def test_delete_other_memories_remain(self):
        store = InMemoryMemoryStore()
        m1 = make_memory("keep")
        m2 = make_memory("remove")
        store.save(m1)
        store.save(m2)
        store.delete(m2.id)
        assert len(store.get_all()) == 1
        assert store.get_all()[0].content == "keep"


class TestGetAll:
    def test_get_all_returns_all(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("a"))
        store.save(make_memory("b"))
        store.save(make_memory("c"))
        assert len(store.get_all()) == 3

    def test_get_all_scope_filter(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("private", scope=MemoryScope.PRIVATE))
        store.save(make_memory("shared", scope=MemoryScope.SHARED))
        all_private = store.get_all(scope=MemoryScope.PRIVATE)
        assert len(all_private) == 1
        assert all_private[0].scope == MemoryScope.PRIVATE

    def test_get_all_empty_store(self):
        store = InMemoryMemoryStore()
        assert store.get_all() == []


class TestEmptyState:
    def test_search_empty_store(self):
        store = InMemoryMemoryStore()
        assert store.search("anything") == []

    def test_get_by_tag_empty_store(self):
        store = InMemoryMemoryStore()
        assert store.get_by_tag("anytag") == []

    def test_delete_empty_store(self):
        store = InMemoryMemoryStore()
        assert store.delete("nothing") is False

    def test_get_all_empty_store(self):
        store = InMemoryMemoryStore()
        assert store.get_all() == []


class TestProtocol:
    def test_satisfies_memory_store_protocol(self):
        store = InMemoryMemoryStore()
        assert isinstance(store, MemoryStore)


class TestRanking:
    def test_higher_frequency_ranks_higher(self):
        store = InMemoryMemoryStore()
        store.save(make_memory("rare", frequency=1))
        store.save(make_memory("common", frequency=100))
        results = store.search("")
        # Empty query returns empty, so test with a matching query
        store2 = InMemoryMemoryStore()
        store2.save(make_memory("rare thing", frequency=1))
        store2.save(make_memory("common thing", frequency=100))
        results = store2.search("thing")
        assert results[0].content == "common thing"

    def test_recent_ranks_higher(self):
        store = InMemoryMemoryStore()
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=30)
        m_old = make_memory("old memory", frequency=5, last_accessed=old_time)
        m_new = make_memory("new memory", frequency=5, last_accessed=now)
        # Insert directly to control timestamps (save() always refreshes last_accessed)
        store._store[m_old.id] = m_old
        store._store[m_new.id] = m_new
        results = store.search("memory")
        assert results[0].content == "new memory"
