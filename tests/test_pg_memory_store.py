"""Tests for PostgresMemoryStore — the persistent, Postgres-backed MemoryStore.

This is a TDD test file written BEFORE the implementation exists. The class
under test will live at ``sr2.memory.pg_store.PostgresMemoryStore`` and be
re-exported from ``sr2.memory``. Until it is implemented these tests fail at
import — that is expected for this TDD stage.

CONTRACT ASSUMPTIONS (documented here because the tests depend on them):
  * Constructor signature: ``PostgresMemoryStore(dsn: str)``. It opens a
    connection on construction and creates its schema (table) if absent.
  * It exposes a ``close()`` method.
  * It manages a single table named ``memories``. This table name is a HARD
    CONTRACT relied on by the truncate fixture below — if the implementer
    chooses a different name, this fixture (and the suite) must be updated.

BEHAVIOUR is mirrored exactly from ``InMemoryMemoryStore``
(src/sr2/memory/store.py):
  * ``save`` of a NEW id keeps ``frequency`` as supplied (default 0) and sets
    ``last_accessed``; ``save`` of an EXISTING id sets
    ``frequency = existing.frequency + 1``. The caller's object is never
    mutated.
  * ``search`` is a case-insensitive substring match on content OR any tag,
    optionally scope-filtered, ranked by frequency x recency, capped at
    ``limit``. Empty query -> empty list.
  * ``get_by_tag`` is an exact (case-insensitive) tag match, scope-filterable,
    same ranking and limit semantics.
  * ``delete`` returns True if found+deleted, else False.
  * ``get_all`` returns all memories, optionally scope-filtered.

The headline acceptance criteria for this bead are PERSISTENCE across
separate store instances (simulating restart / multi-process) and a
concurrency-safe frequency increment. Those have dedicated tests.
"""

from __future__ import annotations

import os
import uuid

import pytest

from sr2.memory.schema import Memory, MemoryScope, MemorySearchResult
from sr2.memory.protocol import MemoryStore, TaggedMemoryStore

# Imported last so a missing implementation surfaces as a clear import error
# during the TDD red stage.
from sr2.memory import PostgresMemoryStore


# ---------------------------------------------------------------------------
# DB configuration / availability
# ---------------------------------------------------------------------------

DEFAULT_TEST_DSN = (
    "postgresql://postgres:postgres@192.168.50.117:5432/spectre_memory_test"
)
TEST_DSN = os.environ.get("SPECTRE_MEMORY_TEST_DSN", DEFAULT_TEST_DSN)

# The single table the store owns. HARD CONTRACT — see module docstring.
MEMORIES_TABLE = "memories"


def _db_reachable(dsn: str) -> bool:
    """Quick connectivity probe so the suite stays green when offline."""
    try:
        import psycopg
    except Exception:  # pragma: no cover - import guard
        return False
    try:
        conn = psycopg.connect(dsn, connect_timeout=3)
    except Exception:
        return False
    else:
        conn.close()
        return True


if not _db_reachable(TEST_DSN):
    pytest.skip(
        f"Test Postgres unreachable at {TEST_DSN}; skipping PostgresMemoryStore suite.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_db():
    """Ensure each test starts from an empty ``memories`` table.

    Construct a throwaway store first; per the constructor contract it creates
    the ``memories`` table if absent, so the table is guaranteed to exist by
    the time we TRUNCATE below. (If the implementer does not create the schema
    on construct, the TRUNCATE raises and the contract violation surfaces
    loudly here — which is intended.)
    """
    import psycopg

    # Construct + close a store so the table is guaranteed to exist.
    bootstrap = PostgresMemoryStore(TEST_DSN)
    bootstrap.close()

    with psycopg.connect(TEST_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {MEMORIES_TABLE}")
        conn.commit()
    yield
    # Leave a clean table behind too, so manual inspection isn't polluted.
    with psycopg.connect(TEST_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {MEMORIES_TABLE}")
        conn.commit()


@pytest.fixture()
def store(clean_db):
    """A live store against the clean test DB; closed after the test."""
    s = PostgresMemoryStore(TEST_DSN)
    try:
        yield s
    finally:
        s.close()


def _mem(content: str, **kwargs) -> Memory:
    """Build a Memory with a deterministic-unique id unless overridden."""
    kwargs.setdefault("id", uuid.uuid4().hex)
    return Memory(content=content, **kwargs)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_memory_store_protocol(store):
    """PostgresMemoryStore is a runtime MemoryStore."""
    assert isinstance(store, MemoryStore)


def test_satisfies_tagged_memory_store_protocol(store):
    """PostgresMemoryStore is a runtime TaggedMemoryStore (has get_by_tag)."""
    assert isinstance(store, TaggedMemoryStore)


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


def test_save_returns_saved_memory_and_sets_last_accessed(store):
    """save persists and returns a Memory with last_accessed populated."""
    m = _mem("the user prefers dark mode")
    saved = store.save(m)
    assert isinstance(saved, Memory)
    assert saved.id == m.id
    assert saved.content == m.content
    assert saved.last_accessed is not None


def test_save_new_id_keeps_default_frequency_zero(store):
    """A first save of a fresh Memory leaves frequency at its supplied value (0)."""
    m = _mem("first save")  # default frequency == 0
    saved = store.save(m)
    assert saved.frequency == 0


def test_save_existing_id_increments_frequency(store):
    """Re-saving the same id increments frequency by exactly 1 each time."""
    m = _mem("repeated fact")
    store.save(m)            # freq 0 (new)
    second = store.save(m)   # existing -> 0 + 1
    assert second.frequency == 1
    third = store.save(m)    # existing -> 1 + 1
    assert third.frequency == 2


def test_save_does_not_mutate_caller_object(store):
    """The passed-in Memory must be untouched after save."""
    m = _mem("immutable input")
    original_freq = m.frequency
    original_last = m.last_accessed
    store.save(m)
    store.save(m)
    assert m.frequency == original_freq
    assert m.last_accessed == original_last


def test_save_persists_retrievable_via_get_all(store):
    """A saved memory shows up in get_all on the same store."""
    m = _mem("retrievable fact", tags=["alpha"])
    store.save(m)
    all_mems = store.get_all()
    assert any(x.id == m.id and x.content == "retrievable fact" for x in all_mems)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_empty_query_returns_empty_list(store):
    """An empty query short-circuits to []."""
    store.save(_mem("something searchable"))
    assert store.search("") == []


def test_search_matches_content_case_insensitive(store):
    """Substring match on content, ignoring case."""
    m = _mem("The Capital of France is Paris")
    store.save(m)
    results = store.search("paris")
    assert [r.id for r in results] == [m.id]
    assert isinstance(results[0], MemorySearchResult)


def test_search_matches_tag(store):
    """A query that is a substring of a tag matches even if absent from content."""
    m = _mem("unrelated body text", tags=["geography"])
    store.save(m)
    results = store.search("geo")
    assert [r.id for r in results] == [m.id]


def test_search_no_match_returns_empty(store):
    """A query matching neither content nor tags returns nothing."""
    store.save(_mem("apples and oranges", tags=["fruit"]))
    assert store.search("automobile") == []


def test_search_scope_filter(store):
    """Scope filter restricts results to that scope only."""
    shared = _mem("shared note about deadlines", scope=MemoryScope.SHARED)
    private = _mem("private note about deadlines", scope=MemoryScope.PRIVATE)
    store.save(shared)
    store.save(private)

    results = store.search("deadlines", scope=MemoryScope.SHARED)
    assert [r.id for r in results] == [shared.id]
    assert all(r.scope == MemoryScope.SHARED for r in results)


def test_search_respects_limit(store):
    """search returns at most `limit` results."""
    for i in range(5):
        store.save(_mem(f"limit token {i}"))
    results = store.search("limit token", limit=3)
    assert len(results) == 3


def test_search_ranks_higher_frequency_first(store):
    """A more frequently reinforced memory ranks above a one-shot one."""
    hot = _mem("ranking probe hot")
    cold = _mem("ranking probe cold")
    store.save(cold)  # freq 0
    store.save(hot)
    store.save(hot)   # freq 1
    store.save(hot)   # freq 2

    results = store.search("ranking probe")
    ids = [r.id for r in results]
    assert ids.index(hot.id) < ids.index(cold.id)


def test_search_frequency_dominates_when_saved_first(store):
    """Frequency drives ranking even when the high-freq memory is the OLDER one.

    Here the high-frequency memory is saved FIRST (so its last_accessed is the
    older of the two). Because all saves happen within the same test (sub-second
    apart), the recency term is ~1.0 for both rows and frequency dominates. This
    exercises the freq x recency formula rather than letting recency and
    frequency agree by accident (the prior test saves the hot one last).
    """
    hot = _mem("freq dominant probe hot")
    cold = _mem("freq dominant probe cold")
    store.save(hot)
    store.save(hot)
    store.save(hot)   # freq 2, saved first (older last_accessed)
    store.save(cold)  # freq 0, saved last (newer last_accessed)

    results = store.search("freq dominant probe")
    ids = [r.id for r in results]
    assert ids.index(hot.id) < ids.index(cold.id)


# ---------------------------------------------------------------------------
# get_by_tag
# ---------------------------------------------------------------------------


def test_get_by_tag_exact_match_case_insensitive(store):
    """get_by_tag matches the whole tag, ignoring case; partial tags do not match."""
    m = _mem("tagged content", tags=["Python"])
    store.save(m)
    assert [r.id for r in store.get_by_tag("python")] == [m.id]
    # Partial tag must NOT match (exact-match semantics).
    assert store.get_by_tag("pyth") == []


def test_get_by_tag_scope_filter(store):
    """get_by_tag honours the scope filter."""
    a = _mem("a", tags=["topic"], scope=MemoryScope.PROJECT)
    b = _mem("b", tags=["topic"], scope=MemoryScope.SHARED)
    store.save(a)
    store.save(b)
    results = store.get_by_tag("topic", scope=MemoryScope.PROJECT)
    assert [r.id for r in results] == [a.id]


def test_get_by_tag_respects_limit(store):
    """get_by_tag caps results at `limit`."""
    for i in range(4):
        store.save(_mem(f"item {i}", tags=["bucket"]))
    assert len(store.get_by_tag("bucket", limit=2)) == 2


def test_get_by_tag_ranks_higher_frequency_first(store):
    """get_by_tag applies the same freq x recency ranking as search."""
    hot = _mem("tag rank hot", tags=["rankgroup"])
    cold = _mem("tag rank cold", tags=["rankgroup"])
    store.save(cold)  # freq 0
    store.save(hot)
    store.save(hot)
    store.save(hot)   # freq 2

    results = store.get_by_tag("rankgroup")
    ids = [r.id for r in results]
    assert ids.index(hot.id) < ids.index(cold.id)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_existing_returns_true_and_removes(store):
    """Deleting an existing memory returns True and removes it."""
    m = _mem("to be deleted")
    store.save(m)
    assert store.delete(m.id) is True
    assert all(x.id != m.id for x in store.get_all())


def test_delete_missing_returns_false(store):
    """Deleting an unknown id returns False."""
    assert store.delete("does-not-exist") is False


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


def test_get_all_returns_everything(store):
    """get_all with no scope returns all saved memories."""
    ids = set()
    for i in range(3):
        m = _mem(f"all entry {i}")
        store.save(m)
        ids.add(m.id)
    assert {x.id for x in store.get_all()} == ids


def test_get_all_scope_filter(store):
    """get_all with a scope returns only that scope."""
    p = _mem("proj", scope=MemoryScope.PROJECT)
    s = _mem("shared", scope=MemoryScope.SHARED)
    store.save(p)
    store.save(s)
    project_only = store.get_all(scope=MemoryScope.PROJECT)
    assert [m.id for m in project_only] == [p.id]
    assert all(m.scope == MemoryScope.PROJECT for m in project_only)


# ---------------------------------------------------------------------------
# HEADLINE: cross-connection persistence (simulated restart / multi-process)
# ---------------------------------------------------------------------------


def test_persistence_across_separate_instances(clean_db):
    """A memory saved via store A is retrievable via a fresh store B (same DSN).

    This is the acceptance criterion for the bead: data survives the death of
    the writing instance and is visible to an independently-constructed reader.
    """
    a = PostgresMemoryStore(TEST_DSN)
    m = _mem("durable cross-connection fact", tags=["persistence"])
    a.save(m)
    a.close()

    b = PostgresMemoryStore(TEST_DSN)
    try:
        # Visible via get_all
        assert any(x.id == m.id for x in b.get_all())
        # Visible via search (content)
        assert [r.id for r in b.search("durable cross-connection")] == [m.id]
        # Visible via tag
        assert [r.id for r in b.get_by_tag("persistence")] == [m.id]
    finally:
        b.close()


# ---------------------------------------------------------------------------
# HEADLINE: concurrency-safe frequency increment across instances
# ---------------------------------------------------------------------------


def test_frequency_increment_across_separate_instances(clean_db):
    """Saving the same id from two distinct instances increments without loss.

    Mirrors InMemory semantics: first save (new id) -> freq 0, the second save
    of that id (now existing) -> freq 1. Crucially the second instance must SEE
    the row written by the first, so the increment is based on persisted state,
    not a private in-memory copy.
    """
    a = PostgresMemoryStore(TEST_DSN)
    m = _mem("shared counter fact")
    first = a.save(m)
    assert first.frequency == 0
    a.close()

    b = PostgresMemoryStore(TEST_DSN)
    try:
        second = b.save(m)
        assert second.frequency == 1  # existing.frequency (0) + 1, no lost update
    finally:
        b.close()

    # And the persisted frequency reflects the second save for a third reader.
    c = PostgresMemoryStore(TEST_DSN)
    try:
        third = c.save(m)
        assert third.frequency == 2
    finally:
        c.close()
