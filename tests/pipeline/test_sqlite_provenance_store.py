"""Tests for sr2.pipeline.stores.sqlite — SQLiteProvenanceStore.

Covers:
  FR9:   SQLiteProvenanceStore implementation backed by aiosqlite.
  FR15:  content_hash (SHA-256 of content_json) is computed and stored on write.
  AC1:   Round-trip across process restart: write → close → reopen → same result.
  AC6:   SQLiteProvenanceStore satisfies isinstance(store, ProvenanceStore).
  AC7:   Existing pipeline tests pass (no regressions — guarded by separate run).
"""

from __future__ import annotations

import aiosqlite
import pytest
import pytest_asyncio

from datetime import datetime, timezone

from sr2.models import Message, TextBlock
from sr2.pipeline.provenance import Entry, EntryOrigin, ProvenanceStore
from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_id() -> str:
    global _COUNTER
    _COUNTER += 1
    return f"{_COUNTER:026d}"


def make_entry(
    *,
    id: str | None = None,
    sources: tuple[str, ...] = (),
    origin_kind: str = "resolver",
    origin_name: str = "test_resolver",
    layer: str = "conversation",
    session_id: str = "session-abc",
    created_at: datetime | None = None,
    meta: dict | None = None,
) -> Entry:
    """Build a valid resolver-origin Entry by default.

    Pass origin_kind="transformer" with non-empty sources for transformer entries.
    Datetimes are rounded to whole seconds to avoid SQLite microsecond truncation
    surprises in round-trip comparisons.
    """
    if created_at is None:
        now = datetime.now(tz=timezone.utc)
        created_at = now.replace(microsecond=0)
    return Entry(
        id=id if id is not None else _next_id(),
        content=TextBlock(text="test content"),
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name=origin_name),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at,
        meta=meta if meta is not None else {},
    )


def make_message_entry(
    *,
    id: str | None = None,
    sources: tuple[str, ...] = (),
    origin_kind: str = "resolver",
    session_id: str = "session-abc",
    layer: str = "conversation",
    created_at: datetime | None = None,
) -> Entry:
    """Build a valid Entry backed by a Message content object."""
    if created_at is None:
        now = datetime.now(tz=timezone.utc)
        created_at = now.replace(microsecond=0)
    return Entry(
        id=id if id is not None else _next_id(),
        content=Message(role="user", content=[TextBlock(text="hello")]),
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name="test_resolver"),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at,
        meta={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store(tmp_path):
    """Fresh SQLiteProvenanceStore connected to an isolated DB file."""
    s = SQLiteProvenanceStore(path=tmp_path / "test.db")
    await s.connect()
    yield s
    await s.close()


# ---------------------------------------------------------------------------
# 1. Schema creation
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    @pytest.mark.asyncio
    async def test_entries_table_exists_after_connect(self, tmp_path):
        """After connect(), the `entries` table exists in the DB."""
        s = SQLiteProvenanceStore(path=tmp_path / "schema.db")
        await s.connect()
        try:
            async with aiosqlite.connect(tmp_path / "schema.db") as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
                )
                row = await cursor.fetchone()
            assert row is not None, "entries table was not created"
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_entry_sources_table_exists_after_connect(self, tmp_path):
        """After connect(), the `entry_sources` table exists in the DB."""
        s = SQLiteProvenanceStore(path=tmp_path / "schema2.db")
        await s.connect()
        try:
            async with aiosqlite.connect(tmp_path / "schema2.db") as db:
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='entry_sources'"
                )
                row = await cursor.fetchone()
            assert row is not None, "entry_sources table was not created"
        finally:
            await s.close()

    @pytest.mark.asyncio
    async def test_connect_twice_is_idempotent(self, tmp_path):
        """Calling connect() twice does not raise (CREATE TABLE IF NOT EXISTS)."""
        s = SQLiteProvenanceStore(path=tmp_path / "idempotent.db")
        await s.connect()
        await s.connect()  # must not raise
        await s.close()


# ---------------------------------------------------------------------------
# 2. Round-trip write / get
# ---------------------------------------------------------------------------


class TestRoundTripWriteGet:
    @pytest.mark.asyncio
    async def test_write_then_get_returns_equivalent_entry(self, store):
        """write() then get() returns an Entry equal on all key fields.

        FR9: SQLiteProvenanceStore satisfies the ProvenanceStore protocol.
        """
        entry = make_entry(id=_next_id())
        await store.write(entry)
        retrieved = await store.get(entry.id)

        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.layer == entry.layer
        assert retrieved.session_id == entry.session_id
        assert retrieved.origin.kind == entry.origin.kind
        assert retrieved.origin.name == entry.origin.name
        assert retrieved.created_at.replace(microsecond=0) == entry.created_at.replace(microsecond=0)
        assert retrieved.sources == entry.sources

    @pytest.mark.asyncio
    async def test_write_then_get_sources_round_trip(self, store):
        """sources tuple is stored in entry_sources and comes back correctly.

        The two genesis entries must be written first so the FK constraint is satisfied.
        """
        src1 = make_entry(id=_next_id())
        src2 = make_entry(id=_next_id())
        await store.write_batch([src1, src2])

        transformer = make_entry(
            id=_next_id(),
            sources=(src1.id, src2.id),
            origin_kind="transformer",
            origin_name="compaction",
        )
        await store.write(transformer)
        retrieved = await store.get(transformer.id)

        assert retrieved is not None
        assert isinstance(retrieved.sources, tuple)
        assert set(retrieved.sources) == {src1.id, src2.id}

    @pytest.mark.asyncio
    async def test_write_genesis_entry_sources_is_empty_tuple(self, store):
        """A genesis entry round-trips with sources==()."""
        entry = make_entry(id=_next_id(), sources=())
        await store.write(entry)
        retrieved = await store.get(entry.id)

        assert retrieved is not None
        assert retrieved.sources == ()

    @pytest.mark.asyncio
    async def test_write_batch_all_entries_retrievable(self, store):
        """write_batch() stores all entries; each is retrievable via get().

        FR9: batch write satisfies the protocol contract.
        """
        entries = [make_entry(id=_next_id()) for _ in range(4)]
        await store.write_batch(entries)

        for entry in entries:
            retrieved = await store.get(entry.id)
            assert retrieved is not None
            assert retrieved.id == entry.id

    @pytest.mark.asyncio
    async def test_write_batch_with_transformer_and_sources_in_same_call(self, store):
        """write_batch() handles FK-constrained transformer entries when sources are in the same batch."""
        src1 = make_entry(id=_next_id())
        src2 = make_entry(id=_next_id())
        transformer = make_entry(
            id=_next_id(),
            sources=(src1.id, src2.id),
            origin_kind="transformer",
            origin_name="compaction",
        )
        # All written in a single write_batch call
        await store.write_batch([src1, src2, transformer])
        retrieved = await store.get(transformer.id)
        assert retrieved is not None
        assert set(retrieved.sources) == {src1.id, src2.id}

    @pytest.mark.asyncio
    async def test_get_unknown_id_returns_none(self, store):
        """get() returns None for an entry_id not in the store."""
        result = await store.get("00000000000000000000000000")
        assert result is None

    @pytest.mark.asyncio
    async def test_message_content_round_trips(self, store):
        """An Entry with Message content serializes and deserializes without error."""
        entry = make_message_entry(id=_next_id())
        await store.write(entry)
        retrieved = await store.get(entry.id)

        assert retrieved is not None
        assert retrieved.id == entry.id
        assert isinstance(retrieved.content, Message)


# ---------------------------------------------------------------------------
# 3. get_lineage
# ---------------------------------------------------------------------------


class TestGetLineage:
    @pytest.mark.asyncio
    async def test_genesis_entry_lineage_contains_only_itself(self, store):
        """Single genesis entry → lineage contains just that entry."""
        genesis = make_entry(id=_next_id(), sources=())
        await store.write(genesis)

        lineage = await store.get_lineage(genesis.id)
        assert len(lineage) == 1
        assert lineage[0].id == genesis.id

    @pytest.mark.asyncio
    async def test_two_to_one_merge_lineage_contains_all_three(self, store):
        """Two sources + transformer whose sources=(src1.id, src2.id) → lineage has 3 entries."""
        src1 = make_entry(id=_next_id())
        src2 = make_entry(id=_next_id())
        await store.write_batch([src1, src2])

        transformer = make_entry(
            id=_next_id(),
            sources=(src1.id, src2.id),
            origin_kind="transformer",
            origin_name="summarization",
        )
        await store.write(transformer)

        lineage = await store.get_lineage(transformer.id)
        ids = {e.id for e in lineage}
        assert transformer.id in ids
        assert src1.id in ids
        assert src2.id in ids
        assert len(lineage) == 3

    @pytest.mark.asyncio
    async def test_depth_1_stops_at_immediate_parents(self, store):
        """depth=1 returns the entry and its direct parents; grandparents excluded.

        3-level chain: grandparent → parent → child
        """
        grandparent = make_entry(id=_next_id())
        await store.write(grandparent)

        parent = make_entry(
            id=_next_id(),
            sources=(grandparent.id,),
            origin_kind="transformer",
            origin_name="step1",
        )
        await store.write(parent)

        child = make_entry(
            id=_next_id(),
            sources=(parent.id,),
            origin_kind="transformer",
            origin_name="step2",
        )
        await store.write(child)

        lineage = await store.get_lineage(child.id, depth=1)
        ids = {e.id for e in lineage}
        assert child.id in ids
        assert parent.id in ids
        assert grandparent.id not in ids

    @pytest.mark.asyncio
    async def test_depth_minus_one_traverses_full_graph(self, store):
        """depth=-1 traverses the full ancestor graph across a 3-level chain."""
        grandparent = make_entry(id=_next_id())
        await store.write(grandparent)

        parent = make_entry(
            id=_next_id(),
            sources=(grandparent.id,),
            origin_kind="transformer",
            origin_name="step1",
        )
        await store.write(parent)

        child = make_entry(
            id=_next_id(),
            sources=(parent.id,),
            origin_kind="transformer",
            origin_name="step2",
        )
        await store.write(child)

        lineage = await store.get_lineage(child.id, depth=-1)
        ids = {e.id for e in lineage}
        assert child.id in ids
        assert parent.id in ids
        assert grandparent.id in ids
        assert len(lineage) == 3

    @pytest.mark.asyncio
    async def test_lineage_unknown_id_returns_empty_list(self, store):
        """get_lineage for an unknown id returns []."""
        result = await store.get_lineage("00000000000000000000000000")
        assert result == []


# ---------------------------------------------------------------------------
# 4. get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    @pytest.mark.asyncio
    async def test_returns_all_entries_for_session(self, store):
        """get_session returns every entry matching session_id."""
        sess = "session-xyz"
        entries = [make_entry(id=_next_id(), session_id=sess) for _ in range(3)]
        other = make_entry(id=_next_id(), session_id="other-session")
        await store.write_batch(entries + [other])

        result = await store.get_session(sess)
        ids = {e.id for e in result}
        for e in entries:
            assert e.id in ids
        assert other.id not in ids

    @pytest.mark.asyncio
    async def test_filters_by_layer(self, store):
        """get_session with layer= returns only entries for that layer."""
        sess = "session-layer-test"
        system_entry = make_entry(id=_next_id(), session_id=sess, layer="system")
        convo1 = make_entry(id=_next_id(), session_id=sess, layer="conversation")
        convo2 = make_entry(id=_next_id(), session_id=sess, layer="conversation")
        await store.write_batch([system_entry, convo1, convo2])

        result = await store.get_session(sess, layer="conversation")
        ids = {e.id for e in result}
        assert convo1.id in ids
        assert convo2.id in ids
        assert system_entry.id not in ids

    @pytest.mark.asyncio
    async def test_filters_by_since_boundary_inclusive(self, store):
        """get_session with since= returns entries at or after the boundary (inclusive).

        FR9: since filter is boundary-inclusive per the protocol contract.
        """
        sess = "session-since-test"
        t0 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        old = make_entry(id=_next_id(), session_id=sess, created_at=t0)
        at_boundary = make_entry(id=_next_id(), session_id=sess, created_at=t1)
        after = make_entry(id=_next_id(), session_id=sess, created_at=t2)
        await store.write_batch([old, at_boundary, after])

        result = await store.get_session(sess, since=t1)
        ids = {e.id for e in result}
        assert at_boundary.id in ids
        assert after.id in ids
        assert old.id not in ids

    @pytest.mark.asyncio
    async def test_unknown_session_returns_empty_list(self, store):
        """get_session for an unknown session_id returns []."""
        result = await store.get_session("no-such-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_layer_and_since_combined(self, store):
        """get_session with both layer= and since= applies both filters."""
        sess = "session-combined"
        t0 = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        cutoff = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)

        old_system = make_entry(id=_next_id(), session_id=sess, layer="system", created_at=t0)
        old_convo = make_entry(id=_next_id(), session_id=sess, layer="conversation", created_at=t0)
        new_system = make_entry(id=_next_id(), session_id=sess, layer="system", created_at=t2)
        new_convo = make_entry(id=_next_id(), session_id=sess, layer="conversation", created_at=t2)
        await store.write_batch([old_system, old_convo, new_system, new_convo])

        result = await store.get_session(sess, layer="conversation", since=cutoff)
        ids = {e.id for e in result}
        assert new_convo.id in ids
        assert old_convo.id not in ids
        assert new_system.id not in ids
        assert old_system.id not in ids


# ---------------------------------------------------------------------------
# 5. Persistence across restarts (AC1)
# ---------------------------------------------------------------------------


class TestPersistenceAcrossRestarts:
    @pytest.mark.asyncio
    async def test_entries_survive_close_and_reopen(self, tmp_path):
        """Write entries → close → reopen same DB file → entries are still there.

        AC1: process restart does not lose data.
        """
        db_path = tmp_path / "restart.db"

        src1 = make_entry(id=_next_id(), session_id="sess-restart")
        src2 = make_entry(id=_next_id(), session_id="sess-restart")
        transformer = make_entry(
            id=_next_id(),
            session_id="sess-restart",
            sources=(src1.id, src2.id),
            origin_kind="transformer",
            origin_name="summarization",
        )

        # First process: write and close
        s1 = SQLiteProvenanceStore(path=db_path)
        await s1.connect()
        await s1.write_batch([src1, src2, transformer])
        await s1.close()

        # Second process: reopen and query
        s2 = SQLiteProvenanceStore(path=db_path)
        await s2.connect()
        try:
            r1 = await s2.get(src1.id)
            r2 = await s2.get(src2.id)
            rt = await s2.get(transformer.id)

            assert r1 is not None and r1.id == src1.id
            assert r2 is not None and r2.id == src2.id
            assert rt is not None and rt.id == transformer.id

            lineage = await s2.get_lineage(transformer.id)
            ids = {e.id for e in lineage}
            assert transformer.id in ids
            assert src1.id in ids
            assert src2.id in ids
            assert len(lineage) == 3
        finally:
            await s2.close()

    @pytest.mark.asyncio
    async def test_session_query_survives_restart(self, tmp_path):
        """get_session results are consistent after closing and reopening the DB."""
        db_path = tmp_path / "session_restart.db"
        sess = "sess-session-restart"
        entries = [make_entry(id=_next_id(), session_id=sess) for _ in range(3)]

        s1 = SQLiteProvenanceStore(path=db_path)
        await s1.connect()
        await s1.write_batch(entries)
        await s1.close()

        s2 = SQLiteProvenanceStore(path=db_path)
        await s2.connect()
        try:
            result = await s2.get_session(sess)
            ids = {e.id for e in result}
            for e in entries:
                assert e.id in ids
            assert len(result) == 3
        finally:
            await s2.close()


# ---------------------------------------------------------------------------
# 6. isinstance check (AC6)
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_sqlite_store_satisfies_provenance_store_protocol(self, tmp_path):
        """SQLiteProvenanceStore satisfies isinstance(store, ProvenanceStore).

        AC6: runtime-checkable protocol check passes without connect().
        """
        store = SQLiteProvenanceStore(path=tmp_path / "proto.db")
        assert isinstance(store, ProvenanceStore)


# ---------------------------------------------------------------------------
# 7. content_hash stored (FR15)
# ---------------------------------------------------------------------------


class TestContentHash:
    @pytest.mark.asyncio
    async def test_content_hash_is_stored_as_non_empty_string(self, tmp_path):
        """After write(), the content_hash column is a non-empty string in the raw DB.

        FR15: SHA-256 of content_json is computed and stored on write.
        """
        db_path = tmp_path / "hash.db"
        s = SQLiteProvenanceStore(path=db_path)
        await s.connect()
        entry = make_entry(id=_next_id())
        await s.write(entry)
        await s.close()

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT content_hash FROM entries WHERE id = ?", (entry.id,)
            )
            row = await cursor.fetchone()

        assert row is not None, "entry row not found in raw DB"
        content_hash = row[0]
        assert isinstance(content_hash, str)
        assert len(content_hash) > 0, "content_hash must not be empty"

    @pytest.mark.asyncio
    async def test_content_hash_looks_like_sha256(self, tmp_path):
        """content_hash is a 64-character hex string (SHA-256 output size).

        FR15: verification that the stored hash is plausibly SHA-256.
        """
        db_path = tmp_path / "hash2.db"
        s = SQLiteProvenanceStore(path=db_path)
        await s.connect()
        entry = make_entry(id=_next_id())
        await s.write(entry)
        await s.close()

        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT content_hash FROM entries WHERE id = ?", (entry.id,)
            )
            row = await cursor.fetchone()

        assert row is not None
        content_hash = row[0]
        # SHA-256 produces 32 bytes = 64 hex characters
        assert len(content_hash) == 64
        assert all(c in "0123456789abcdef" for c in content_hash)
