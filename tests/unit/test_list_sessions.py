"""sr2-159: ``ProvenanceStore.list_sessions`` — resume-picker session summaries.

One summary per session holding entries in a layer: ``session_id``, the
count of distinct ``meta["turn_seq"]`` values, and the newest
``created_at`` — newest activity first, capped at ``limit`` (spec
session-resume.md FR 10 / AC11).

Case table (Confirmed by Diego, 2026-10-30):

  AC11-c1 — s1 turns 0-2 + s2 turn 0 in ``turn_archive`` plus a
            non-archive-layer s3 -> [s1(3), s2(1)] newest-first, s3 absent.
  AC11-c2 — same fixture, ``limit=1`` -> the newest session only.
  AC11-c3 — identical fixtures in InMemory and SQLite -> identical summaries.
  AC11-c4 — empty store -> ``[]``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sr2.models import TextBlock
from sr2.pipeline.provenance import (
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
    ProvenanceStore,
    SessionSummary,
    TURN_ARCHIVE_LAYER,
)
from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore


# ---------------------------------------------------------------------------
# Fixture — the one shared archive both stores load
# ---------------------------------------------------------------------------
#
# ``turn_archive`` entries:
#   s1: turns 0, 0 (dup), 1, 2  -> 3 distinct turns, newest 01-04
#   s2: turn 0                  -> 1 turn,          newest 01-05
#   s0: no turn_seq keys         -> 0 turns,         newest 01-02
# Non-archive noise (must never surface for the ``turn_archive`` query):
#   s3: a ``session``-layer turn and a ``prompt``-layer turn.
#
# Every entry also carries an ``ignored`` turn_seq on its non-archive twin so
# layer filtering, not turn counting, is what excludes s3.

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _at(hour: int) -> datetime:
    return _BASE.replace(hour=hour)


# (id, session_id, layer, hour, meta)
_FIXTURE: tuple[tuple[str, str, str, int, dict], ...] = (
    ("01", "s1", TURN_ARCHIVE_LAYER, 1, {"turn_seq": 0, "seq": 0}),
    ("02", "s1", TURN_ARCHIVE_LAYER, 2, {"turn_seq": 1, "seq": 0}),
    ("03", "s1", TURN_ARCHIVE_LAYER, 3, {"turn_seq": 2, "seq": 0}),
    ("04", "s1", TURN_ARCHIVE_LAYER, 4, {"turn_seq": 0, "seq": 1}),  # dup turn 0
    ("05", "s2", TURN_ARCHIVE_LAYER, 5, {"turn_seq": 0, "seq": 0}),
    ("06", "s0", TURN_ARCHIVE_LAYER, 2, {"seq": 9}),                 # no turn_seq
    ("07", "s3", "session", 6, {"turn_seq": 0}),                     # other layer
    ("08", "s3", "prompt", 7, {"turn_seq": 0}),                      # other layer
)

def _entry(row: tuple[str, str, str, int, dict]) -> Entry:
    entry_id, session_id, layer, hour, meta = row
    return Entry(
        id=entry_id,
        content=TextBlock(text=f"archived {entry_id}"),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="turn_archivist"),
        layer=layer,
        session_id=session_id,
        created_at=_at(hour),
        meta=meta,
    )


async def _load(store) -> None:
    for row in _FIXTURE:
        await store.write(_entry(row))


async def _memory_store() -> InMemoryProvenanceStore:
    store = InMemoryProvenanceStore()
    await _load(store)
    return store


@pytest.fixture
async def sqlite_store(tmp_path):
    store = SQLiteProvenanceStore(path=tmp_path / "list_sessions.db")
    await store.connect()
    await _load(store)
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# AC11-c1 — one row per archived session, newest-first, other layers excluded
# ---------------------------------------------------------------------------


async def test_list_sessions_summarizes_and_orders_newest_first():
    """AC11-c1: [s2(1)@01-05, s1(3)@01-04, s0(0)@01-02]; s3 absent."""
    store = await _memory_store()

    summaries = await store.list_sessions(TURN_ARCHIVE_LAYER, 10)

    assert summaries == [
        SessionSummary("s2", 1, _at(5)),
        SessionSummary("s1", 3, _at(4)),
        SessionSummary("s0", 0, _at(2)),
    ]
    listed = {summary.session_id for summary in summaries}
    assert "s3" not in listed
    # s1 spans turns 0,1,2 (the duplicate turn-0 row must not add a fourth).
    s1 = next(s for s in summaries if s.session_id == "s1")
    assert s1.turn_count == 3


async def test_list_sessions_honors_limit_newest_only():
    """AC11-c2: limit=1 keeps only the newest session (s2)."""
    store = await _memory_store()

    summaries = await store.list_sessions(TURN_ARCHIVE_LAYER, 1)

    assert summaries == [SessionSummary("s2", 1, _at(5))]


# ---------------------------------------------------------------------------
# AC11-c4 — an empty store lists nothing
# ---------------------------------------------------------------------------


async def test_empty_store_lists_no_sessions_memory():
    assert await InMemoryProvenanceStore().list_sessions(TURN_ARCHIVE_LAYER, 10) == []


async def test_empty_store_lists_no_sessions_sqlite(tmp_path):
    store = SQLiteProvenanceStore(path=tmp_path / "empty.db")
    await store.connect()
    try:
        assert await store.list_sessions(TURN_ARCHIVE_LAYER, 10) == []
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# AC11-c3 — InMemory and SQLite agree on identical fixtures
# ---------------------------------------------------------------------------


async def test_in_memory_and_sqlite_agree_on_identical_fixtures(sqlite_store):
    memory = await _memory_store()

    assert await memory.list_sessions(
        TURN_ARCHIVE_LAYER, 10
    ) == await sqlite_store.list_sessions(TURN_ARCHIVE_LAYER, 10)
    assert await memory.list_sessions(
        TURN_ARCHIVE_LAYER, 1
    ) == await sqlite_store.list_sessions(TURN_ARCHIVE_LAYER, 1)


async def test_sqlite_store_summarizes_and_orders_newest_first(sqlite_store):
    """Pin SQLite directly so agreement cannot mask a two-sided error."""
    summaries = await sqlite_store.list_sessions(TURN_ARCHIVE_LAYER, 10)

    assert summaries == [
        SessionSummary("s2", 1, _at(5)),
        SessionSummary("s1", 3, _at(4)),
        SessionSummary("s0", 0, _at(2)),
    ]


async def test_sqlite_store_honors_limit_newest_only(sqlite_store):
    summaries = await sqlite_store.list_sessions(TURN_ARCHIVE_LAYER, 1)

    assert summaries == [SessionSummary("s2", 1, _at(5))]


async def test_sqlite_last_activity_survives_the_text_round_trip(sqlite_store):
    """last_activity decodes back to the exact stored instant, tz-aware."""
    (summary,) = await sqlite_store.list_sessions(TURN_ARCHIVE_LAYER, 1)

    assert summary.last_activity == _at(5)
    assert summary.last_activity.tzinfo is not None


async def test_sqlite_skips_malformed_meta_without_failing(tmp_path):
    """A row with un-parseable ``meta_json`` must not crash the listing.

    The store only ever writes valid JSON, but a legacy or hand-corrupted
    row can hold malformed text. ``json_valid`` guards it: the row yields
    no turn yet the session still lists (with its activity) rather than
    raising. Written by direct SQL because the write path cannot produce
    such a row.
    """
    store = SQLiteProvenanceStore(path=tmp_path / "corrupt.db")
    await store.connect()
    try:
        await store.write(
            Entry(
                id="1",
                content=TextBlock(text="ok"),
                sources=(),
                origin=EntryOrigin(kind="resolver", name="turn_archivist"),
                layer=TURN_ARCHIVE_LAYER,
                session_id="s1",
                created_at=_at(1),
                meta={"turn_seq": 0},
            )
        )
        db = store._assert_connected()
        await db.execute(
            "INSERT INTO entries (id, session_id, layer, origin_kind, origin_name, "
            "content_type, content_json, content_hash, token_count, created_at, "
            "meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "2",
                "s1",
                TURN_ARCHIVE_LAYER,
                "resolver",
                "turn_archivist",
                "TextBlock",
                '{"text": "corrupt"}',
                "hash",
                None,
                _at(9).isoformat(),
                "{not valid json",
            ),
        )
        await db.commit()

        # The corrupt row contributes no turn but does carry activity:
        # s1 stays listed with its single real turn, newest activity 01-09.
        assert await store.list_sessions(TURN_ARCHIVE_LAYER, 10) == [
            SessionSummary("s1", 1, _at(9))
        ]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Distinct-turn counting is strict by type — both stores agree (FR 10)
# ---------------------------------------------------------------------------


async def test_distinct_turn_count_separates_types_both_stores(tmp_path):
    """0, 1, True, 'x' and a bool are each their own turn; absent/None are not.

    A store entry with an int turn_seq and one with a float turn_seq of the
    same numeric value count separately, matching SQLite's strict JSON-type
    key; a missing turn_seq is activity, never a turn.
    """
    rows = [
        ("1", "sx", {"turn_seq": 0}),
        ("2", "sx", {"turn_seq": 1}),
        ("3", "sx", {"turn_seq": True}),   # 'true' JSON type -> distinct from 1
        ("4", "sx", {"turn_seq": "x"}),    # text -> distinct
        ("5", "sx", {"turn_seq": None}),   # json null -> no turn, still activity
        ("6", "sx", {"seq": 1}),           # absent turn_seq -> no turn
    ]
    stamps = {i: _at(i + 10) for i in range(1, 7)}

    memory = InMemoryProvenanceStore()
    for entry_id, session_id, meta in rows:
        await memory.write(
            Entry(
                id=entry_id,
                content=TextBlock(text=entry_id),
                sources=(),
                origin=EntryOrigin(kind="resolver", name="turn_archivist"),
                layer=TURN_ARCHIVE_LAYER,
                session_id=session_id,
                created_at=stamps[int(entry_id)],
                meta=meta,
            )
        )

    sqlite = SQLiteProvenanceStore(path=tmp_path / "typed.db")
    await sqlite.connect()
    try:
        for entry_id, session_id, meta in rows:
            await sqlite.write(
                Entry(
                    id=entry_id,
                    content=TextBlock(text=entry_id),
                    sources=(),
                    origin=EntryOrigin(kind="resolver", name="turn_archivist"),
                    layer=TURN_ARCHIVE_LAYER,
                    session_id=session_id,
                    created_at=stamps[int(entry_id)],
                    meta=meta,
                )
            )

        memory_out = await memory.list_sessions(TURN_ARCHIVE_LAYER, 10)
        sqlite_out = await sqlite.list_sessions(TURN_ARCHIVE_LAYER, 10)

        # 4 distinct turns (0, 1, True-as-bool, "x"); None and absent add none.
        assert [s.turn_count for s in memory_out] == [4]
        assert memory_out == sqlite_out
    finally:
        await sqlite.close()


# ---------------------------------------------------------------------------
# Protocol surface (FR 10: "implemented on the Protocol")
# ---------------------------------------------------------------------------


def test_stores_satisfy_the_provenance_store_protocol():
    assert isinstance(InMemoryProvenanceStore(), ProvenanceStore)
