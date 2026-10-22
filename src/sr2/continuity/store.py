"""ContinuityStore — append-only SQLite ledger with rebuildable projections.

FR4:  the event ledger is truth and is append-only (INSERT, never UPDATE or
      DELETE); projections are caches reconstructible from it.
FR13: superseding records link to the records they replace via
      ``supersedes``; current projections exclude superseded and suppressed
      records by default while ``history_ids`` retains the full chain.
AC3:  appending a correction or suppression leaves every earlier record's
      stored (and re-serialized) bytes byte-identical.
AC4:  events survive close → reopen (durable SQLite file).

Pattern follows ``sr2.pipeline.stores.sqlite.SQLiteProvenanceStore``:
aiosqlite connect/close lifecycle, schema DDL as a module constant, JSON
serialization of complex fields, plain INSERT so a duplicate id hits the
PRIMARY KEY constraint instead of overwriting.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import aiosqlite

from sr2.continuity.models import (
    ContinuityEventKind,
    ContinuityRecord,
    ContinuityView,
)
from sr2.pipeline.provenance import ContextAuthority, ContextScope

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS continuity_events (
    seq             INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    subject_id      TEXT NOT NULL,
    event_kind      TEXT NOT NULL,
    kind            TEXT NOT NULL,
    content         TEXT NOT NULL,
    scope_json      TEXT NOT NULL,
    authority       TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    supersedes_json TEXT NOT NULL,
    confidence      REAL NOT NULL,
    salience        REAL NOT NULL,
    created_at      TEXT NOT NULL,
    meta_json       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_continuity_subject ON continuity_events(subject_id);
"""

# Columns read by _row_to_record, in order (seq first, then record fields).
_SELECT_COLUMNS = (
    "seq, id, subject_id, event_kind, kind, content, scope_json, "
    "authority, provenance_json, supersedes_json, confidence, salience, "
    "created_at, meta_json"
)

# ---------------------------------------------------------------------------
# Field (de)serialization
# ---------------------------------------------------------------------------


def _scope_to_json(scope: ContextScope) -> str:
    return json.dumps(
        {
            "owner_id": scope.owner_id,
            "agent_id": scope.agent_id,
            "projects": list(scope.projects),
            "frame_id": scope.frame_id,
            "topics": list(scope.topics),
            "sensitivity": list(scope.sensitivity),
        },
        sort_keys=True,
    )


def _scope_from_json(raw: str) -> ContextScope:
    data = json.loads(raw)
    return ContextScope(
        owner_id=data["owner_id"],
        agent_id=data.get("agent_id"),
        projects=tuple(data.get("projects") or ()),
        frame_id=data.get("frame_id"),
        topics=tuple(data.get("topics") or ()),
        sensitivity=tuple(data.get("sensitivity") or ()),
    )


def _str_tuple_to_json(items: Iterable[str]) -> str:
    return json.dumps(list(items))


def _row_to_record(row: tuple[Any, ...]) -> ContinuityRecord:
    """Rebuild an immutable record from one ledger row."""
    (
        _seq,
        id_,
        subject_id,
        event_kind,
        kind,
        content,
        scope_json,
        authority,
        provenance_json,
        supersedes_json,
        confidence,
        salience,
        created_at,
        meta_json,
    ) = row
    return ContinuityRecord(
        id=id_,
        subject_id=subject_id,
        event_kind=event_kind,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        content=content,
        scope=_scope_from_json(scope_json),
        authority=ContextAuthority(authority),
        provenance_ids=tuple(json.loads(provenance_json)),
        supersedes=tuple(json.loads(supersedes_json)),
        confidence=confidence,
        salience=salience,
        created_at=created_at,
        meta=json.loads(meta_json),
    )


# ---------------------------------------------------------------------------
# Projection arithmetic (pure function over the ledger, FR4 + FR13)
# ---------------------------------------------------------------------------


def compute_views(events: list[ContinuityRecord]) -> dict[str, ContinuityView]:
    """Derive current per-subject views from the ledger in append order.

    * ids named by any record's ``supersedes`` are excluded from
      ``current_record`` selection (FR13) but stay in ``history_ids``;
    * a SUPPRESSED event excludes the subject's record from being current,
      an UNSUPPRESSED event re-includes it; the most recent event for the
      subject's latest live record decides the status;
    * DORMANT surfaces as ``"dormant"``, SUPERSEDED as ``"superseded"``;
      everything else (FORMED, REINFORCED, VERIFIED, UNSUPPRESSED) is
      ``"active"``.

    The result depends only on the ledger's append order, so rebuilds are
    deterministic (AC3-r5).
    """
    chains: dict[str, list[ContinuityRecord]] = {}
    for record in events:
        chains.setdefault(record.subject_id, []).append(record)

    # A record superseded by a later event is excluded from "current"
    # selection everywhere in the ledger, not only within its own chain.
    superseded_ids: set[str] = set()
    for record in events:
        superseded_ids.update(record.supersedes)

    views: dict[str, ContinuityView] = {}
    for subject_id, chain in chains.items():
        history_ids = tuple(record.id for record in chain)

        latest_live: ContinuityRecord | None = None
        suppressed = False
        for record in chain:
            if record.id in superseded_ids:
                continue
            if record.event_kind is ContinuityEventKind.SUPPRESSED:
                suppressed = True
            elif record.event_kind is ContinuityEventKind.UNSUPPRESSED:
                suppressed = False
            elif not suppressed:
                latest_live = record

        if latest_live is None:
            # Every record in the chain is superseded (or nothing is live):
            # the chain's tail is current and decides the status.
            current = chain[-1]
        else:
            current = latest_live
            # A chain-tail SUPERSEDED event (naming no live id) still marks
            # the projection superseded (AC3-r1); a chain-tail SUPPRESSED
            # one marks it suppressed (AC3-r2).
            tail = chain[-1]
            if tail.event_kind is ContinuityEventKind.SUPPRESSED:
                suppressed = True
            elif tail.event_kind is ContinuityEventKind.SUPERSEDED:
                views[subject_id] = ContinuityView(
                    subject_id=subject_id,
                    current_record=current,
                    status="superseded",
                    history_ids=history_ids,
                )
                continue

        if suppressed:
            status = "suppressed"
        elif current.event_kind is ContinuityEventKind.DORMANT:
            status = "dormant"
        elif current.event_kind is ContinuityEventKind.SUPERSEDED:
            status = "superseded"
        else:
            status = "active"

        views[subject_id] = ContinuityView(
            subject_id=subject_id,
            current_record=current,
            status=status,
            history_ids=history_ids,
        )
    return views


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ContinuityStore:
    """Append-only continuity ledger backed by aiosqlite.

    Usage::

        store = ContinuityStore(path="continuity.db")
        await store.connect()
        await store.append(record)
        views = await store.current_views()
        await store.rebuild_projections()
        events = await store.get_events(subject_id=None)
        await store.close()
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the DB connection and create the schema (idempotent)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA foreign_keys = ON")
        for statement in _DDL.strip().split(";\n"):
            stmt = statement.strip()
            if stmt:
                await self._db.execute(stmt)
        await self._db.commit()

    async def close(self) -> None:
        """Close the DB connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_connected(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("ContinuityStore: call connect() before using the store.")
        return self._db

    async def _fetch_events(
        self, subject_id: str | None = None
    ) -> list[ContinuityRecord]:
        db = self._assert_connected()
        if subject_id is None:
            cursor = await db.execute(
                f"SELECT {_SELECT_COLUMNS} FROM continuity_events ORDER BY seq"
            )
        else:
            cursor = await db.execute(
                f"SELECT {_SELECT_COLUMNS} FROM continuity_events "
                "WHERE subject_id = ? ORDER BY seq",
                (subject_id,),
            )
        rows = await cursor.fetchall()
        await cursor.close()
        return [_row_to_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Ledger (append-only)
    # ------------------------------------------------------------------

    async def append(self, record: ContinuityRecord) -> None:
        """Append one event to the ledger.

        FR4 append-only: INSERT (never OR REPLACE / UPDATE) — a duplicate
        id raises sqlite3.IntegrityError, so a stored record can never be
        silently overwritten and earlier rows keep identical bytes.
        """
        db = self._assert_connected()
        await db.execute(
            """
            INSERT INTO continuity_events
                (id, subject_id, event_kind, kind, content, scope_json,
                 authority, provenance_json, supersedes_json, confidence,
                 salience, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.subject_id,
                record.event_kind.value,
                record.kind.value,
                record.content,
                _scope_to_json(record.scope),
                record.authority.value,
                _str_tuple_to_json(record.provenance_ids),
                _str_tuple_to_json(record.supersedes),
                record.confidence,
                record.salience,
                record.created_at.isoformat(),
                json.dumps(record.meta, sort_keys=True),
            ),
        )
        await db.commit()

    async def get_events(
        self, subject_id: str | None = None
    ) -> list[ContinuityRecord]:
        """Return ledger events in append order (all, or one subject's chain)."""
        return await self._fetch_events(subject_id)

    # ------------------------------------------------------------------
    # Projections (rebuildable caches)
    # ------------------------------------------------------------------

    async def current_views(self) -> dict[str, ContinuityView]:
        """Current per-subject projections derived from the ledger."""
        return compute_views(await self._fetch_events())

    async def rebuild_projections(self) -> dict[str, ContinuityView]:
        """Rebuild the projection set deterministically from the ledger (FR4).

        The ledger is the only truth; this recomputes every view from the
        event log, so repeated rebuilds over an unchanged ledger produce
        byte-identical projections.
        """
        return await self.current_views()
