"""SQLiteProvenanceStore — aiosqlite-backed ProvenanceStore implementation.

FR9:  SQLiteProvenanceStore satisfies the ProvenanceStore protocol.
FR15: content_hash (SHA-256 of content_json bytes) is computed and stored on write.
AC1:  Entries survive close → reopen (durable SQLite file).
AC6:  isinstance(store, ProvenanceStore) passes.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel

from sr2.models import ContentBlock, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.provenance import (
    ContextAuthority,
    ContextLifecycle,
    ContextReason,
    ContextScope,
    Entry,
    EntryOrigin,
    SessionSummary,
)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS entries (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    layer           TEXT NOT NULL,
    origin_kind     TEXT NOT NULL,
    origin_name     TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    content_json    TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    token_count     INTEGER,
    created_at      TEXT NOT NULL,
    meta_json       TEXT,
    reason          TEXT,
    authority       TEXT,
    lifecycle       TEXT,
    scope_json      TEXT
);

CREATE TABLE IF NOT EXISTS entry_sources (
    entry_id        TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    PRIMARY KEY (entry_id, source_id),
    FOREIGN KEY (entry_id) REFERENCES entries(id),
    FOREIGN KEY (source_id) REFERENCES entries(id)
);

CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
CREATE INDEX IF NOT EXISTS idx_entries_hash    ON entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_sources_source  ON entry_sources(source_id);
"""

# Entry columns added by the causal-metadata extension (sr2-92). Listed for
# the in-place migration below; older DBs simply have NULLs for them.
_CAUSAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("reason", "TEXT"),
    ("authority", "TEXT"),
    ("lifecycle", "TEXT"),
    ("scope_json", "TEXT"),
)

# Columns read by _row_to_entry, in order.
_SELECT_COLUMNS = (
    "id, session_id, layer, origin_kind, origin_name, "
    "content_type, content_json, content_hash, token_count, created_at, "
    "meta_json, reason, authority, lifecycle, scope_json"
)

# ---------------------------------------------------------------------------
# Content type registry for deserialization
# ---------------------------------------------------------------------------

_CONTENT_TYPES: dict[str, type] = {
    "TextBlock": TextBlock,
    "ToolUseBlock": ToolUseBlock,
    "ToolResultBlock": ToolResultBlock,
    "ThinkingBlock": ThinkingBlock,
    "Message": Message,
}


def _serialize_content(content: ContentBlock | Message) -> tuple[str, str]:
    """Return (content_type, content_json)."""
    content_type = type(content).__name__
    content_json = content.model_dump_json()
    return content_type, content_json


def _deserialize_content(content_type: str, content_json: str) -> ContentBlock | Message:
    cls = _CONTENT_TYPES.get(content_type)
    if cls is None:
        raise ValueError(f"Unknown content_type: {content_type!r}")
    return cls.model_validate_json(content_json)


def _hash_content(content_json: str) -> str:
    return hashlib.sha256(content_json.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Content polymorphism (Message.content holds heterogeneous block types)
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {"__type__": type(value).__name__, **value.model_dump()}
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        type_name = value.get("__type__")
        if type_name is not None:
            cls = _CONTENT_TYPES.get(type_name)
            if cls is None:
                raise ValueError(f"Unknown content type tag: {type_name!r}")
            fields = {k: _from_jsonable(v) for k, v in value.items() if k != "__type__"}
            return cls.model_validate(fields)
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


def _serialize_content(content: ContentBlock | Message) -> tuple[str, str]:
    """Return (content_type, content_json).

    Uses the polymorphic envelope (_to_jsonable) so Messages whose content
    mixes block types survive the JSON round-trip exactly.
    """
    content_type = type(content).__name__
    content_json = json.dumps(_to_jsonable(content), sort_keys=True)
    return content_type, content_json


def _deserialize_content(content_type: str, content_json: str) -> ContentBlock | Message:
    cls = _CONTENT_TYPES.get(content_type)
    if cls is None:
        raise ValueError(f"Unknown content_type: {content_type!r}")
    return cls.model_validate(_from_jsonable(json.loads(content_json)))


# ---------------------------------------------------------------------------
# Scope helpers
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


def _scope_from_json(raw: str | None) -> ContextScope:
    if not raw:
        return ContextScope(owner_id="default")
    data = json.loads(raw)
    return ContextScope(
        owner_id=data["owner_id"],
        agent_id=data.get("agent_id"),
        projects=tuple(data.get("projects") or ()),
        frame_id=data.get("frame_id"),
        topics=tuple(data.get("topics") or ()),
        sensitivity=tuple(data.get("sensitivity") or ()),
    )


def _row_to_entry(row: tuple[Any, ...], sources: tuple[str, ...]) -> Entry:
    """Convert a raw DB row + sources tuple into an Entry."""
    (
        id_,
        session_id,
        layer,
        origin_kind,
        origin_name,
        content_type,
        content_json,
        _content_hash,
        _token_count,
        created_at_str,
        meta_json,
        reason_str,
        authority_str,
        lifecycle_str,
        scope_json,
    ) = row

    content = _deserialize_content(content_type, content_json)
    created_at = datetime.fromisoformat(created_at_str)
    meta: dict[str, Any] = json.loads(meta_json) if meta_json else {}

    kwargs: dict[str, Any] = {}
    if reason_str is not None:
        kwargs["reason"] = ContextReason(reason_str)
    if authority_str is not None:
        kwargs["authority"] = ContextAuthority(authority_str)
    if lifecycle_str is not None:
        kwargs["lifecycle"] = ContextLifecycle(lifecycle_str)
    if scope_json is not None:
        kwargs["scope"] = _scope_from_json(scope_json)

    return Entry(
        id=id_,
        content=content,
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name=origin_name),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at,
        meta=meta,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SQLiteProvenanceStore
# ---------------------------------------------------------------------------


class SQLiteProvenanceStore:
    """Persistent ProvenanceStore backed by aiosqlite.

    Usage::

        store = SQLiteProvenanceStore(path="provenance.db")
        await store.connect()
        await store.write(entry)
        lineage = await store.get_lineage(entry.id)
        await store.close()
    """

    def __init__(self, path: str | Path | None = None, *, db_path: str | Path | None = None) -> None:
        resolved = db_path if path is None else path
        if resolved is None:
            raise ValueError("Either 'path' or 'db_path' must be provided")
        self._path = Path(resolved)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the DB connection and create the schema (idempotent).

        In-place migration (sr2-92): tables created before the causal
        metadata extension gain the reason/authority/lifecycle/scope_json
        columns via ALTER TABLE; existing rows keep NULLs and deserialize
        with Entry defaults.
        """
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA foreign_keys = ON")
        for statement in _DDL.strip().split(";\n"):
            stmt = statement.strip()
            if stmt:
                await self._db.execute(stmt)
        cursor = await self._db.execute("PRAGMA table_info(entries)")
        columns = {row[1] async for row in cursor}
        for name, sql_type in _CAUSAL_COLUMNS:
            if name not in columns:
                await self._db.execute(
                    f"ALTER TABLE entries ADD COLUMN {name} {sql_type}"
                )
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
            raise RuntimeError("SQLiteProvenanceStore: call connect() before using the store.")
        return self._db

    async def _insert_entry_row(self, entry: Entry, *, skip_existing: bool = False) -> None:
        db = self._assert_connected()
        content_type, content_json = _serialize_content(entry.content)
        content_hash = _hash_content(content_json)
        meta_json = json.dumps(entry.meta) if entry.meta else None
        # FR4 append-only: INSERT (never OR REPLACE). By default a duplicate
        # id hits the PRIMARY KEY constraint and the caller's transaction
        # rolls back; skip_existing keeps the stored row instead. Either way
        # a stored entry can never be silently overwritten.
        conflict = " ON CONFLICT(id) DO NOTHING" if skip_existing else ""
        await db.execute(
            """
            INSERT INTO entries
                (id, session_id, layer, origin_kind, origin_name,
                 content_type, content_json, content_hash,
                 token_count, created_at, meta_json,
                 reason, authority, lifecycle, scope_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            + conflict,
            (
                entry.id,
                entry.session_id,
                entry.layer,
                entry.origin.kind,
                entry.origin.name,
                content_type,
                content_json,
                content_hash,
                None,  # token_count not tracked in Phase 1
                entry.created_at.isoformat(),
                meta_json,
                entry.reason.value,
                entry.authority.value,
                entry.lifecycle.value,
                _scope_to_json(entry.scope),
            ),
        )

    async def _insert_sources(self, entry: Entry) -> None:
        db = self._assert_connected()
        for source_id in entry.sources:
            # FK enforcement (PRAGMA foreign_keys=ON) rejects a source_id
            # with no entries row — dangling lineage cannot be persisted.
            await db.execute(
                "INSERT OR IGNORE INTO entry_sources (entry_id, source_id) VALUES (?, ?)",
                (entry.id, source_id),
            )

    async def _fetch_sources(self, entry_id: str) -> tuple[str, ...]:
        db = self._assert_connected()
        cursor = await db.execute(
            "SELECT source_id FROM entry_sources WHERE entry_id = ?", (entry_id,)
        )
        rows = await cursor.fetchall()
        return tuple(r[0] for r in rows)

    async def _fetch_entries_by_ids(self, ids: list[str]) -> dict[str, Entry]:
        """Fetch entries by a list of IDs in one query. Returns {id: Entry}."""
        if not ids:
            return {}
        db = self._assert_connected()
        placeholders = ",".join("?" * len(ids))
        cursor = await db.execute(
            f"SELECT {_SELECT_COLUMNS} FROM entries WHERE id IN ({placeholders})",
            ids,
        )
        rows = await cursor.fetchall()
        result: dict[str, Entry] = {}
        for row in rows:
            entry_id = row[0]
            sources = await self._fetch_sources(entry_id)
            result[entry_id] = _row_to_entry(row, sources)
        return result

    # ------------------------------------------------------------------
    # ProvenanceStore protocol implementation
    # ------------------------------------------------------------------

    async def write(self, entry: Entry) -> None:
        """Persist a single entry and its source links.

        All-or-nothing: on any constraint failure (duplicate id — FR4;
        dangling source — FR5) the transaction rolls back and nothing from
        this write survives.
        """
        db = self._assert_connected()
        try:
            await self._insert_entry_row(entry)
            await self._insert_sources(entry)
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    async def write_batch(self, entries: list[Entry]) -> None:
        """Persist a list of entries in one all-or-nothing transaction.

        Inserts all entry rows first (satisfying FK constraints), then
        inserts entry_sources rows. This allows sources and transformer
        entries to coexist in the same batch call. On any constraint
        failure the whole batch rolls back — a rejected write leaves no
        entry or entry_sources residue.

        FR4: an entry whose id is already stored — earlier in this batch,
        by a previous write, or by another process sharing the DB file — is
        skipped and the first stored row wins. Content-addressed ids make
        re-persisting unchanged content a no-op.
        """
        db = self._assert_connected()
        try:
            # Pass 1: all entry rows (append-only: existing ids never rewritten)
            for entry in entries:
                await self._insert_entry_row(entry, skip_existing=True)
            # Pass 2: all source links (FKs now satisfied)
            for entry in entries:
                await self._insert_sources(entry)
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    async def get(self, entry_id: str) -> Entry | None:
        """Fetch a single entry by ID. Returns None if not found."""
        db = self._assert_connected()
        cursor = await db.execute(
            f"SELECT {_SELECT_COLUMNS} FROM entries WHERE id = ?",
            (entry_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        sources = await self._fetch_sources(entry_id)
        return _row_to_entry(row, sources)

    async def get_lineage(self, entry_id: str, depth: int = -1) -> list[Entry]:
        """BFS traversal following source links.

        depth=-1: unlimited.
        depth=N:  at most N hops from the start entry.
        Returns [] if entry_id is unknown.
        """
        # Check the start entry exists
        start = await self.get(entry_id)
        if start is None:
            return []

        visited: dict[str, Entry] = {}
        queue: deque[tuple[str, int]] = deque()
        queue.append((entry_id, 0))

        while queue:
            current_id, current_depth = queue.popleft()
            if current_id in visited:
                continue

            # Fetch in batch where possible; here we fetch one-at-a-time
            # but reuse the already-fetched start entry.
            if current_id == entry_id and start is not None:
                entry = start
            else:
                entry = await self.get(current_id)
            if entry is None:
                continue

            visited[current_id] = entry

            if depth == -1 or current_depth < depth:
                for source_id in entry.sources:
                    if source_id not in visited:
                        queue.append((source_id, current_depth + 1))

        return list(visited.values())

    async def get_session(
        self,
        session_id: str,
        layer: str | None = None,
        since: datetime | None = None,
    ) -> list[Entry]:
        """Return all entries for a session, optionally filtered by layer and since."""
        db = self._assert_connected()

        query = (
            f"SELECT {_SELECT_COLUMNS} FROM entries WHERE session_id = ?"
        )
        params: list[Any] = [session_id]

        if layer is not None:
            query += " AND layer = ?"
            params.append(layer)

        if since is not None:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        results: list[Entry] = []
        for row in rows:
            entry_id = row[0]
            sources = await self._fetch_sources(entry_id)
            results.append(_row_to_entry(row, sources))

        return results

    async def list_sessions(self, layer: str, limit: int) -> list[SessionSummary]:
        """One summary per session holding ``layer`` entries (FR 10).

        One read query over the indexed ``layer`` column — no schema
        change, no new index, no write-path change. ``created_at`` is
        stored as ISO-8601 text (a single UTC offset in the write path),
        whose lexical order equals instant order, so ``MAX`` yields the
        newest activity per session, read back loss-free with
        ``fromisoformat``. ``json_extract`` reads
        ``meta.turn_seq``; each distinct turn is keyed strictly by JSON
        type plus canonical value text — the mirror of the in-memory
        ``(type, value)`` key — so ``0``/``1``, ``False``/``True`` and
        int/float spellings each count as their own turn. ``json_valid``
        guards malformed ``meta_json``; ``json_type`` then classifies
        those rows (and absent or ``null`` turn_seq values) as no turn,
        while the session still appears with a zero turn count. Newest
        activity first, ties broken on session id for a total order
        matching the in-memory store; capped at ``limit``.
        """
        db = self._assert_connected()
        cursor = await db.execute(
            """
            SELECT s.session_id, s.last_activity,
                   (
                       SELECT COUNT(DISTINCT
                           CASE json_type(e2.meta_json, '$.turn_seq')
                               WHEN 'integer'
                               THEN 'integer:' || CAST(json_extract(e2.meta_json, '$.turn_seq') AS TEXT)
                               WHEN 'real'
                               THEN 'real:' || CAST(json_extract(e2.meta_json, '$.turn_seq') AS TEXT)
                               WHEN 'text'
                               THEN 'text:' || CAST(json_extract(e2.meta_json, '$.turn_seq') AS TEXT)
                               WHEN 'true'  THEN 'boolean:true'
                               WHEN 'false' THEN 'boolean:false'
                           END
                       )
                       FROM entries e2
                       WHERE e2.layer = ?
                         AND e2.session_id = s.session_id
                         AND json_valid(e2.meta_json) = 1
                   ) AS turn_count
            FROM (
                SELECT session_id, MAX(created_at) AS last_activity
                FROM entries
                WHERE layer = ?
                GROUP BY session_id
            ) AS s
            ORDER BY s.last_activity DESC, s.session_id ASC
            LIMIT ?
            """,
            (layer, layer, limit),
        )
        rows = await cursor.fetchall()
        return [
            SessionSummary(
                session_id=session_id,
                turn_count=turn_count,
                last_activity=datetime.fromisoformat(last_activity),
            )
            for session_id, last_activity, turn_count in rows
        ]
