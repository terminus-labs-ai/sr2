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

from sr2.models import ContentBlock, Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.provenance import Entry, EntryOrigin

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
    meta_json       TEXT
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
    ) = row

    content = _deserialize_content(content_type, content_json)
    created_at = datetime.fromisoformat(created_at_str)
    meta: dict[str, Any] = json.loads(meta_json) if meta_json else {}

    return Entry(
        id=id_,
        content=content,
        sources=sources,
        origin=EntryOrigin(kind=origin_kind, name=origin_name),  # type: ignore[arg-type]
        layer=layer,
        session_id=session_id,
        created_at=created_at,
        meta=meta,
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
        """Open the DB connection and create the schema (idempotent)."""
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
            raise RuntimeError("SQLiteProvenanceStore: call connect() before using the store.")
        return self._db

    async def _insert_entry_row(self, entry: Entry) -> None:
        db = self._assert_connected()
        content_type, content_json = _serialize_content(entry.content)
        content_hash = _hash_content(content_json)
        meta_json = json.dumps(entry.meta) if entry.meta else None
        await db.execute(
            """
            INSERT OR REPLACE INTO entries
                (id, session_id, layer, origin_kind, origin_name,
                 content_type, content_json, content_hash,
                 token_count, created_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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
            ),
        )

    async def _insert_sources(self, entry: Entry) -> None:
        db = self._assert_connected()
        for source_id in entry.sources:
            await db.execute(
                "INSERT OR REPLACE INTO entry_sources (entry_id, source_id) VALUES (?, ?)",
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
            f"SELECT id, session_id, layer, origin_kind, origin_name, "
            f"content_type, content_json, content_hash, token_count, created_at, meta_json "
            f"FROM entries WHERE id IN ({placeholders})",
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
        """Persist a single entry and its source links."""
        db = self._assert_connected()
        await self._insert_entry_row(entry)
        await self._insert_sources(entry)
        await db.commit()

    async def write_batch(self, entries: list[Entry]) -> None:
        """Persist a list of entries.

        Inserts all entry rows first (satisfying FK constraints), then inserts
        entry_sources rows. This allows sources and transformer entries to
        coexist in the same batch call.
        """
        db = self._assert_connected()
        # Pass 1: all entry rows
        for entry in entries:
            await self._insert_entry_row(entry)
        # Pass 2: all source links (FKs now satisfied)
        for entry in entries:
            await self._insert_sources(entry)
        await db.commit()

    async def get(self, entry_id: str) -> Entry | None:
        """Fetch a single entry by ID. Returns None if not found."""
        db = self._assert_connected()
        cursor = await db.execute(
            "SELECT id, session_id, layer, origin_kind, origin_name, "
            "content_type, content_json, content_hash, token_count, created_at, meta_json "
            "FROM entries WHERE id = ?",
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
            "SELECT id, session_id, layer, origin_kind, origin_name, "
            "content_type, content_json, content_hash, token_count, created_at, meta_json "
            "FROM entries WHERE session_id = ?"
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
