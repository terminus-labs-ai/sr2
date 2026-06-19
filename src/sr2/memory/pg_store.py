"""PostgresMemoryStore — psycopg3-backed, persistent MemoryStore implementation.

Mirrors the behaviour of ``InMemoryMemoryStore`` (src/sr2/memory/store.py) but
persists to Postgres so memories survive process restarts and are visible across
independently-constructed store instances (multi-process / restart safety).

Behaviour parity with InMemory:
  * ``save`` of a NEW id keeps ``frequency`` as supplied (default 0) and sets
    ``last_accessed``; ``save`` of an EXISTING id sets
    ``frequency = existing.frequency + 1``. The caller's object is never mutated.
    The increment is atomic and based on PERSISTED state (INSERT ... ON CONFLICT
    ... DO UPDATE ... RETURNING) so there is no read-then-write race.
  * ``search`` / ``get_by_tag`` rank by the same ``freq x recency`` formula and
    do the ranking in Python to guarantee identical ordering to InMemory.

The synchronous psycopg3 API is used because the MemoryStore protocol is
synchronous (unlike the async SQLiteProvenanceStore).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import psycopg

from .schema import Memory, MemoryScope, MemorySearchResult


def _require_psycopg() -> "psycopg":
    """Import psycopg lazily, failing with an actionable error if it is missing.

    The Postgres backend is optional: psycopg is only needed when a
    ``PostgresMemoryStore`` is actually instantiated. Importing this module (or
    the wider ``sr2.memory`` package) must never require psycopg — a missing
    backend degrades to ``InMemoryMemoryStore`` instead of crashing every
    importer. See bead spc-72.
    """
    try:
        import psycopg
    except ImportError as exc:  # ModuleNotFoundError is an ImportError
        raise ImportError(
            "PostgresMemoryStore requires the optional 'psycopg' dependency, "
            "which is not installed in this environment. Install it with "
            "`pip install 'psycopg[binary]>=3.1'` (or use InMemoryMemoryStore "
            "instead)."
        ) from exc
    return psycopg

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    scope           TEXT NOT NULL,
    tags            TEXT[] NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL,
    frequency       INTEGER NOT NULL DEFAULT 0,
    last_accessed   TIMESTAMPTZ
);
"""

# Column order shared by SELECT statements and the row->model helper.
_COLUMNS = "id, content, scope, tags, created_at, frequency, last_accessed"


# ---------------------------------------------------------------------------
# Row <-> model helpers
# ---------------------------------------------------------------------------


def _row_to_memory(row: tuple[Any, ...]) -> Memory:
    """Convert a raw DB row (in ``_COLUMNS`` order) into a Memory."""
    id_, content, scope, tags, created_at, frequency, last_accessed = row
    return Memory(
        id=id_,
        content=content,
        scope=MemoryScope(scope),
        tags=list(tags) if tags is not None else [],
        created_at=_as_aware_utc(created_at),
        frequency=frequency,
        last_accessed=_as_aware_utc(last_accessed) if last_accessed else None,
    )


def _as_aware_utc(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (psycopg returns aware for timestamptz)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _rank_score(m: Memory) -> float:
    """Rank by frequency x recency — IDENTICAL to InMemoryMemoryStore._rank_score."""
    freq = max(m.frequency, 1)  # At least 1 so new memories aren't zero-scored
    recency = 1.0
    if m.last_accessed:
        delta = datetime.now(timezone.utc) - m.last_accessed
        days = max(delta.total_seconds() / 86400, 0.01)
        recency = 1.0 / (1.0 + days)  # Decays over time
    return freq * recency


def _rank(memories: list[Memory], limit: int) -> list[MemorySearchResult]:
    """Score, sort (desc), cap, and project memories to search results."""
    scored: list[tuple[float, Memory]] = [(_rank_score(m), m) for m in memories]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        MemorySearchResult(
            id=m.id,
            content=m.content,
            score=score,
            scope=m.scope,
            tags=m.tags,
        )
        for score, m in scored[:limit]
    ]


# ---------------------------------------------------------------------------
# PostgresMemoryStore
# ---------------------------------------------------------------------------


class PostgresMemoryStore:
    """Persistent MemoryStore backed by Postgres via psycopg3 (synchronous)."""

    def __init__(self, dsn: str) -> None:
        psycopg = _require_psycopg()
        self._conn = psycopg.connect(dsn)
        with self._conn.cursor() as cur:
            cur.execute(_DDL)
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(self, memory: Memory) -> Memory:
        """Persist a Memory. If the id already exists, increment frequency.

        Atomic upsert based on persisted state: a fresh INSERT keeps the supplied
        frequency; an ON CONFLICT increments the stored value by 1. The caller's
        object is never mutated — a new Memory built from the RETURNING values is
        returned.
        """
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories
                    (id, content, scope, tags, created_at, frequency, last_accessed)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    frequency = memories.frequency + 1,
                    last_accessed = EXCLUDED.last_accessed
                RETURNING frequency, last_accessed
                """,
                (
                    memory.id,
                    memory.content,
                    memory.scope.value,
                    list(memory.tags),
                    memory.created_at,
                    memory.frequency,
                    now,
                ),
            )
            frequency, last_accessed = cur.fetchone()
        self._conn.commit()

        return memory.model_copy(
            update={
                "frequency": frequency,
                "last_accessed": _as_aware_utc(last_accessed),
            }
        )

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self, query: str, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemorySearchResult]:
        """Keyword match on content and tags, ranked by frequency x recency."""
        if not query:
            return []

        pattern = f"%{query}%"
        sql = (
            f"SELECT {_COLUMNS} FROM memories "
            "WHERE (content ILIKE %s "
            "OR EXISTS (SELECT 1 FROM unnest(tags) AS t WHERE t ILIKE %s))"
        )
        params: list[Any] = [pattern, pattern]
        if scope is not None:
            sql += " AND scope = %s"
            params.append(scope.value)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        memories = [_row_to_memory(row) for row in rows]
        return _rank(memories, limit)

    # ------------------------------------------------------------------
    # get_by_tag
    # ------------------------------------------------------------------

    def get_by_tag(
        self, tag: str, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemorySearchResult]:
        """Exact (case-insensitive) tag membership, ranked by frequency x recency."""
        tag_lower = tag.lower()
        sql = (
            f"SELECT {_COLUMNS} FROM memories "
            "WHERE EXISTS (SELECT 1 FROM unnest(tags) AS t WHERE lower(t) = %s)"
        )
        params: list[Any] = [tag_lower]
        if scope is not None:
            sql += " AND scope = %s"
            params.append(scope.value)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        memories = [_row_to_memory(row) for row in rows]
        return _rank(memories, limit)

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete(self, memory_id: str) -> bool:
        """Remove by id. Returns True if a row was deleted."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            deleted = cur.rowcount
        self._conn.commit()
        return deleted > 0

    # ------------------------------------------------------------------
    # get_all
    # ------------------------------------------------------------------

    def get_all(self, scope: MemoryScope | None = None) -> list[Memory]:
        """Return all memories, optionally filtered by scope."""
        sql = f"SELECT {_COLUMNS} FROM memories"
        params: list[Any] = []
        if scope is not None:
            sql += " WHERE scope = %s"
            params.append(scope.value)

        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [_row_to_memory(row) for row in rows]
