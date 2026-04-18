"""Memory storage backends: protocol, in-memory (testing), and SQLite."""

import aiosqlite
import json
import logging
import math
import re
from datetime import datetime
from typing import Protocol

from sr2.memory.schema import Memory, MemorySearchResult

logger = logging.getLogger(__name__)


class MemoryStore(Protocol):
    """Protocol for memory storage backends."""

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        """Save a memory. If ID exists, update it."""
        ...

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID. Returns None if not found."""
        ...

    async def get_by_key(
        self,
        key: str,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[Memory]:
        """Get all memories with a given key. Sorted by extracted_at desc."""
        ...

    async def search_by_key_prefix(
        self, prefix: str, include_archived: bool = False
    ) -> list[Memory]:
        """Get all memories whose key starts with prefix."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted, False if not found."""
        ...

    async def archive(self, memory_id: str) -> bool:
        """Set archived=True on a memory. Returns True if found."""
        ...

    async def search_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search by vector embedding. Returns top_k results sorted by similarity."""
        ...

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Keyword search across key and value fields."""
        ...

    async def count(self, include_archived: bool = False) -> int:
        """Count total memories."""
        ...

    async def list_scope_refs(
        self,
        scope_filter: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[tuple[str, str | None]]:
        """Return distinct (scope, scope_ref) pairs. Sorted by scope, scope_ref."""
        ...

    async def list_top_keys(
        self,
        scope_ref: str,
        limit: int = 30,
    ) -> list[str]:
        """Return the most-accessed distinct keys for a scope_ref.

        Used to provide key hints during extraction so the LLM reuses
        existing keys instead of inventing synonyms. Sorted by access_count
        descending, then key alphabetically.
        """
        ...

    async def list_memories(
        self,
        *,
        memory_types: list[str] | None = None,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
        older_than: datetime | None = None,
        min_access_count: int | None = None,
        max_access_count: int | None = None,
        limit: int = 500,
    ) -> list[Memory]:
        """List memories matching filter criteria.

        All filters combine with AND logic. None means no filter,
        empty list means match nothing. Returns at most `limit` memories.
        """
        ...


class InMemoryMemoryStore:
    """In-memory implementation for testing. No vector search — uses simple string matching."""

    def __init__(self) -> None:
        self._memories: dict[str, Memory] = {}

    def _scope_match(
        self,
        memory: Memory,
        scope_filter: list[str] | None,
        scope_refs: list[str] | None,
    ) -> bool:
        """Check if a memory matches the scope filter criteria."""
        if scope_filter is None:
            return True
        if memory.scope not in scope_filter:
            return False
        if scope_refs is not None:
            # Allow memories with no scope_ref (legacy) or matching scope_ref
            if memory.scope_ref is not None and memory.scope_ref not in scope_refs:
                return False
        return True

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        self._memories[memory.id] = memory

    async def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    async def get_by_key(
        self,
        key: str,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[Memory]:
        results = [m for m in self._memories.values() if m.key == key]
        if not include_archived:
            results = [m for m in results if not m.archived]
        results = [m for m in results if self._scope_match(m, scope_filter, scope_refs)]
        return sorted(results, key=lambda m: m.extracted_at, reverse=True)

    async def search_by_key_prefix(
        self, prefix: str, include_archived: bool = False
    ) -> list[Memory]:
        results = [m for m in self._memories.values() if m.key.startswith(prefix)]
        if not include_archived:
            results = [m for m in results if not m.archived]
        return sorted(results, key=lambda m: m.extracted_at, reverse=True)

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    async def archive(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            self._memories[memory_id].archived = True
            return True
        return False

    async def search_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        mems = [m for m in self._memories.values() if not m.archived or include_archived]
        mems = [m for m in mems if self._scope_match(m, scope_filter, scope_refs)]
        mems.sort(key=lambda m: m.id)
        return [
            MemorySearchResult(memory=m, relevance_score=0.5, match_type="semantic")
            for m in mems[:top_k]
        ]

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        words = list({w.lower() for w in re.split(r"[\s,!?.;:'\"]+", query) if len(w) >= 3})
        if not words:
            words = [query.strip().lower()]
        results = []
        for m in self._memories.values():
            if m.archived and not include_archived:
                continue
            if not self._scope_match(m, scope_filter, scope_refs):
                continue
            key_lower = m.key.lower()
            value_lower = m.value.lower()
            if any(w in key_lower or w in value_lower for w in words):
                results.append(
                    MemorySearchResult(memory=m, relevance_score=0.7, match_type="keyword")
                )
        results.sort(key=lambda r: r.memory.id)
        return results[:top_k]

    async def count(self, include_archived: bool = False) -> int:
        if include_archived:
            return len(self._memories)
        return sum(1 for m in self._memories.values() if not m.archived)

    async def list_scope_refs(
        self,
        scope_filter: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[tuple[str, str | None]]:
        pairs: set[tuple[str, str | None]] = set()
        for m in self._memories.values():
            if not include_archived and m.archived:
                continue
            if scope_filter is not None and m.scope not in scope_filter:
                continue
            pairs.add((m.scope, m.scope_ref))
        return sorted(pairs)

    async def list_top_keys(
        self,
        scope_ref: str,
        limit: int = 30,
    ) -> list[str]:
        seen: dict[str, int] = {}
        for m in self._memories.values():
            if m.archived or m.scope_ref != scope_ref:
                continue
            if m.key not in seen or m.access_count > seen[m.key]:
                seen[m.key] = m.access_count
        ranked = sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        return [k for k, _ in ranked[:limit]]

    async def list_memories(
        self,
        *,
        memory_types: list[str] | None = None,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
        older_than: datetime | None = None,
        min_access_count: int | None = None,
        max_access_count: int | None = None,
        limit: int = 500,
    ) -> list[Memory]:
        results: list[Memory] = []
        for m in self._memories.values():
            if not include_archived and m.archived:
                continue
            if memory_types is not None and m.memory_type not in memory_types:
                continue
            if scope_filter is not None and m.scope not in scope_filter:
                continue
            if scope_refs is not None:
                if m.scope_ref is None or m.scope_ref not in scope_refs:
                    continue
            if older_than is not None and m.extracted_at >= older_than:
                continue
            if min_access_count is not None and m.access_count < min_access_count:
                continue
            if max_access_count is not None and m.access_count > max_access_count:
                continue
            results.append(m)
            if len(results) >= limit:
                break
        return results


class SQLiteMemoryStore:
    """SQLite implementation for memory storage. Suitable for single-agent or development use.

    No vector search support (embeddings stored as JSON for reference, but not searchable).
    Uses aiosqlite for async database access.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """Args:
        db_path: Path to SQLite database file. Use ":memory:" for in-memory DB.
        """
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open database connection and create tables if needed."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self.create_tables()

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def create_tables(self) -> None:
        """Create the memories table if it doesn't exist."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'semi_stable',
                stability_score REAL NOT NULL DEFAULT 0.7,
                confidence REAL NOT NULL DEFAULT 0.7,
                confidence_source TEXT NOT NULL DEFAULT 'contextual_mention',
                dimensions TEXT NOT NULL DEFAULT '{}',
                scope TEXT NOT NULL DEFAULT 'private',
                scope_ref TEXT,
                source TEXT,
                extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER NOT NULL DEFAULT 0,
                conflicts_with TEXT,
                archived BOOLEAN NOT NULL DEFAULT 0,
                raw_text TEXT,
                embedding TEXT
            )
        """)

        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)")
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
        )
        await self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)")
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_scope_ref ON memories(scope_ref)"
        )
        await self._conn.commit()

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        """Save a memory. If ID exists, update it."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        data = memory.model_dump()
        embed_str = json.dumps(embedding) if embedding is not None else None

        await self._conn.execute(
            """
            INSERT INTO memories (id, key, value, memory_type, stability_score,
                confidence, confidence_source, dimensions, scope,
                scope_ref, source, extracted_at, last_accessed, access_count,
                conflicts_with, archived, raw_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                key = excluded.key, value = excluded.value,
                memory_type = excluded.memory_type, stability_score = excluded.stability_score,
                confidence = excluded.confidence, confidence_source = excluded.confidence_source,
                dimensions = excluded.dimensions, scope = excluded.scope,
                scope_ref = excluded.scope_ref, source = excluded.source,
                last_accessed = excluded.last_accessed,
                access_count = excluded.access_count, conflicts_with = excluded.conflicts_with,
                archived = excluded.archived, raw_text = excluded.raw_text,
                embedding = COALESCE(excluded.embedding, embedding)
            """,
            (
                data["id"],
                data["key"],
                data["value"],
                data["memory_type"],
                data["stability_score"],
                data["confidence"],
                data["confidence_source"],
                json.dumps(data["dimensions"]),
                data["scope"],
                data["scope_ref"],
                data["source"],
                data["extracted_at"].isoformat() if data["extracted_at"] else None,
                data["last_accessed"].isoformat() if data["last_accessed"] else None,
                data["access_count"],
                data["conflicts_with"],
                1 if data["archived"] else 0,
                data["raw_text"],
                embed_str,
            ),
        )
        await self._conn.commit()

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID. Returns None if not found."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        cursor = await self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    async def get_by_key(
        self,
        key: str,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[Memory]:
        """Get all memories with a given key. Sorted by extracted_at desc."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        query = "SELECT * FROM memories WHERE key = ?"
        params: list = [key]
        if not include_archived:
            query += " AND archived = 0"
        if scope_filter is not None:
            placeholders = ",".join("?" for _ in scope_filter)
            query += f" AND scope IN ({placeholders})"
            params.extend(scope_filter)
        if scope_refs is not None:
            placeholders = ",".join("?" for _ in scope_refs)
            query += f" AND (scope_ref IN ({placeholders}) OR scope_ref IS NULL)"
            params.extend(scope_refs)
        query += " ORDER BY extracted_at DESC"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def search_by_key_prefix(
        self, prefix: str, include_archived: bool = False
    ) -> list[Memory]:
        """Get all memories whose key starts with prefix."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        query = "SELECT * FROM memories WHERE key LIKE ?"
        params: list = [f"{prefix}%"]
        if not include_archived:
            query += " AND archived = 0"
        query += " ORDER BY extracted_at DESC"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory. Returns True if deleted, False if not found."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        cursor = await self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        result = cursor.rowcount > 0
        await self._conn.commit()
        return result

    async def archive(self, memory_id: str) -> bool:
        """Set archived=True on a memory. Returns True if found."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        cursor = await self._conn.execute(
            "UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,)
        )
        result = cursor.rowcount > 0
        await self._conn.commit()
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors using pure Python."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def search_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search by vector embedding.

        Brute-force cosine similarity in Python. Fine for hundreds to low
        thousands of memories; use PostgreSQL + pgvector for larger scales.
        """
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        query = "SELECT * FROM memories WHERE embedding IS NOT NULL"
        params: list = []
        if not include_archived:
            query += " AND archived = 0"
        if scope_filter is not None:
            placeholders = ",".join("?" for _ in scope_filter)
            query += f" AND scope IN ({placeholders})"
            params.extend(scope_filter)
        if scope_refs is not None:
            placeholders = ",".join("?" for _ in scope_refs)
            query += f" AND (scope_ref IN ({placeholders}) OR scope_ref IS NULL)"
            params.extend(scope_refs)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()

        scored: list[tuple[float, MemorySearchResult]] = []
        for r in rows:
            stored_embedding = json.loads(r["embedding"])
            similarity = self._cosine_similarity(embedding, stored_embedding)
            scored.append(
                (
                    similarity,
                    MemorySearchResult(
                        memory=self._row_to_memory(r),
                        relevance_score=max(0.0, similarity),
                        match_type="semantic",
                    ),
                )
            )

        scored.sort(key=lambda t: -t[0])
        return [result for _, result in scored[:top_k]]

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Keyword search across key and value fields."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        words = list({w.lower() for w in re.split(r"[\s,!?.;:'\"]+", query) if len(w) >= 3})
        if not words:
            words = [query.strip().lower()]

        conditions = " OR ".join("key LIKE ? OR value LIKE ?" for _ in words)
        params = [item for w in words for item in (f"%{w}%", f"%{w}%")]

        sql = f"SELECT * FROM memories WHERE ({conditions})"
        if not include_archived:
            sql += " AND archived = 0"
        if scope_filter is not None:
            placeholders = ",".join("?" for _ in scope_filter)
            sql += f" AND scope IN ({placeholders})"
            params.extend(scope_filter)
        if scope_refs is not None:
            placeholders = ",".join("?" for _ in scope_refs)
            sql += f" AND (scope_ref IN ({placeholders}) OR scope_ref IS NULL)"
            params.extend(scope_refs)
        sql += f" LIMIT {top_k}"

        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            MemorySearchResult(
                memory=self._row_to_memory(r),
                relevance_score=0.7,
                match_type="keyword",
            )
            for r in rows
        ]

    async def count(self, include_archived: bool = False) -> int:
        """Count total memories."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        query = "SELECT COUNT(*) FROM memories"
        if not include_archived:
            query += " WHERE archived = 0"

        cursor = await self._conn.execute(query)
        result = await cursor.fetchone()
        return result[0] if result else 0

    async def list_scope_refs(
        self,
        scope_filter: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[tuple[str, str | None]]:
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")
        query = "SELECT DISTINCT scope, scope_ref FROM memories"
        params: list = []
        conditions: list[str] = []
        if not include_archived:
            conditions.append("archived = 0")
        if scope_filter is not None:
            placeholders = ",".join("?" for _ in scope_filter)
            conditions.append(f"scope IN ({placeholders})")
            params.extend(scope_filter)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY scope, scope_ref"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [(r["scope"], r["scope_ref"]) for r in rows]

    async def list_top_keys(
        self,
        scope_ref: str,
        limit: int = 30,
    ) -> list[str]:
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        cursor = await self._conn.execute(
            """
            SELECT DISTINCT key, MAX(access_count) as max_access
            FROM memories
            WHERE scope_ref = ? AND archived = 0
            GROUP BY key
            ORDER BY max_access DESC, key ASC
            LIMIT ?
            """,
            (scope_ref, limit),
        )
        rows = await cursor.fetchall()
        return [r["key"] for r in rows]

    async def list_memories(
        self,
        *,
        memory_types: list[str] | None = None,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
        older_than: datetime | None = None,
        min_access_count: int | None = None,
        max_access_count: int | None = None,
        limit: int = 500,
    ) -> list[Memory]:
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        conditions: list[str] = []
        params: list = []

        if not include_archived:
            conditions.append("archived = 0")

        if memory_types is not None:
            if not memory_types:
                return []
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"memory_type IN ({placeholders})")
            params.extend(memory_types)

        if scope_filter is not None:
            if not scope_filter:
                return []
            placeholders = ",".join("?" for _ in scope_filter)
            conditions.append(f"scope IN ({placeholders})")
            params.extend(scope_filter)

        if scope_refs is not None:
            if not scope_refs:
                return []
            placeholders = ",".join("?" for _ in scope_refs)
            conditions.append(f"scope_ref IN ({placeholders})")
            params.extend(scope_refs)

        if older_than is not None:
            conditions.append("extracted_at < ?")
            params.append(older_than.isoformat())

        if min_access_count is not None:
            conditions.append("access_count >= ?")
            params.append(min_access_count)

        if max_access_count is not None:
            conditions.append("access_count <= ?")
            params.append(max_access_count)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM memories WHERE {where} LIMIT ?"
        params.append(limit)

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(r) for r in rows]

    @staticmethod
    def _row_to_memory(row) -> Memory:
        """Convert a database row to a Memory object."""
        data = {key: row[key] for key in row.keys()}
        if isinstance(data.get("dimensions"), str):
            data["dimensions"] = json.loads(data["dimensions"])
        data["archived"] = bool(data["archived"])
        data.pop("embedding", None)
        return Memory.model_validate(data)
