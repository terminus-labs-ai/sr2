"""Memory storage backends: protocol, in-memory (testing), PostgreSQL, and SQLite."""

import aiosqlite
import json
import re
from typing import Protocol

from sr2.memory.schema import Memory, MemorySearchResult


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
        query_lower = query.lower()
        results = []
        for m in self._memories.values():
            if m.archived and not include_archived:
                continue
            if not self._scope_match(m, scope_filter, scope_refs):
                continue
            if query_lower in m.key.lower() or query_lower in m.value.lower():
                results.append(
                    MemorySearchResult(memory=m, relevance_score=0.7, match_type="keyword")
                )
        results.sort(key=lambda r: r.memory.id)
        return results[:top_k]

    async def count(self, include_archived: bool = False) -> int:
        if include_archived:
            return len(self._memories)
        return sum(1 for m in self._memories.values() if not m.archived)


class PostgresMemoryStore:
    """PostgreSQL + pgvector implementation.

    Requires tables created via `create_tables()`.
    Uses asyncpg for async database access.
    """

    def __init__(self, pool) -> None:
        """Args:
        pool: asyncpg connection pool
        """
        self._pool = pool

    async def create_tables(self) -> None:
        """Create the memories table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'semi_stable',
                    stability_score REAL NOT NULL DEFAULT 0.7,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    confidence_source TEXT NOT NULL DEFAULT 'contextual_mention',
                    dimensions JSONB NOT NULL DEFAULT '{}',
                    scope TEXT NOT NULL DEFAULT 'private',
                    scope_ref TEXT,
                    source TEXT,
                    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    access_count INTEGER NOT NULL DEFAULT 0,
                    conflicts_with TEXT,
                    archived BOOLEAN NOT NULL DEFAULT FALSE,
                    raw_text TEXT,
                    embedding vector
                );
                CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
                CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
                CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
                CREATE INDEX IF NOT EXISTS idx_memories_scope_ref ON memories(scope_ref);
            """)
            # Migrate existing tables that were created with a fixed-dimension vector column
            # (e.g. vector(1536)) to unbounded vector so any embedding model can be used.
            # Try unconditionally — harmless if already unbounded.
            try:
                await conn.execute("ALTER TABLE memories ALTER COLUMN embedding TYPE vector")
            except Exception:
                pass

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        """Save a memory. If ID exists, update it."""
        data = memory.model_dump()
        embed_str = (
            "[" + ",".join(str(v) for v in embedding) + "]" if embedding is not None else None
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memories (id, key, value, memory_type, stability_score,
                    confidence, confidence_source, dimensions, scope,
                    scope_ref, source, extracted_at, last_accessed, access_count,
                    conflicts_with, archived, raw_text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18::vector)
                ON CONFLICT (id) DO UPDATE SET
                    key = EXCLUDED.key, value = EXCLUDED.value,
                    memory_type = EXCLUDED.memory_type, stability_score = EXCLUDED.stability_score,
                    confidence = EXCLUDED.confidence, confidence_source = EXCLUDED.confidence_source,
                    dimensions = EXCLUDED.dimensions, scope = EXCLUDED.scope,
                    scope_ref = EXCLUDED.scope_ref, source = EXCLUDED.source,
                    last_accessed = EXCLUDED.last_accessed,
                    access_count = EXCLUDED.access_count, conflicts_with = EXCLUDED.conflicts_with,
                    archived = EXCLUDED.archived, raw_text = EXCLUDED.raw_text,
                    embedding = COALESCE(EXCLUDED.embedding, memories.embedding)
                """,
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
                data["extracted_at"],
                data["last_accessed"],
                data["access_count"],
                data["conflicts_with"],
                data["archived"],
                data["raw_text"],
                embed_str,
            )

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1", memory_id)
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
        query = "SELECT * FROM memories WHERE key = $1"
        params: list = [key]
        if not include_archived:
            query += " AND archived = false"
        if scope_filter is not None:
            params.append(scope_filter)
            query += f" AND scope = ANY(${len(params)})"
        if scope_refs is not None:
            params.append(scope_refs)
            query += f" AND (scope_ref = ANY(${len(params)}) OR scope_ref IS NULL)"
        query += " ORDER BY extracted_at DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_memory(r) for r in rows]

    async def search_by_key_prefix(
        self, prefix: str, include_archived: bool = False
    ) -> list[Memory]:
        """Get all memories whose key starts with prefix."""
        query = "SELECT * FROM memories WHERE key LIKE $1 ESCAPE '\\'"
        if not include_archived:
            query += " AND archived = false"
        query += " ORDER BY extracted_at DESC"
        async with self._pool.acquire() as conn:
            # Escape special LIKE characters in the prefix
            escaped_prefix = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = await conn.fetch(query, f"{escaped_prefix}%")
            return [self._row_to_memory(r) for r in rows]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)
            return result == "DELETE 1"

    async def archive(self, memory_id: str) -> bool:
        """Set archived=True on a memory."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE memories SET archived = true WHERE id = $1", memory_id
            )
            return result == "UPDATE 1"

    async def search_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search by vector embedding."""
        embed_str = "[" + ",".join(str(v) for v in embedding) + "]"
        conditions = ["embedding IS NOT NULL"]
        params: list = [embed_str, top_k]
        if not include_archived:
            conditions.append("archived = false")
        if scope_filter is not None:
            params.append(scope_filter)
            conditions.append(f"scope = ANY(${len(params)})")
        if scope_refs is not None:
            params.append(scope_refs)
            conditions.append(f"(scope_ref = ANY(${len(params)}) OR scope_ref IS NULL)")
        where = " AND ".join(conditions)
        query = f"""
            SELECT *, 1 - (embedding <=> $1::vector) as similarity
            FROM memories WHERE {where}
            ORDER BY embedding <=> $1::vector LIMIT $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                MemorySearchResult(
                    memory=self._row_to_memory(r),
                    relevance_score=float(r["similarity"] or 0.0),
                    match_type="semantic",
                )
                for r in rows
            ]

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Keyword search across key and value fields.

        Splits the query into individual words so that conversational messages
        (e.g. "Hey Miranda, what are my tasks?") can match memory values that
        contain any of the meaningful words rather than requiring the entire
        message to appear as a substring.
        """
        words = list({w.lower() for w in re.split(r"[\s,!?.;:'\"]+", query) if len(w) >= 3})
        if not words:
            words = [query.strip().lower()]

        word_patterns = [f"%{w}%" for w in words]
        conditions = " OR ".join(
            f"(key ILIKE ${i + 1} OR value ILIKE ${i + 1})" for i in range(len(words))
        )
        sql = f"SELECT * FROM memories WHERE ({conditions})"
        extra_params: list = []
        if not include_archived:
            sql += " AND archived = false"
        if scope_filter is not None:
            extra_params.append(scope_filter)
            sql += f" AND scope = ANY(${len(words) + 1 + len(extra_params)})"
        if scope_refs is not None:
            extra_params.append(scope_refs)
            sql += f" AND (scope_ref = ANY(${len(words) + 1 + len(extra_params)}) OR scope_ref IS NULL)"
        sql += f" ORDER BY id LIMIT ${len(words) + 1}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *word_patterns, top_k, *extra_params)
            return [
                MemorySearchResult(
                    memory=self._row_to_memory(r),
                    relevance_score=0.7,
                    match_type="keyword",
                )
                for r in rows
            ]

    async def list_without_embeddings(self, include_archived: bool = False) -> list[Memory]:
        """Return all memories that have no embedding stored."""
        sql = "SELECT * FROM memories WHERE embedding IS NULL"
        if not include_archived:
            sql += " AND archived = false"
        sql += " ORDER BY extracted_at ASC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
            return [self._row_to_memory(r) for r in rows]

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> bool:
        """Store an embedding for an existing memory. Returns True if the row existed."""
        embed_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE memories SET embedding = $1::vector WHERE id = $2",
                embed_str,
                memory_id,
            )
            return result == "UPDATE 1"

    async def count(self, include_archived: bool = False) -> int:
        """Count total memories."""
        query = "SELECT COUNT(*) FROM memories"
        if not include_archived:
            query += " WHERE archived = false"
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query)

    @staticmethod
    def _row_to_memory(row) -> Memory:
        """Convert a database row to a Memory object."""
        data = dict(row)
        if isinstance(data.get("dimensions"), str):
            data["dimensions"] = json.loads(data["dimensions"])
        data.pop("embedding", None)
        data.pop("similarity", None)
        return Memory.model_validate(data)


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

        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_scope_ref ON memories(scope_ref)"
        )
        await self._conn.commit()

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        """Save a memory. If ID exists, update it."""
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        data = memory.model_dump()
        embed_str = (
            json.dumps(embedding) if embedding is not None else None
        )

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

        cursor = await self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        )
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

    async def search_vector(
        self,
        embedding: list[float],
        top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search by vector embedding.

        SQLite has no native vector distance support. Returns up to top_k memories
        with embeddings in order of storage (oldest first). Use PostgreSQL backend
        for real semantic search.
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
        query += f" LIMIT {top_k}"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            MemorySearchResult(
                memory=self._row_to_memory(r),
                relevance_score=0.5,
                match_type="semantic",
            )
            for r in rows
        ]

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

    @staticmethod
    def _row_to_memory(row) -> Memory:
        """Convert a database row to a Memory object."""
        data = {key: row[key] for key in row.keys()}
        if isinstance(data.get("dimensions"), str):
            data["dimensions"] = json.loads(data["dimensions"])
        data["archived"] = bool(data["archived"])
        data.pop("embedding", None)
        return Memory.model_validate(data)
