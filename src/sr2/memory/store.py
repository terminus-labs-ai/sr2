"""Memory storage backends: protocol, in-memory (testing), and PostgreSQL."""

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

    async def get_by_key(self, key: str, include_archived: bool = False) -> list[Memory]:
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
    ) -> list[MemorySearchResult]:
        """Semantic search by vector embedding. Returns top_k results sorted by similarity."""
        ...

    async def search_keyword(
        self,
        query: str,
        top_k: int = 10,
        include_archived: bool = False,
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

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        self._memories[memory.id] = memory

    async def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    async def get_by_key(self, key: str, include_archived: bool = False) -> list[Memory]:
        results = [m for m in self._memories.values() if m.key == key]
        if not include_archived:
            results = [m for m in results if not m.archived]
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
    ) -> list[MemorySearchResult]:
        mems = [m for m in self._memories.values() if not m.archived or include_archived]
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
    ) -> list[MemorySearchResult]:
        query_lower = query.lower()
        results = []
        for m in self._memories.values():
            if m.archived and not include_archived:
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
                    source_conversation TEXT,
                    source_turn INTEGER,
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
                    confidence, confidence_source, dimensions, source_conversation,
                    source_turn, extracted_at, last_accessed, access_count,
                    conflicts_with, archived, raw_text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17::vector)
                ON CONFLICT (id) DO UPDATE SET
                    key = EXCLUDED.key, value = EXCLUDED.value,
                    memory_type = EXCLUDED.memory_type, stability_score = EXCLUDED.stability_score,
                    confidence = EXCLUDED.confidence, confidence_source = EXCLUDED.confidence_source,
                    dimensions = EXCLUDED.dimensions, source_conversation = EXCLUDED.source_conversation,
                    source_turn = EXCLUDED.source_turn, last_accessed = EXCLUDED.last_accessed,
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
                data["source_conversation"],
                data["source_turn"],
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

    async def get_by_key(self, key: str, include_archived: bool = False) -> list[Memory]:
        """Get all memories with a given key. Sorted by extracted_at desc."""
        query = "SELECT * FROM memories WHERE key = $1"
        if not include_archived:
            query += " AND archived = false"
        query += " ORDER BY extracted_at DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, key)
            return [self._row_to_memory(r) for r in rows]

    async def search_by_key_prefix(
        self, prefix: str, include_archived: bool = False
    ) -> list[Memory]:
        """Get all memories whose key starts with prefix."""
        query = "SELECT * FROM memories WHERE key LIKE $1"
        if not include_archived:
            query += " AND archived = false"
        query += " ORDER BY extracted_at DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, f"{prefix}%")
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
    ) -> list[MemorySearchResult]:
        """Semantic search by vector embedding."""
        embed_str = "[" + ",".join(str(v) for v in embedding) + "]"
        if include_archived:
            query = """
                SELECT *, 1 - (embedding <=> $1::vector) as similarity
                FROM memories WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector LIMIT $2
            """
            params = [embed_str, top_k]
        else:
            query = """
                SELECT *, 1 - (embedding <=> $1::vector) as similarity
                FROM memories WHERE archived = false AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector LIMIT $2
            """
            params = [embed_str, top_k]
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
        if not include_archived:
            sql += " AND archived = false"
        sql += f" ORDER BY id LIMIT ${len(words) + 1}"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *word_patterns, top_k)
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
