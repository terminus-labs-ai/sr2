"""In-memory MemoryStore implementation.

Dict-backed store for Spectre MVP — no persistence, no embeddings.
Search uses keyword matching on content + tags, ranked by frequency × recency.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .protocol import MemoryStore
from .schema import Memory, MemoryScope, MemorySearchResult


class InMemoryMemoryStore(MemoryStore):
    """Dict-backed memory store for development and testing."""

    def __init__(self) -> None:
        self._store: dict[str, Memory] = {}

    def save(self, memory: Memory) -> Memory:
        """Persist a Memory. If the id already exists, increment frequency.

        Never mutates the caller's object — returns a new Memory with updated fields.
        """
        existing = self._store.get(memory.id)
        updates: dict = {"last_accessed": datetime.now(timezone.utc)}
        if existing:
            updates["frequency"] = existing.frequency + 1
        saved = memory.model_copy(update=updates)
        self._store[saved.id] = saved
        return saved

    def search(
        self, query: str, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemorySearchResult]:
        """Keyword match on content and tags, ranked by frequency × recency."""
        if not query:
            return []

        query_lower = query.lower()
        candidates: list[tuple[float, Memory]] = []

        for m in self._store.values():
            if scope is not None and m.scope != scope:
                continue

            if query_lower in m.content.lower() or any(
                query_lower in tag.lower() for tag in m.tags
            ):
                score = self._rank_score(m)
                candidates.append((score, m))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            MemorySearchResult(
                id=m.id,
                content=m.content,
                score=score,
                scope=m.scope,
                tags=m.tags,
            )
            for score, m in candidates[:limit]
        ]

    def get_by_tag(
        self, tag: str, scope: MemoryScope | None = None, limit: int = 10
    ) -> list[MemorySearchResult]:
        """Filter by tag, optionally scoped, ranked by frequency × recency."""
        tag_lower = tag.lower()
        candidates: list[tuple[float, Memory]] = []

        for m in self._store.values():
            if scope is not None and m.scope != scope:
                continue
            if any(tag_lower == t.lower() for t in m.tags):
                score = self._rank_score(m)
                candidates.append((score, m))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [
            MemorySearchResult(
                id=m.id,
                content=m.content,
                score=score,
                scope=m.scope,
                tags=m.tags,
            )
            for score, m in candidates[:limit]
        ]

    def delete(self, memory_id: str) -> bool:
        """Remove by id. Returns True if found and deleted."""
        return self._store.pop(memory_id, None) is not None

    def get_all(self, scope: MemoryScope | None = None) -> list[Memory]:
        """Return all memories, optionally filtered by scope."""
        if scope is None:
            return list(self._store.values())
        return [m for m in self._store.values() if m.scope == scope]

    @staticmethod
    def _rank_score(m: Memory) -> float:
        """Rank by frequency × recency. More frequent and more recently accessed = higher."""
        freq = max(m.frequency, 1)  # At least 1 so new memories aren't zero-scored
        recency = 1.0
        if m.last_accessed:
            delta = datetime.now(timezone.utc) - m.last_accessed
            days = max(delta.total_seconds() / 86400, 0.01)
            recency = 1.0 / (1.0 + days)  # Decays over time
        return freq * recency
