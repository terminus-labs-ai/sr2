"""Conflict detection between new and existing memories."""

from dataclasses import dataclass

from sr2.memory.schema import Memory
from sr2.memory.store import MemoryStore


@dataclass
class Conflict:
    """A detected conflict between two memories."""

    new_memory: Memory
    existing_memory: Memory
    conflict_type: str  # "key_match", "semantic", "entity_temporal"
    confidence: float  # How confident we are this is a real conflict (0-1)


class ConflictDetector:
    """Detects conflicts between new and existing memories."""

    def __init__(
        self,
        store: MemoryStore,
        semantic_threshold: float = 0.85,
        use_semantic: bool = False,
        embedding_callable=None,
    ):
        """Args:
            store: Memory store to check against.
            semantic_threshold: Similarity threshold for semantic conflict detection.
            use_semantic: Whether to use semantic similarity (requires embedding_callable).
            embedding_callable: async function(text: str) -> list[float]. Optional.
        """
        self._store = store
        self._threshold = semantic_threshold
        self._use_semantic = use_semantic
        self._embed = embedding_callable

    async def detect(self, new_memory: Memory) -> list[Conflict]:
        """Detect conflicts for a new memory.

        Detection order:
        1. Key-based: exact key match with different value
        2. Semantic (optional): high similarity + different value

        Returns list of Conflict objects (may be empty).
        """
        conflicts = []

        # 1. Key-based detection
        key_conflicts = await self._detect_key_conflicts(new_memory)
        conflicts.extend(key_conflicts)

        # 2. Semantic detection (optional, if enabled)
        if self._use_semantic and self._embed and not key_conflicts:
            semantic_conflicts = await self._detect_semantic_conflicts(new_memory)
            conflicts.extend(semantic_conflicts)

        return conflicts

    async def _detect_key_conflicts(self, new: Memory) -> list[Conflict]:
        """Find existing memories with the same key but different value."""
        existing = await self._store.get_by_key(new.key, include_archived=False)
        conflicts = []
        for mem in existing:
            if mem.id == new.id:
                continue  # Don't conflict with self
            if mem.value.strip().lower() != new.value.strip().lower():
                conflicts.append(
                    Conflict(
                        new_memory=new,
                        existing_memory=mem,
                        conflict_type="key_match",
                        confidence=1.0,
                    )
                )
        return conflicts

    async def _detect_semantic_conflicts(self, new: Memory) -> list[Conflict]:
        """Find semantically similar memories with different values."""
        if not self._embed:
            return []

        embedding = await self._embed(f"{new.key}: {new.value}")
        results = await self._store.search_vector(embedding, top_k=5)

        conflicts = []
        for result in results:
            mem = result.memory
            if mem.id == new.id:
                continue
            if result.relevance_score >= self._threshold:
                if mem.value.strip().lower() != new.value.strip().lower():
                    conflicts.append(
                        Conflict(
                            new_memory=new,
                            existing_memory=mem,
                            conflict_type="semantic",
                            confidence=result.relevance_score,
                        )
                    )
        return conflicts
