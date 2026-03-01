"""Hybrid retrieval engine combining semantic, keyword, and recency signals."""

import math
from datetime import UTC, datetime

from sr2.memory.schema import MemorySearchResult
from sr2.memory.store import MemoryStore


class HybridRetriever:
    """Combines semantic, keyword, and recency signals for memory retrieval."""

    def __init__(
        self,
        store: MemoryStore,
        embedding_callable=None,
        strategy: str = "hybrid",
        top_k: int = 10,
        recency_decay_days: float = 30.0,
        frequency_weight: float = 0.1,
        recency_weight: float = 0.2,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.2,
    ):
        self._store = store
        self._embed = embedding_callable
        self._strategy = strategy
        self._top_k = top_k
        self._recency_decay = recency_decay_days
        self._weights = {
            "semantic": semantic_weight,
            "keyword": keyword_weight,
            "recency": recency_weight,
            "frequency": frequency_weight,
        }

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> list[MemorySearchResult]:
        """Retrieve relevant memories using the configured strategy.

        1. Run enabled search methods (semantic, keyword)
        2. Merge results, deduplicating by memory ID
        3. Apply recency and frequency boosts
        4. Sort by final score
        5. Return top_k results (optionally capped by token count)
        """
        k = top_k or self._top_k
        candidates: dict[str, MemorySearchResult] = {}

        # Semantic search
        if self._strategy in ("hybrid", "semantic") and self._embed:
            embedding = await self._embed(query)
            semantic_results = await self._store.search_vector(embedding, top_k=k * 2)
            for r in semantic_results:
                candidates[r.memory.id] = r

        # Keyword search
        if self._strategy in ("hybrid", "keyword"):
            keyword_results = await self._store.search_keyword(query, top_k=k * 2)
            for r in keyword_results:
                if r.memory.id in candidates:
                    existing = candidates[r.memory.id]
                    existing.relevance_score = max(existing.relevance_score, r.relevance_score)
                else:
                    candidates[r.memory.id] = r

        # Apply recency and frequency boosts
        scored = self._apply_boosts(list(candidates.values()))

        # Sort by final score descending
        scored.sort(key=lambda r: (-r.relevance_score, r.memory.id))

        # Cap by top_k
        results = scored[:k]

        # Cap by token count if specified
        if max_tokens is not None:
            results = self._cap_by_tokens(results, max_tokens)

        # Touch accessed memories
        for r in results:
            r.memory.touch()

        return results

    def _apply_boosts(self, results: list[MemorySearchResult]) -> list[MemorySearchResult]:
        """Apply recency and frequency boosts to relevance scores."""
        now = datetime.now(UTC)
        for r in results:
            mem = r.memory
            # Recency boost: exponential decay
            age_days = (now - mem.last_accessed).total_seconds() / 86400
            recency_score = math.exp(-age_days / self._recency_decay)

            # Frequency boost: log scale
            frequency_score = min(1.0, math.log1p(mem.access_count) / 5.0)

            # Weighted combination
            base = r.relevance_score
            r.relevance_score = (
                base * (1 - self._weights["recency"] - self._weights["frequency"])
                + recency_score * self._weights["recency"]
                + frequency_score * self._weights["frequency"]
            )
        return results

    def _cap_by_tokens(
        self,
        results: list[MemorySearchResult],
        max_tokens: int,
    ) -> list[MemorySearchResult]:
        """Keep results until token budget is exhausted.
        Rough estimation: 1 token ~ 4 characters."""
        capped = []
        total = 0
        for r in results:
            est_tokens = len(f"{r.memory.key}: {r.memory.value}") // 4
            if total + est_tokens > max_tokens:
                break
            capped.append(r)
            total += est_tokens
        return capped
