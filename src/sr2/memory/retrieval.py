"""Hybrid retrieval engine combining semantic, keyword, and recency signals."""

import math
import time
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
        # Instrumentation stats (updated on each retrieve() call)
        self.last_latency_ms: float = 0.0
        self.last_avg_precision: float = 0.0
        self.last_was_empty: bool = False
        self._total_retrievals: int = 0
        self._empty_retrievals: int = 0

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
        t0 = time.perf_counter()
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

        # Touch accessed memories and persist updates
        for r in results:
            r.memory.touch()
            await self._store.save(r.memory)

        # Update instrumentation stats
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_was_empty = len(results) == 0
        self.last_avg_precision = (
            sum(r.relevance_score for r in results) / len(results) if results else 0.0
        )
        self._total_retrievals += 1
        if self.last_was_empty:
            self._empty_retrievals += 1

        return results

    @property
    def empty_rate(self) -> float:
        """Fraction of retrievals that returned no results."""
        if self._total_retrievals == 0:
            return 0.0
        return self._empty_retrievals / self._total_retrievals

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
            r.relevance_score = min(1.0, max(0.0,
                base * (1 - self._weights["recency"] - self._weights["frequency"])
                + recency_score * self._weights["recency"]
                + frequency_score * self._weights["frequency"]
            ))
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
