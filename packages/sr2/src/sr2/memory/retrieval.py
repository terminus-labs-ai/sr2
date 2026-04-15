"""Hybrid retrieval engine combining semantic, keyword, and recency signals."""

import logging
import math
import time
from datetime import UTC, datetime

from sr2.config.models import MemoryScopeConfig
from sr2.memory.schema import MemorySearchResult
from sr2.memory.store import MemoryStore

logger = logging.getLogger(__name__)


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
        scope_config: MemoryScopeConfig | None = None,
        current_context: dict | None = None,
        trace_collector=None,
    ):
        self._store = store
        self._embed = embedding_callable
        self._strategy = strategy
        self._scope_config = scope_config
        self._current_context = current_context
        self._top_k = top_k
        self._recency_decay = recency_decay_days
        self._trace = trace_collector
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
        # IDs of memories accessed in the last retrieve() call, for deferred touch
        self._pending_touch_ids: list[str] = []

    def _build_scope_params(self) -> tuple[list[str] | None, list[str] | None]:
        """Build scope_filter and scope_refs from config and context."""
        if not self._scope_config:
            return None, None

        scope_filter = list(self._scope_config.allowed_read)
        scope_refs: list[str] = []

        if "private" in scope_filter and self._scope_config.agent_name:
            scope_refs.append(f"agent:{self._scope_config.agent_name}")
        if "project" in scope_filter:
            project_id = (self._current_context or {}).get("project_id")
            if project_id:
                scope_refs.append(project_id)

        return scope_filter, scope_refs if scope_refs else None

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

        scope_filter, scope_refs = self._build_scope_params()

        # Semantic search
        if self._strategy in ("hybrid", "semantic") and self._embed:
            embedding = await self._embed(query)
            semantic_results = await self._store.search_vector(
                embedding,
                top_k=k * 2,
                scope_filter=scope_filter,
                scope_refs=scope_refs,
            )
            for r in semantic_results:
                candidates[r.memory.id] = r

        # Keyword search
        if self._strategy in ("hybrid", "keyword"):
            keyword_results = await self._store.search_keyword(
                query,
                top_k=k * 2,
                scope_filter=scope_filter,
                scope_refs=scope_refs,
            )
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

        # Record IDs for deferred touch (called after pipeline is stable)
        self._pending_touch_ids = [r.memory.id for r in results]

        # Update instrumentation stats
        self.last_latency_ms = (time.perf_counter() - t0) * 1000
        self.last_was_empty = len(results) == 0
        self.last_avg_precision = (
            sum(r.relevance_score for r in results) / len(results) if results else 0.0
        )
        self._total_retrievals += 1
        if self.last_was_empty:
            self._empty_retrievals += 1

        # Emit trace event with all scored candidates
        if self._trace:
            self._trace.emit("retrieve", {
                "query": query,
                "strategy": self._strategy,
                "candidates_scored": len(scored),
                "results_returned": len(results),
                "top_k": k,
                "threshold": 0.0,
                "results": [
                    {
                        "key": r.memory.key,
                        "value_preview": r.memory.value[:100],
                        "relevance_score": r.relevance_score,
                        "match_type": r.match_type,
                        "selected": r in results,
                        "memory_type": r.memory.memory_type,
                        "scope": r.memory.scope,
                    }
                    for r in scored
                ],
                "latency_ms": self.last_latency_ms,
            }, duration_ms=self.last_latency_ms)

        return results

    async def flush_touches(self) -> None:
        """Persist deferred touch() calls for memories accessed in the last retrieve().

        Should be called during post-LLM processing, after the pipeline has
        finished compiling context, so that touch side-effects don't destabilise
        layer content between compile() calls.
        """
        if not self._pending_touch_ids:
            return
        ids = self._pending_touch_ids
        self._pending_touch_ids = []
        for memory_id in ids:
            mem = await self._store.get(memory_id)
            if mem is not None:
                mem.touch()
                await self._store.save(mem)

    @property
    def empty_rate(self) -> float:
        """Fraction of retrievals that returned no results."""
        if self._total_retrievals == 0:
            return 0.0
        return self._empty_retrievals / self._total_retrievals

    def update_context(self, current_context: dict | None) -> None:
        """Update current_context for the next retrieve() call.

        Prefer this over direct attribute assignment for clarity and
        future hook points (e.g., scope detection invalidation).
        """
        self._current_context = current_context

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
            r.relevance_score = min(
                1.0,
                max(
                    0.0,
                    base * (1 - self._weights["recency"] - self._weights["frequency"])
                    + recency_score * self._weights["recency"]
                    + frequency_score * self._weights["frequency"],
                ),
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
        dropped = len(results) - len(capped)
        if dropped:
            logger.warning(
                "Retrieval token cap: kept %d/%d results (%d tokens), %d results dropped",
                len(capped),
                len(results),
                total,
                dropped,
            )
        return capped
