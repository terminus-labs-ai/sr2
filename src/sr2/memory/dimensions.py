"""Dimensional memory matching — filter and rerank by context dimensions."""

from sr2.memory.schema import MemorySearchResult


class DimensionalMatcher:
    """Filters and reranks memories based on context dimensions."""

    def __init__(
        self,
        matching_strategy: str = "best_fit",
        known_dimensions: list[dict] | None = None,
    ):
        """Args:
            matching_strategy: "best_fit" | "exact" | "fallback_to_generic"
            known_dimensions: list of {"name": str, "values": list[str] | "dynamic"}
        """
        self._strategy = matching_strategy
        self._known = known_dimensions or []

    def filter(
        self,
        results: list[MemorySearchResult],
        current_dimensions: dict[str, str],
    ) -> list[MemorySearchResult]:
        """Filter and rerank results based on dimensional matching.

        best_fit: Prefer dimension-matched, keep unscoped as fallback.
                  Boost score for dimension matches.
        exact: Only return memories that match ALL current dimensions.
        fallback_to_generic: Try exact first, if no results, return unscoped.
        """
        if not current_dimensions:
            return results

        if self._strategy == "exact":
            return self._exact_match(results, current_dimensions)
        elif self._strategy == "fallback_to_generic":
            exact = self._exact_match(results, current_dimensions)
            if exact:
                return exact
            return self._unscoped_only(results)
        else:  # best_fit
            return self._best_fit(results, current_dimensions)

    def _best_fit(
        self,
        results: list[MemorySearchResult],
        dims: dict[str, str],
    ) -> list[MemorySearchResult]:
        """Score memories by dimensional overlap. Keep all, rerank."""
        scored = []
        for r in results:
            mem_dims = r.memory.dimensions
            if not mem_dims:
                # Unscoped memory — keep with small penalty
                r.relevance_score *= 0.9
                scored.append(r)
                continue

            # Count matching dimensions
            matches = sum(1 for k, v in dims.items() if mem_dims.get(k) == v)
            mismatches = sum(
                1 for k, v in dims.items() if k in mem_dims and mem_dims[k] != v
            )

            if mismatches > 0:
                # Dimension mismatch — penalize heavily
                r.relevance_score *= 0.3
            elif matches > 0:
                # Dimension match — boost
                r.relevance_score *= 1.0 + 0.2 * matches

            scored.append(r)

        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        return scored

    def _exact_match(
        self,
        results: list[MemorySearchResult],
        dims: dict[str, str],
    ) -> list[MemorySearchResult]:
        """Only return memories that match all specified dimensions."""
        return [
            r
            for r in results
            if all(r.memory.dimensions.get(k) == v for k, v in dims.items())
        ]

    def _unscoped_only(
        self,
        results: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        """Return only memories with no dimensions set."""
        return [r for r in results if not r.memory.dimensions]
