"""CostGate: decides whether compacting a context is worth the cost."""

from __future__ import annotations


class CostGate:
    """Evaluate whether compaction yields a net benefit.

    Parameters
    ----------
    cost_multiplier:
        Scale factor applied to ``cache_cost`` before comparison.  A
        multiplier > 1 raises the bar; < 1 lowers it.  Defaults to 1.0.
    """

    def __init__(self, cost_multiplier: float = 1.0) -> None:
        self._cost_multiplier = cost_multiplier

    def should_compact(self, token_savings: int, cache_cost: int) -> bool:
        """Return True iff token_savings strictly exceed the effective cache cost.

        Parameters
        ----------
        token_savings:
            Estimated tokens saved by compaction (positive means savings).
        cache_cost:
            Tokens / cost units consumed by invalidating the cache due to
            compaction.
        """
        if token_savings <= 0:
            return False

        effective_cost = cache_cost * self._cost_multiplier
        return token_savings > effective_cost
