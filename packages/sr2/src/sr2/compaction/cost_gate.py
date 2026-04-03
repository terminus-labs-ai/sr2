"""Cache-cost-aware compaction gate.

Decides whether compacting a turn is net-positive after accounting for
prompt caching economics. Compaction saves input tokens but invalidates
cached prefixes, forcing expensive cache writes.
"""

import logging
from dataclasses import dataclass

from sr2.compaction.pricing import CachePricing, resolve_pricing
from sr2.config.models import CostGateConfig

logger = logging.getLogger(__name__)


@dataclass
class CompactionCandidate:
    """A turn being evaluated for compaction."""

    turn_index: int
    turn_tokens: int
    estimated_compacted_tokens: int
    total_tokens_after_turn: int


@dataclass
class CostGateDecision:
    """Result of evaluating whether a turn should be compacted."""

    allowed: bool
    tokens_saved: int  # turn_tokens - estimated_compacted_tokens
    estimated_savings_usd: float  # tokens_saved * input_cost_per_token
    cache_invalidation_tokens: int  # total_tokens_after_turn
    estimated_invalidation_cost_usd: float  # invalidation_tokens * (write - read)
    net_usd: float  # savings - invalidation_cost
    reason: str
    pricing_source: str


class CompactionCostGate:
    """Decides whether compacting a turn is cost-positive after cache economics."""

    def __init__(self, config: CostGateConfig):
        self.config = config
        self._pricing_cache: dict[str, CachePricing] = {}

    def _get_pricing(self, model_hint: str | None) -> CachePricing:
        """Resolve and cache pricing for the given model hint."""
        cache_key = model_hint or "__none__"
        if cache_key not in self._pricing_cache:
            self._pricing_cache[cache_key] = resolve_pricing(
                model_hint=model_hint,
                fallback_model=self.config.fallback_model,
                custom_pricing=self.config.custom_pricing,
            )
        return self._pricing_cache[cache_key]

    def should_compact(
        self,
        turn_index: int,
        turn_tokens: int,
        estimated_compacted_tokens: int,
        total_tokens_after_turn: int,
        model_hint: str | None = None,
    ) -> CostGateDecision:
        """Evaluate whether compacting this turn saves money.

        Args:
            turn_index: Position of this turn in the message array (0-based).
            turn_tokens: Current token count of the turn (before compaction).
            estimated_compacted_tokens: Estimated tokens after compaction.
            total_tokens_after_turn: Total tokens in all messages AFTER this turn.
                This is the cache that would be invalidated.
            model_hint: Optional model name to resolve pricing.

        Returns:
            CostGateDecision with allow/block and the cost math.
        """
        pricing = self._get_pricing(model_hint)

        # Fail-open: if no pricing data, always allow
        if pricing.source == "fail_open":
            return CostGateDecision(
                allowed=True,
                tokens_saved=turn_tokens - estimated_compacted_tokens,
                estimated_savings_usd=0.0,
                cache_invalidation_tokens=total_tokens_after_turn,
                estimated_invalidation_cost_usd=0.0,
                net_usd=0.0,
                reason="no pricing data available, failing open",
                pricing_source=pricing.source,
            )

        token_savings = turn_tokens - estimated_compacted_tokens
        savings_usd = token_savings * pricing.input_cost

        # Cache invalidation: downstream changes from cache_read to cache_write
        invalidation_cost_usd = total_tokens_after_turn * (
            pricing.cache_write_cost - pricing.cache_read_cost
        )

        net = savings_usd - invalidation_cost_usd
        allowed = net > self.config.min_net_savings_usd

        if allowed:
            reason = (
                f"save {token_savings:,} tokens (${savings_usd:.4f}), "
                f"invalidate {total_tokens_after_turn:,} tokens (${invalidation_cost_usd:.4f}), "
                f"net +${net:.4f}"
            )
        else:
            reason = (
                f"save {token_savings:,} tokens (${savings_usd:.4f}), "
                f"invalidate {total_tokens_after_turn:,} tokens (${invalidation_cost_usd:.4f}), "
                f"net -${abs(net):.4f}"
            )

        decision = CostGateDecision(
            allowed=allowed,
            tokens_saved=token_savings,
            estimated_savings_usd=savings_usd,
            cache_invalidation_tokens=total_tokens_after_turn,
            estimated_invalidation_cost_usd=invalidation_cost_usd,
            net_usd=net,
            reason=reason,
            pricing_source=pricing.source,
        )

        log_status = "ALLOWED" if allowed else "BLOCKED"
        logger.info(
            "Turn %d: %s — %s [pricing: %s]",
            turn_index,
            log_status,
            reason,
            pricing.source,
        )
        logger.debug("Turn %d: full decision: %s", turn_index, decision)

        return decision

    def evaluate_batch(
        self,
        candidates: list[CompactionCandidate],
        model_hint: str | None = None,
    ) -> list[CostGateDecision]:
        """Evaluate multiple turns for compaction, earliest to latest.

        Accounts for cascading invalidation — if turn N is compacted,
        turn N+1's invalidation cost is partially pre-paid.
        """
        # Sort by turn_index to process earliest first
        sorted_candidates = sorted(candidates, key=lambda c: c.turn_index)
        decisions: list[CostGateDecision] = []
        already_invalidated: int = 0

        for candidate in sorted_candidates:
            # Reduce effective invalidation by what's already been invalidated
            effective_invalidation = max(
                0, candidate.total_tokens_after_turn - already_invalidated
            )

            decision = self.should_compact(
                turn_index=candidate.turn_index,
                turn_tokens=candidate.turn_tokens,
                estimated_compacted_tokens=candidate.estimated_compacted_tokens,
                total_tokens_after_turn=effective_invalidation,
                model_hint=model_hint,
            )
            decisions.append(decision)

            if decision.allowed:
                # This turn's downstream is now "pre-paid" for subsequent turns
                already_invalidated = candidate.total_tokens_after_turn

        return decisions
