"""Cache-cost-aware compaction gate.

.. deprecated::
    Use :class:`sr2.compaction.budget_optimizer.BudgetOptimizer` instead.
    CompactionCostGate evaluates turns individually and does not account for
    budget pressure or correct input token economics. It is retained for
    backward compatibility when ``budget_optimizer.enabled`` is False.
"""

import logging
import warnings
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
        warnings.warn(
            "CompactionCostGate is deprecated. Use BudgetOptimizer "
            "(compaction.budget_optimizer.enabled: true) for budget-aware "
            "compaction decisions.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        # Fail-closed: if cost gate is enabled but pricing can't be resolved,
        # block compaction to prevent cost blowup. If cost gate is disabled
        # (i.e. CompactionCostGate was instantiated but not gating), fail open.
        if pricing.source == "fail_open":
            if self.config.enabled:
                logger.error(
                    "Cost gate enabled but no pricing data (model_hint=%r, fallback_model=%r). "
                    "Blocking compaction to prevent cost blowup. "
                    "Set custom_pricing or fallback_model in cost_gate config.",
                    model_hint,
                    self.config.fallback_model,
                )
                return CostGateDecision(
                    allowed=False,
                    tokens_saved=turn_tokens - estimated_compacted_tokens,
                    estimated_savings_usd=0.0,
                    cache_invalidation_tokens=total_tokens_after_turn,
                    estimated_invalidation_cost_usd=0.0,
                    net_usd=0.0,
                    reason="cost gate enabled but no pricing data — blocking to prevent cost blowup",
                    pricing_source=pricing.source,
                )
            else:
                return CostGateDecision(
                    allowed=True,
                    tokens_saved=turn_tokens - estimated_compacted_tokens,
                    estimated_savings_usd=0.0,
                    cache_invalidation_tokens=total_tokens_after_turn,
                    estimated_invalidation_cost_usd=0.0,
                    net_usd=0.0,
                    reason="cost gate disabled, no pricing data — failing open",
                    pricing_source=pricing.source,
                )

        token_savings = turn_tokens - estimated_compacted_tokens
        # Savings compound: removed tokens avoid cache_read_cost on every
        # subsequent API call for the rest of the session.
        remaining = self.config.expected_remaining_turns
        savings_usd = token_savings * pricing.cache_read_cost * remaining

        # Cache invalidation: one-time cost — downstream tokens transition
        # from cache_read to cache_write on the next request only.
        invalidation_cost_usd = total_tokens_after_turn * (
            pricing.cache_write_cost - pricing.cache_read_cost
        )

        net = savings_usd - invalidation_cost_usd
        allowed = net > self.config.min_net_savings_usd

        if allowed:
            reason = (
                f"remove {token_savings:,} cached tokens (save ${savings_usd:.4f} cache_read), "
                f"invalidate {total_tokens_after_turn:,} tokens (cost ${invalidation_cost_usd:.4f}), "
                f"net +${net:.4f}"
            )
        else:
            reason = (
                f"remove {token_savings:,} cached tokens (save ${savings_usd:.4f} cache_read), "
                f"invalidate {total_tokens_after_turn:,} tokens (cost ${invalidation_cost_usd:.4f}), "
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
            effective_invalidation = max(0, candidate.total_tokens_after_turn - already_invalidated)

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
