"""Budget-aware compaction optimizer.

Replaces the per-turn CompactionCostGate with holistic batch reasoning
that accounts for budget pressure, correct token economics, and dry-run
size estimation using actual compaction rules.
"""

import logging
from dataclasses import dataclass, field

from sr2.compaction.pricing import CachePricing, resolve_pricing
from sr2.compaction.rules import CompactionInput, CompactionOutput, get_rule
from sr2.config.models import BudgetOptimizerConfig, CompactionConfig

logger = logging.getLogger(__name__)


@dataclass
class BudgetContext:
    """Snapshot of current budget situation."""

    token_budget: int
    current_tokens: int
    prefix_budget: int | None = None
    model_hint: str | None = None

    @property
    def utilization(self) -> float:
        if self.token_budget <= 0:
            return 0.0
        return self.current_tokens / self.token_budget

    @property
    def headroom(self) -> int:
        return max(0, self.token_budget - self.current_tokens)


@dataclass
class TurnCandidate:
    """A turn evaluated for potential compaction."""

    turn_index: int
    turn_number: int
    original_tokens: int
    estimated_compacted_tokens: int
    estimated_compacted_content: str | None = None
    recovery_hint: str | None = None
    rule_name: str | None = None
    in_raw_window: bool = False
    in_cached_prefix: bool = False
    cumulative_tokens_before: int = 0

    @property
    def savings(self) -> int:
        return self.original_tokens - self.estimated_compacted_tokens

    @property
    def savings_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return self.savings / self.original_tokens


@dataclass
class CostAnalysis:
    """Economic breakdown for a selected compaction set."""

    total_token_savings: int
    input_cost_savings_usd: float
    cache_read_savings_usd: float
    cache_write_penalty_usd: float
    pressure_multiplier: float
    net_usd: float
    remaining_turns: int


@dataclass
class OptimizationDecision:
    """Result of the optimizer's holistic evaluation."""

    turns_to_compact: list[int] = field(default_factory=list)
    raw_window_invaded: bool = False
    budget_pressure: float = 0.0
    force_mode: bool = False
    total_estimated_savings: int = 0
    cost_analysis: CostAnalysis | None = None
    reason: str = ""


class BudgetOptimizer:
    """Holistic budget-aware compaction optimizer.

    Makes compaction decisions based on the full picture: budget utilization,
    batch economics with correct token pricing, and dry-run rule estimation.
    """

    def __init__(self, config: BudgetOptimizerConfig, compaction_config: CompactionConfig):
        self._config = config
        self._compaction_config = compaction_config
        self._rule_map = self._build_rule_map(compaction_config)
        self._pricing_cache: dict[str, CachePricing] = {}

    @staticmethod
    def _build_rule_map(compaction_config: CompactionConfig) -> dict[str, tuple]:
        result = {}
        for rule_config in compaction_config.rules:
            rule = get_rule(rule_config.strategy)
            result[rule_config.type] = (rule_config, rule)
        return result

    def select_turns(
        self,
        turns: list,
        raw_window: int,
        budget_context: BudgetContext,
    ) -> OptimizationDecision:
        """Select which turns to compact based on budget pressure and economics."""
        if not turns:
            return OptimizationDecision(reason="no turns to evaluate")

        candidates = self._build_candidates(turns, raw_window, budget_context)
        if not candidates:
            return OptimizationDecision(reason="no compactable candidates found")

        if budget_context.token_budget <= 0:
            return self._select_all_compactable(candidates)

        pressure = self._compute_pressure(budget_context.utilization)
        force_mode = budget_context.utilization >= self._config.force_threshold

        logger.debug(
            "BudgetOptimizer: utilization=%.2f, pressure=%.3f, force=%s, candidates=%d",
            budget_context.utilization,
            pressure,
            force_mode,
            len(candidates),
        )

        if force_mode:
            return self._select_force_mode(candidates, budget_context, pressure)

        pricing = self._resolve_pricing(budget_context.model_hint)

        if pressure > 0:
            selected = self._select_by_value(candidates, budget_context, pressure, pricing)
        else:
            selected = self._select_by_value(candidates, budget_context, 0.0, pricing)

        if not selected:
            return OptimizationDecision(
                budget_pressure=pressure,
                reason="no turns worth compacting at current pressure",
            )

        selected_candidates = [c for c in candidates if c.turn_index in selected]
        total_savings = sum(c.savings for c in selected_candidates)
        cost_analysis = self._calculate_economics(
            candidates, selected_candidates, budget_context, pressure, pricing,
        )

        return OptimizationDecision(
            turns_to_compact=sorted(selected),
            budget_pressure=pressure,
            total_estimated_savings=total_savings,
            cost_analysis=cost_analysis,
            reason=f"selected {len(selected)} turns, saving ~{total_savings} tokens "
            f"(pressure={pressure:.2f}, net=${cost_analysis.net_usd:.6f})",
        )

    def _build_candidates(
        self,
        turns: list,
        raw_window: int,
        budget_context: BudgetContext,
    ) -> list[TurnCandidate]:
        candidates = []
        min_size = self._compaction_config.min_content_size
        raw_start = max(0, len(turns) - raw_window)
        cumulative = 0

        for i, turn in enumerate(turns):
            est_tokens = len(turn.content) // 4
            in_rw = i >= raw_start
            in_prefix = (
                budget_context.prefix_budget is not None
                and cumulative <= budget_context.prefix_budget
            )

            can_compact = (
                turn.role != "user"
                and not turn.compacted
                and est_tokens >= min_size
                and turn.content_type in self._rule_map
            )

            if can_compact:
                est_compacted, est_content, hint, rule_name = self._estimate_compacted_size(turn)
                candidates.append(
                    TurnCandidate(
                        turn_index=i,
                        turn_number=turn.turn_number,
                        original_tokens=est_tokens,
                        estimated_compacted_tokens=est_compacted,
                        estimated_compacted_content=est_content,
                        recovery_hint=hint,
                        rule_name=rule_name,
                        in_raw_window=in_rw,
                        in_cached_prefix=in_prefix,
                        cumulative_tokens_before=cumulative,
                    )
                )

            cumulative += est_tokens

        return candidates

    def _select_all_compactable(
        self, candidates: list[TurnCandidate],
    ) -> OptimizationDecision:
        """Select all compactable turns outside raw_window (no economic gating)."""
        selected = [c.turn_index for c in candidates if not c.in_raw_window and c.savings > 0]
        total_savings = sum(c.savings for c in candidates if c.turn_index in set(selected))
        return OptimizationDecision(
            turns_to_compact=sorted(selected),
            total_estimated_savings=total_savings,
            reason=f"no budget info — compacting all {len(selected)} eligible turns",
        )

    def _estimate_compacted_size(
        self, turn,
    ) -> tuple[int, str | None, str | None, str | None]:
        """Estimate compacted size. Returns (tokens, content, recovery_hint, rule_name).

        When dry_run is enabled, runs the actual compaction rule to get real output.
        Falls back to original_tokens // 4 heuristic otherwise.
        """
        if turn.content_type not in self._rule_map:
            est = max(len(turn.content) // 4 // 4, 10)
            return est, None, None, None

        rule_config, rule = self._rule_map[turn.content_type]
        rule_name = rule_config.strategy

        if not self._config.dry_run:
            est = max(len(turn.content) // 4 // 4, 10)
            return est, None, None, rule_name

        try:
            inp = CompactionInput(
                content=turn.content,
                content_type=turn.content_type,
                tokens=len(turn.content) // 4,
                metadata=turn.metadata,
            )
            output: CompactionOutput = rule.compact(inp, rule_config.model_dump())
            if output.was_compacted:
                content = output.content
                if output.recovery_hint:
                    content += f"\n  Recovery: {output.recovery_hint}"
                return output.tokens, content, output.recovery_hint, rule_name
            return len(turn.content) // 4, None, None, rule_name
        except Exception:
            logger.debug(
                "Dry-run failed for turn %d (content_type=%s), falling back to heuristic",
                turn.turn_number,
                turn.content_type,
            )
            est = max(len(turn.content) // 4 // 4, 10)
            return est, None, None, rule_name

    def _compute_pressure(self, utilization: float) -> float:
        t = self._config.pressure_threshold
        f = self._config.force_threshold
        if utilization <= t:
            return 0.0
        if utilization >= f:
            return 1.0
        normalized = (utilization - t) / (f - t)
        return normalized ** 2

    def _select_force_mode(
        self,
        candidates: list[TurnCandidate],
        budget_context: BudgetContext,
        pressure: float,
    ) -> OptimizationDecision:
        selected: list[int] = []
        total_savings = 0
        raw_window_invaded = False

        outside_rw = [c for c in candidates if not c.in_raw_window]
        for c in outside_rw:
            selected.append(c.turn_index)
            total_savings += c.savings

        tokens_after = budget_context.current_tokens - total_savings
        if tokens_after > budget_context.token_budget:
            inside_rw = sorted(
                [c for c in candidates if c.in_raw_window],
                key=lambda c: c.savings_ratio,
                reverse=True,
            )
            for c in inside_rw:
                if tokens_after <= budget_context.token_budget:
                    break
                selected.append(c.turn_index)
                total_savings += c.savings
                tokens_after -= c.savings
                raw_window_invaded = True

        return OptimizationDecision(
            turns_to_compact=sorted(selected),
            raw_window_invaded=raw_window_invaded,
            budget_pressure=pressure,
            force_mode=True,
            total_estimated_savings=total_savings,
            reason=f"force mode: {len(selected)} turns, ~{total_savings} tokens saved"
            + (", invaded raw_window" if raw_window_invaded else ""),
        )

    def _select_by_value(
        self,
        candidates: list[TurnCandidate],
        budget_context: BudgetContext,
        pressure: float,
        pricing: CachePricing,
    ) -> set[int]:
        selected: set[int] = set()
        remaining_turns = self._config.expected_remaining_turns

        adjusted_threshold = self._config.min_net_savings_usd * (1 - pressure)

        outside_prefix = [c for c in candidates if not c.in_cached_prefix and not c.in_raw_window]
        for c in outside_prefix:
            if c.savings > 0:
                selected.add(c.turn_index)

        inside_prefix = sorted(
            [c for c in candidates if c.in_cached_prefix and not c.in_raw_window],
            key=lambda c: c.turn_index,
        )

        already_invalidated = 0
        for c in inside_prefix:
            if c.savings <= 0:
                continue

            savings_per_turn = c.savings * pricing.cache_read_cost
            marginal_savings = savings_per_turn * remaining_turns

            tokens_after = c.cumulative_tokens_before + c.original_tokens
            new_invalidation = max(0, tokens_after - already_invalidated)
            marginal_invalidation = new_invalidation * (
                pricing.cache_write_cost - pricing.cache_read_cost
            )

            marginal_net = marginal_savings - max(0, marginal_invalidation)

            if marginal_net > adjusted_threshold:
                selected.add(c.turn_index)
                already_invalidated = max(already_invalidated, tokens_after)

        return selected

    def _calculate_economics(
        self,
        all_candidates: list[TurnCandidate],
        selected: list[TurnCandidate],
        budget_context: BudgetContext,
        pressure: float,
        pricing: CachePricing,
    ) -> CostAnalysis:
        remaining = self._config.expected_remaining_turns
        total_savings = sum(c.savings for c in selected)

        inside_savings = sum(c.savings for c in selected if c.in_cached_prefix)
        outside_savings = sum(c.savings for c in selected if not c.in_cached_prefix)

        cache_read_savings = inside_savings * pricing.cache_read_cost * remaining
        input_cost_savings = outside_savings * pricing.input_cost * remaining

        max_invalidated = 0
        for c in sorted(selected, key=lambda x: x.turn_index):
            if c.in_cached_prefix:
                after = c.cumulative_tokens_before + c.original_tokens
                max_invalidated = max(max_invalidated, after)

        write_penalty = max_invalidated * max(0, pricing.cache_write_cost - pricing.cache_read_cost)

        pressure_multiplier = 1 + pressure * 10
        raw_net = (cache_read_savings + input_cost_savings) - write_penalty
        net = raw_net * pressure_multiplier

        return CostAnalysis(
            total_token_savings=total_savings,
            input_cost_savings_usd=input_cost_savings,
            cache_read_savings_usd=cache_read_savings,
            cache_write_penalty_usd=write_penalty,
            pressure_multiplier=pressure_multiplier,
            net_usd=net,
            remaining_turns=remaining,
        )

    def _resolve_pricing(self, model_hint: str | None) -> CachePricing:
        cache_key = model_hint or "_default"
        if cache_key not in self._pricing_cache:
            self._pricing_cache[cache_key] = resolve_pricing(
                model_hint=model_hint,
                fallback_model=self._config.fallback_model,
                custom_pricing=self._config.custom_pricing,
            )
        return self._pricing_cache[cache_key]

