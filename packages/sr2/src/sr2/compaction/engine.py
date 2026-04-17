"""Compaction engine that applies rules to conversation turns."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sr2.compaction.budget_optimizer import BudgetContext, BudgetOptimizer, OptimizationDecision
from sr2.compaction.cost_gate import CompactionCostGate
from sr2.compaction.rules import CompactionInput, get_rule
from sr2.config.models import CompactionConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""

    turn_number: int
    role: str  # "user", "assistant", "tool_result"
    content: str
    content_type: str | None = None  # "tool_output", "file_content", etc.
    metadata: dict | None = None
    compacted: bool = False


@dataclass
class CostGateResult:
    """Result of the compaction cost gate evaluation."""

    passed: bool
    token_savings_usd: float
    cache_invalidation_usd: float
    net_savings_usd: float


@dataclass
class TurnCompactionDetail:
    """What happened to a single turn during compaction."""

    turn_number: int
    role: str
    content_type: str | None
    rule_applied: str | None
    original_tokens: int
    compacted_tokens: int
    original_content: str
    compacted_content: str


@dataclass
class CompactionResult:
    """Result of running compaction on conversation history."""

    turns: list[ConversationTurn]
    original_tokens: int
    compacted_tokens: int
    turns_compacted: int
    prefix_exempt_turns: int = 0
    analysis: dict | None = None  # Structured analysis from LLM compaction strategy
    details: list[TurnCompactionDetail] = field(default_factory=list)
    cost_gate_result: CostGateResult | None = None
    optimization_decision: OptimizationDecision | None = None


class CompactionEngine:
    """Applies compaction rules to conversation history."""

    def __init__(self, config: CompactionConfig):
        self._config = config
        self._rule_map = self._build_rule_map()
        self._optimizer: BudgetOptimizer | None = None
        self._cost_gate: CompactionCostGate | None = None
        if config.budget_optimizer.enabled:
            self._optimizer = BudgetOptimizer(config.budget_optimizer, config)
        elif config.cost_gate.enabled:
            self._cost_gate = CompactionCostGate(config.cost_gate)

    @property
    def raw_window(self) -> int:
        """Number of recent turns kept in full detail."""
        return self._config.raw_window

    def _build_rule_map(self) -> dict[str, tuple]:
        """Map content_type -> (rule_config, rule_instance)."""
        result = {}
        for rule_config in self._config.rules:
            rule = get_rule(rule_config.strategy)
            result[rule_config.type] = (rule_config, rule)
        return result

    def compact(
        self,
        turns: list[ConversationTurn],
        model_hint: str | None = None,
        prefix_budget: int | None = None,
        token_budget: int | None = None,
        current_tokens: int | None = None,
    ) -> CompactionResult:
        """Apply compaction to turns outside the raw window.

        Args:
            prefix_budget: Session-layer tokens within the cached prefix.
                None = unknown (conservative: cost gate applies to all turns).
                0 = nothing cached (cost gate skipped for all turns).
                N = first N tokens are in the prefix; turns past that boundary
                skip the cost gate since compacting them won't invalidate the cache.
            token_budget: Total token budget for context. Used by BudgetOptimizer
                to calculate utilization and pressure.
            current_tokens: Current total tokens across all layers. Used by
                BudgetOptimizer to calculate utilization.
        """
        raw_window = self._config.raw_window

        if len(turns) <= raw_window:
            total = sum(len(t.content) // 4 for t in turns)
            logger.debug(
                "Compaction skipped: %d turns <= raw_window %d (%d tokens)",
                len(turns),
                raw_window,
                total,
            )
            return CompactionResult(
                turns=turns,
                original_tokens=total,
                compacted_tokens=total,
                turns_compacted=0,
            )

        if self._optimizer:
            return self._compact_with_optimizer(
                turns, raw_window, model_hint, prefix_budget,
                token_budget, current_tokens,
            )

        return self._compact_with_cost_gate(turns, raw_window, model_hint, prefix_budget)

    def _compact_with_optimizer(
        self,
        turns: list[ConversationTurn],
        raw_window: int,
        model_hint: str | None,
        prefix_budget: int | None,
        token_budget: int | None,
        current_tokens: int | None,
    ) -> CompactionResult:
        """Optimizer path: batch selection then apply rules to selected turns."""
        turns = list(turns)  # Work on a copy to avoid mutating caller's list
        budget_ctx = BudgetContext(
            token_budget=token_budget or 0,
            current_tokens=current_tokens or sum(len(t.content) // 4 for t in turns),
            prefix_budget=prefix_budget,
            model_hint=model_hint,
        )

        decision = self._optimizer.select_turns(turns, raw_window, budget_ctx)

        if not decision.turns_to_compact:
            total = sum(len(t.content) // 4 for t in turns)
            logger.info("BudgetOptimizer: %s", decision.reason)
            return CompactionResult(
                turns=turns,
                original_tokens=total,
                compacted_tokens=total,
                turns_compacted=0,
                optimization_decision=decision,
            )

        selected_set = set(decision.turns_to_compact)
        original_tokens = 0
        compacted_tokens = 0
        turns_compacted = 0
        details: list[TurnCompactionDetail] = []

        # Build candidate lookup from optimizer for dry-run content reuse
        candidates = self._optimizer._build_candidates(turns, raw_window, budget_ctx)
        candidate_by_index = {c.turn_index: c for c in candidates}

        for i, turn in enumerate(turns):
            est_tokens = len(turn.content) // 4
            original_tokens += est_tokens

            if i not in selected_set:
                compacted_tokens += est_tokens
                continue

            candidate = candidate_by_index.get(i)

            # Reuse dry-run content if available
            if candidate and candidate.estimated_compacted_content is not None:
                compacted_content = candidate.estimated_compacted_content
                new_tokens = candidate.estimated_compacted_tokens
                rule_name = candidate.rule_name
            else:
                # Run rule fresh
                if turn.content_type not in self._rule_map:
                    compacted_tokens += est_tokens
                    continue
                rule_config, rule = self._rule_map[turn.content_type]
                inp = CompactionInput(
                    content=turn.content,
                    content_type=turn.content_type,
                    tokens=est_tokens,
                    metadata=turn.metadata,
                )
                output = rule.compact(inp, rule_config.model_dump())
                if not output.was_compacted:
                    compacted_tokens += est_tokens
                    continue
                compacted_content = output.content
                if output.recovery_hint:
                    compacted_content += f"\n  Recovery: {output.recovery_hint}"
                new_tokens = output.tokens
                rule_name = rule_config.strategy

            new_turn = ConversationTurn(
                turn_number=turn.turn_number,
                role=turn.role,
                content=compacted_content,
                content_type=turn.content_type,
                metadata=turn.metadata,
                compacted=True,
            )
            turns[i] = new_turn

            details.append(TurnCompactionDetail(
                turn_number=turn.turn_number,
                role=turn.role,
                content_type=turn.content_type,
                rule_applied=rule_name,
                original_tokens=est_tokens,
                compacted_tokens=new_tokens,
                original_content=turn.content,
                compacted_content=compacted_content,
            ))
            compacted_tokens += new_tokens
            turns_compacted += 1

        logger.info(
            "Compaction (optimizer) completed - %d -> %d tokens, "
            "compacted %d/%d turns (pressure=%.2f, force=%s)",
            original_tokens,
            compacted_tokens,
            turns_compacted,
            len(turns),
            decision.budget_pressure,
            decision.force_mode,
        )

        return CompactionResult(
            turns=turns,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            turns_compacted=turns_compacted,
            details=details,
            optimization_decision=decision,
        )

    def _compact_with_cost_gate(
        self,
        turns: list[ConversationTurn],
        raw_window: int,
        model_hint: str | None,
        prefix_budget: int | None,
    ) -> CompactionResult:
        """Legacy cost gate path: per-turn evaluation."""
        min_size = self._config.min_content_size

        compactable = turns[:-raw_window]
        protected = turns[-raw_window:]

        original_tokens = 0
        compacted_tokens = 0
        turns_compacted = 0
        prefix_exempt_count = 0
        cumulative_tokens = 0
        details: list[TurnCompactionDetail] = []
        last_cost_gate_result: CostGateResult | None = None

        for turn in compactable:
            est_tokens = len(turn.content) // 4
            original_tokens += est_tokens
            cumulative_tokens += est_tokens

            if turn.role == "user":
                compacted_tokens += est_tokens
                continue
            if turn.compacted:
                compacted_tokens += est_tokens
                continue
            if est_tokens < min_size:
                compacted_tokens += est_tokens
                continue
            if turn.content_type not in self._rule_map:
                compacted_tokens += est_tokens
                continue

            rule_config, rule = self._rule_map[turn.content_type]

            past_prefix = (
                prefix_budget is not None
                and cumulative_tokens > prefix_budget
            )

            if self._cost_gate is not None and not past_prefix:
                estimated_compacted = max(est_tokens // 4, 10)
                idx = compactable.index(turn)
                tokens_after = sum(len(t.content) // 4 for t in compactable[idx + 1 :]) + sum(
                    len(t.content) // 4 for t in protected
                )

                decision = self._cost_gate.should_compact(
                    turn_index=turn.turn_number,
                    turn_tokens=est_tokens,
                    estimated_compacted_tokens=estimated_compacted,
                    total_tokens_after_turn=tokens_after,
                    model_hint=model_hint,
                )
                last_cost_gate_result = CostGateResult(
                    passed=decision.allowed,
                    token_savings_usd=decision.estimated_savings_usd,
                    cache_invalidation_usd=decision.estimated_invalidation_cost_usd,
                    net_savings_usd=decision.net_usd,
                )
                if not decision.allowed:
                    logger.info(
                        "Turn %d: cost gate blocked compaction — %s",
                        turn.turn_number,
                        decision.reason,
                    )
                    details.append(TurnCompactionDetail(
                        turn_number=turn.turn_number,
                        role=turn.role,
                        content_type=turn.content_type,
                        rule_applied=None,
                        original_tokens=est_tokens,
                        compacted_tokens=est_tokens,
                        original_content=turn.content,
                        compacted_content=turn.content,
                    ))
                    compacted_tokens += est_tokens
                    continue
            elif past_prefix:
                prefix_exempt_count += 1

            inp = CompactionInput(
                content=turn.content,
                content_type=turn.content_type,
                tokens=est_tokens,
                metadata=turn.metadata,
            )
            output = rule.compact(inp, rule_config.model_dump())

            if output.was_compacted:
                compacted_content = output.content
                if output.recovery_hint:
                    compacted_content += f"\n  Recovery: {output.recovery_hint}"
                new_turn = ConversationTurn(
                    turn_number=turn.turn_number,
                    role=turn.role,
                    content=compacted_content,
                    content_type=turn.content_type,
                    metadata=turn.metadata,
                    compacted=True,
                )
                compactable[compactable.index(turn)] = new_turn
                details.append(TurnCompactionDetail(
                    turn_number=turn.turn_number,
                    role=turn.role,
                    content_type=turn.content_type,
                    rule_applied=rule_config.strategy,
                    original_tokens=est_tokens,
                    compacted_tokens=output.tokens,
                    original_content=turn.content,
                    compacted_content=compacted_content,
                ))
                compacted_tokens += output.tokens
                turns_compacted += 1
            else:
                compacted_tokens += est_tokens

        for turn in protected:
            est = len(turn.content) // 4
            original_tokens += est
            compacted_tokens += est

        all_turns = compactable + protected
        logger.info(
            "Compaction completed - went from %d to %d, compacted %d/%d turns",
            original_tokens,
            compacted_tokens,
            turns_compacted,
            len(all_turns),
        )

        return CompactionResult(
            turns=all_turns,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            turns_compacted=turns_compacted,
            prefix_exempt_turns=prefix_exempt_count,
            details=details,
            cost_gate_result=last_cost_gate_result,
        )
