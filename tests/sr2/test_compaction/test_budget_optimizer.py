"""Tests for BudgetOptimizer."""

import pytest

from sr2.compaction.budget_optimizer import (
    BudgetContext,
    BudgetOptimizer,
    CostAnalysis,
    OptimizationDecision,
    TurnCandidate,
)
from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.compaction.pricing import CachePricing
from sr2.config.models import (
    BudgetOptimizerConfig,
    CompactionConfig,
    CompactionRuleConfig,
    CostGateConfig,
)

_NO_COST_GATE = CostGateConfig(enabled=False)

_SONNET_PRICING = CachePricing(
    input_cost=3.0 / 1_000_000,
    cache_write_cost=3.75 / 1_000_000,
    cache_read_cost=0.30 / 1_000_000,
    source="test",
)


def _make_config(
    pressure_threshold: float = 0.8,
    force_threshold: float = 0.95,
    dry_run: bool = True,
    min_net: float = 0.0,
) -> tuple[BudgetOptimizerConfig, CompactionConfig]:
    opt_config = BudgetOptimizerConfig(
        enabled=True,
        pressure_threshold=pressure_threshold,
        force_threshold=force_threshold,
        dry_run=dry_run,
        min_net_savings_usd=min_net,
    )
    comp_config = CompactionConfig(
        enabled=True,
        raw_window=3,
        min_content_size=10,
        cost_gate=_NO_COST_GATE,
        budget_optimizer=opt_config,
        rules=[
            CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
            CompactionRuleConfig(type="file_content", strategy="reference"),
        ],
    )
    return opt_config, comp_config


def _make_optimizer(**kwargs) -> BudgetOptimizer:
    opt_config, comp_config = _make_config(**kwargs)
    optimizer = BudgetOptimizer(opt_config, comp_config)
    optimizer._pricing_cache["_default"] = _SONNET_PRICING
    return optimizer


def _long_content(lines: int = 20) -> str:
    return "\n".join(f"data line {i}: some verbose output" for i in range(lines))


def _make_turns(n_compactable: int, n_raw: int) -> list[ConversationTurn]:
    turns = []
    for i in range(n_compactable):
        turns.append(ConversationTurn(
            turn_number=i,
            role="assistant",
            content=_long_content(20),
            content_type="tool_output",
        ))
    for i in range(n_raw):
        turns.append(ConversationTurn(
            turn_number=n_compactable + i,
            role="assistant",
            content="short reply",
        ))
    return turns


# ---------------------------------------------------------------------------
# Pressure calculation
# ---------------------------------------------------------------------------

class TestPressureCalculation:

    def test_below_threshold_returns_zero(self):
        optimizer = _make_optimizer(pressure_threshold=0.8)
        assert optimizer._compute_pressure(0.5) == 0.0

    def test_at_threshold_returns_zero(self):
        optimizer = _make_optimizer(pressure_threshold=0.8)
        assert optimizer._compute_pressure(0.8) == 0.0

    def test_midpoint_quadratic(self):
        optimizer = _make_optimizer(pressure_threshold=0.8, force_threshold=1.0)
        # midpoint = 0.9, normalized = 0.5, quadratic = 0.25
        assert optimizer._compute_pressure(0.9) == pytest.approx(0.25)

    def test_at_force_returns_one(self):
        optimizer = _make_optimizer(force_threshold=0.95)
        assert optimizer._compute_pressure(0.95) == 1.0

    def test_above_force_returns_one(self):
        optimizer = _make_optimizer(force_threshold=0.95)
        assert optimizer._compute_pressure(1.2) == 1.0

    def test_custom_thresholds(self):
        optimizer = _make_optimizer(pressure_threshold=0.5, force_threshold=0.7)
        assert optimizer._compute_pressure(0.4) == 0.0
        assert optimizer._compute_pressure(0.7) == 1.0
        # 0.6 = midpoint, normalized = 0.5, quadratic = 0.25
        assert optimizer._compute_pressure(0.6) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Dry-run estimation
# ---------------------------------------------------------------------------

class TestDryRunEstimation:

    def test_dry_run_returns_actual_rule_output(self):
        optimizer = _make_optimizer(dry_run=True)
        turn = ConversationTurn(
            turn_number=0, role="assistant",
            content=_long_content(20),
            content_type="tool_output",
        )
        tokens, content, hint, rule_name = optimizer._estimate_compacted_size(turn)
        assert content is not None
        assert "lines" in content
        assert rule_name == "schema_and_sample"
        assert tokens < len(turn.content) // 4

    def test_dry_run_false_uses_heuristic(self):
        optimizer = _make_optimizer(dry_run=False)
        turn = ConversationTurn(
            turn_number=0, role="assistant",
            content=_long_content(20),
            content_type="tool_output",
        )
        tokens, content, hint, rule_name = optimizer._estimate_compacted_size(turn)
        assert content is None
        assert rule_name == "schema_and_sample"
        expected = max(len(turn.content) // 4 // 4, 10)
        assert tokens == expected

    def test_no_matching_rule_uses_heuristic(self):
        optimizer = _make_optimizer()
        turn = ConversationTurn(
            turn_number=0, role="assistant",
            content=_long_content(20),
            content_type="unknown_type",
        )
        tokens, content, hint, rule_name = optimizer._estimate_compacted_size(turn)
        assert content is None
        assert rule_name is None

    def test_dry_run_with_few_lines_not_compacted(self):
        """SchemaAndSampleRule skips content with <= 3 lines."""
        optimizer = _make_optimizer(dry_run=True)
        turn = ConversationTurn(
            turn_number=0, role="assistant",
            content="line1\nline2\nline3",
            content_type="tool_output",
        )
        tokens, content, hint, rule_name = optimizer._estimate_compacted_size(turn)
        # Rule returns was_compacted=False for <= 3 lines
        assert content is None


# ---------------------------------------------------------------------------
# Turn selection — no pressure
# ---------------------------------------------------------------------------

class TestSelectionNoPressure:

    def test_no_budget_compacts_everything(self):
        """When token_budget=0, all compactable turns are selected."""
        optimizer = _make_optimizer()
        turns = _make_turns(5, 3)
        ctx = BudgetContext(token_budget=0, current_tokens=0)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert len(decision.turns_to_compact) == 5
        assert not decision.force_mode

    def test_low_utilization_no_compaction(self):
        """At 50% utilization with high min_net and cached prefix, nothing compacts."""
        optimizer = _make_optimizer(min_net=999.0)
        turns = _make_turns(5, 3)
        total = sum(len(t.content) // 4 for t in turns)
        ctx = BudgetContext(
            token_budget=total * 2,
            current_tokens=total,
            prefix_budget=total * 2,  # Everything is in cached prefix
        )
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert len(decision.turns_to_compact) == 0

    def test_raw_window_never_selected(self):
        """Turns in raw_window are never selected at zero pressure."""
        optimizer = _make_optimizer()
        turns = _make_turns(2, 3)
        # All turns are tool_output but last 3 are in raw window
        for t in turns:
            t.content_type = "tool_output"
            t.content = _long_content(20)
        ctx = BudgetContext(token_budget=0, current_tokens=0)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        # Only turns 0 and 1 should be selected (outside raw window)
        assert set(decision.turns_to_compact) == {0, 1}

    def test_user_messages_never_selected(self):
        """User messages are filtered by _build_candidates."""
        optimizer = _make_optimizer()
        turns = [
            ConversationTurn(turn_number=0, role="user", content=_long_content(20), content_type="tool_output"),
            ConversationTurn(turn_number=1, role="assistant", content=_long_content(20), content_type="tool_output"),
            ConversationTurn(turn_number=2, role="assistant", content="ok"),
            ConversationTurn(turn_number=3, role="assistant", content="done"),
            ConversationTurn(turn_number=4, role="assistant", content="end"),
        ]
        ctx = BudgetContext(token_budget=0, current_tokens=0)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert 0 not in decision.turns_to_compact
        assert 1 in decision.turns_to_compact

    def test_already_compacted_skipped(self):
        optimizer = _make_optimizer()
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(20),
                content_type="tool_output", compacted=True,
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="ok"),
            ConversationTurn(turn_number=3, role="assistant", content="ok"),
        ]
        ctx = BudgetContext(token_budget=0, current_tokens=0)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert len(decision.turns_to_compact) == 0

    def test_empty_turns(self):
        optimizer = _make_optimizer()
        ctx = BudgetContext(token_budget=1000, current_tokens=500)
        decision = optimizer.select_turns([], raw_window=3, budget_context=ctx)
        assert len(decision.turns_to_compact) == 0


# ---------------------------------------------------------------------------
# Turn selection — pressure mode
# ---------------------------------------------------------------------------

class TestSelectionPressure:

    def test_more_turns_selected_at_higher_pressure(self):
        """Higher pressure lowers the threshold, selecting more turns."""
        opt_config, comp_config = _make_config(min_net=0.01)
        optimizer = BudgetOptimizer(opt_config, comp_config)
        optimizer._pricing_cache["_default"] = _SONNET_PRICING

        turns = _make_turns(5, 3)
        total = sum(len(t.content) // 4 for t in turns)

        # Low utilization (50%) — zero pressure
        ctx_low = BudgetContext(token_budget=total * 2, current_tokens=total)
        decision_low = optimizer.select_turns(turns, raw_window=3, budget_context=ctx_low)

        # High utilization (90%) — significant pressure
        ctx_high = BudgetContext(
            token_budget=int(total / 0.9),
            current_tokens=total,
        )
        decision_high = optimizer.select_turns(turns, raw_window=3, budget_context=ctx_high)

        assert len(decision_high.turns_to_compact) >= len(decision_low.turns_to_compact)

    def test_raw_window_still_respected_in_pressure(self):
        optimizer = _make_optimizer()
        turns = _make_turns(3, 3)
        for t in turns:
            t.content_type = "tool_output"
            t.content = _long_content(20)
        total = sum(len(t.content) // 4 for t in turns)
        # 90% utilization — pressure mode but not force
        ctx = BudgetContext(token_budget=int(total / 0.9), current_tokens=total)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        # Raw window turns (last 3) should NOT be selected
        raw_indices = {3, 4, 5}
        assert not (set(decision.turns_to_compact) & raw_indices)
        assert not decision.raw_window_invaded


# ---------------------------------------------------------------------------
# Turn selection — force mode
# ---------------------------------------------------------------------------

class TestSelectionForce:

    def test_force_mode_selects_all_outside_raw(self):
        optimizer = _make_optimizer(force_threshold=0.95)
        turns = _make_turns(5, 3)
        total = sum(len(t.content) // 4 for t in turns)
        # 96% utilization — force mode
        ctx = BudgetContext(token_budget=int(total / 0.96), current_tokens=total)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert decision.force_mode
        assert len(decision.turns_to_compact) == 5

    def test_force_mode_invades_raw_window_when_needed(self):
        optimizer = _make_optimizer(force_threshold=0.95)
        # Only raw window turns are compactable (all tool_output)
        turns = [
            ConversationTurn(turn_number=i, role="assistant",
                             content=_long_content(20), content_type="tool_output")
            for i in range(4)
        ]
        total = sum(len(t.content) // 4 for t in turns)
        # Way over budget — force mode, but all turns are in raw_window(3) except turn 0
        ctx = BudgetContext(token_budget=int(total * 0.3), current_tokens=total)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert decision.force_mode
        assert decision.raw_window_invaded
        assert len(decision.turns_to_compact) > 1

    def test_force_mode_picks_highest_savings_ratio_from_raw(self):
        optimizer = _make_optimizer(force_threshold=0.95)
        turns = [
            # Outside raw window — always selected
            ConversationTurn(turn_number=0, role="assistant",
                             content=_long_content(5), content_type="tool_output"),
            # Raw window — turn 1 has small content, turn 2 has large
            ConversationTurn(turn_number=1, role="assistant",
                             content=_long_content(5), content_type="tool_output"),
            ConversationTurn(turn_number=2, role="assistant",
                             content=_long_content(50), content_type="tool_output"),
            ConversationTurn(turn_number=3, role="assistant", content="ok"),
        ]
        total = sum(len(t.content) // 4 for t in turns)
        ctx = BudgetContext(token_budget=int(total * 0.3), current_tokens=total)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert decision.force_mode
        # Turn 2 (large content) should be picked before turn 1 (small content)
        if decision.raw_window_invaded and len(decision.turns_to_compact) >= 2:
            assert 2 in decision.turns_to_compact

    def test_force_decision_has_no_cost_analysis(self):
        optimizer = _make_optimizer(force_threshold=0.95)
        turns = _make_turns(3, 3)
        total = sum(len(t.content) // 4 for t in turns)
        ctx = BudgetContext(token_budget=int(total / 0.96), current_tokens=total)
        decision = optimizer.select_turns(turns, raw_window=3, budget_context=ctx)
        assert decision.force_mode
        assert decision.cost_analysis is None


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

class TestEconomics:

    def test_cache_read_savings_for_prefix_tokens(self):
        optimizer = _make_optimizer()
        candidates = [
            TurnCandidate(
                turn_index=0, turn_number=0, original_tokens=1000,
                estimated_compacted_tokens=100, in_cached_prefix=True,
                cumulative_tokens_before=0,
            ),
        ]
        analysis = optimizer._calculate_economics(
            candidates, candidates,
            BudgetContext(token_budget=5000, current_tokens=4000),
            pressure=0.0, pricing=_SONNET_PRICING,
        )
        # savings = 900 tokens * cache_read * 10 turns
        expected_cache = 900 * _SONNET_PRICING.cache_read_cost * 10
        assert analysis.cache_read_savings_usd == pytest.approx(expected_cache)

    def test_input_cost_savings_for_non_prefix_tokens(self):
        optimizer = _make_optimizer()
        candidates = [
            TurnCandidate(
                turn_index=0, turn_number=0, original_tokens=1000,
                estimated_compacted_tokens=100, in_cached_prefix=False,
                cumulative_tokens_before=0,
            ),
        ]
        analysis = optimizer._calculate_economics(
            candidates, candidates,
            BudgetContext(token_budget=5000, current_tokens=4000),
            pressure=0.0, pricing=_SONNET_PRICING,
        )
        expected_input = 900 * _SONNET_PRICING.input_cost * 10
        assert analysis.input_cost_savings_usd == pytest.approx(expected_input)

    def test_pressure_multiplier_applied(self):
        optimizer = _make_optimizer()
        candidates = [
            TurnCandidate(
                turn_index=0, turn_number=0, original_tokens=1000,
                estimated_compacted_tokens=100, in_cached_prefix=False,
                cumulative_tokens_before=0,
            ),
        ]
        analysis_zero = optimizer._calculate_economics(
            candidates, candidates,
            BudgetContext(token_budget=5000, current_tokens=4000),
            pressure=0.0, pricing=_SONNET_PRICING,
        )
        analysis_high = optimizer._calculate_economics(
            candidates, candidates,
            BudgetContext(token_budget=5000, current_tokens=4000),
            pressure=0.5, pricing=_SONNET_PRICING,
        )
        assert analysis_zero.pressure_multiplier == 1.0
        assert analysis_high.pressure_multiplier == 6.0
        assert analysis_high.net_usd > analysis_zero.net_usd


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineIntegration:

    def test_optimizer_wired_when_enabled(self):
        _, comp_config = _make_config()
        engine = CompactionEngine(comp_config)
        assert engine._optimizer is not None
        assert engine._cost_gate is None

    def test_optimizer_disabled_falls_back_to_cost_gate(self):
        config = CompactionConfig(
            enabled=True,
            cost_gate=CostGateConfig(enabled=True),
            budget_optimizer=BudgetOptimizerConfig(enabled=False),
            rules=[CompactionRuleConfig(type="tool_output", strategy="schema_and_sample")],
        )
        engine = CompactionEngine(config)
        assert engine._optimizer is None
        assert engine._cost_gate is not None

    def test_compact_with_optimizer_no_budget(self):
        """No budget info: optimizer compacts everything compactable."""
        _, comp_config = _make_config()
        comp_config = comp_config.model_copy(update={"raw_window": 2})
        engine = CompactionEngine(comp_config)
        turns = [
            ConversationTurn(turn_number=0, role="assistant", content=_long_content(20), content_type="tool_output"),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 1
        assert result.optimization_decision is not None

    def test_compact_with_budget_info(self):
        """Budget info provided: optimizer uses pressure calculation."""
        _, comp_config = _make_config()
        comp_config = comp_config.model_copy(update={"raw_window": 2})
        engine = CompactionEngine(comp_config)
        turns = [
            ConversationTurn(turn_number=0, role="assistant", content=_long_content(20), content_type="tool_output"),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        total = sum(len(t.content) // 4 for t in turns)
        result = engine.compact(turns, token_budget=total * 2, current_tokens=total)
        assert result.optimization_decision is not None
        assert result.optimization_decision.budget_pressure == 0.0

    def test_dry_run_content_reused(self):
        """Dry-run content from optimizer is reused, not re-computed."""
        _, comp_config = _make_config(dry_run=True)
        comp_config = comp_config.model_copy(update={"raw_window": 1})
        engine = CompactionEngine(comp_config)
        turns = [
            ConversationTurn(turn_number=0, role="assistant", content=_long_content(20), content_type="tool_output"),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 1
        # The compacted content should contain the sample lines
        assert "lines" in result.turns[0].content

    def test_raw_window_respected_with_budget(self):
        """Raw window turns not compacted even with budget pressure."""
        _, comp_config = _make_config()
        comp_config = comp_config.model_copy(update={"raw_window": 3})
        engine = CompactionEngine(comp_config)
        turns = [
            ConversationTurn(turn_number=i, role="assistant",
                             content=_long_content(20), content_type="tool_output")
            for i in range(6)
        ]
        total = sum(len(t.content) // 4 for t in turns)
        # 85% utilization — pressure mode
        result = engine.compact(turns, token_budget=int(total / 0.85), current_tokens=total)
        # Last 3 turns should not be compacted
        for turn in result.turns[-3:]:
            assert not turn.compacted

    def test_idempotent(self):
        """Compaction is idempotent with optimizer."""
        _, comp_config = _make_config()
        comp_config = comp_config.model_copy(update={"raw_window": 1})
        engine = CompactionEngine(comp_config)
        turns = [
            ConversationTurn(turn_number=0, role="assistant", content=_long_content(20), content_type="tool_output"),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result1 = engine.compact(turns)
        result2 = engine.compact(result1.turns)
        assert result2.turns_compacted == 0


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_force_must_be_greater_than_pressure(self):
        with pytest.raises(ValueError, match="force_threshold.*must be greater"):
            BudgetOptimizerConfig(pressure_threshold=0.9, force_threshold=0.8)

    def test_equal_thresholds_rejected(self):
        with pytest.raises(ValueError, match="force_threshold.*must be greater"):
            BudgetOptimizerConfig(pressure_threshold=0.8, force_threshold=0.8)

    def test_valid_thresholds_accepted(self):
        config = BudgetOptimizerConfig(pressure_threshold=0.5, force_threshold=0.9)
        assert config.pressure_threshold == 0.5
        assert config.force_threshold == 0.9


# ---------------------------------------------------------------------------
# BudgetContext
# ---------------------------------------------------------------------------

class TestBudgetContext:

    def test_utilization(self):
        ctx = BudgetContext(token_budget=1000, current_tokens=800)
        assert ctx.utilization == pytest.approx(0.8)

    def test_utilization_zero_budget(self):
        ctx = BudgetContext(token_budget=0, current_tokens=500)
        assert ctx.utilization == 0.0

    def test_headroom(self):
        ctx = BudgetContext(token_budget=1000, current_tokens=800)
        assert ctx.headroom == 200

    def test_headroom_over_budget(self):
        ctx = BudgetContext(token_budget=1000, current_tokens=1200)
        assert ctx.headroom == 0
