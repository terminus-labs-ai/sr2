"""Tests for compaction detail tracking (TurnCompactionDetail, CostGateResult)."""

import pytest

from sr2.compaction.engine import (
    CompactionEngine,
    CompactionResult,
    ConversationTurn,
    CostGateResult,
    TurnCompactionDetail,
)
from sr2.config.models import CompactionConfig, CompactionRuleConfig, CostGateConfig


def _long_content(lines: int = 20) -> str:
    """Generate content long enough to trigger compaction."""
    return "\n".join(f"data line {i}: some verbose output" for i in range(lines))


def _make_config(
    raw_window: int = 2,
    min_content_size: int = 10,
    cost_gate: bool = False,
) -> CompactionConfig:
    """Create a CompactionConfig with schema_and_sample rule."""
    return CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=min_content_size,
        cost_gate=CostGateConfig(enabled=cost_gate) if cost_gate else CostGateConfig(enabled=False),
        rules=[
            CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
        ],
    )


class TestCostGateResultDataclass:
    """CostGateResult can be created and has the expected fields."""

    def test_creation(self):
        result = CostGateResult(
            passed=True,
            token_savings_usd=0.005,
            cache_invalidation_usd=0.001,
            net_savings_usd=0.004,
        )
        assert result.passed is True
        assert result.token_savings_usd == 0.005
        assert result.cache_invalidation_usd == 0.001
        assert result.net_savings_usd == 0.004

    def test_blocked(self):
        result = CostGateResult(
            passed=False,
            token_savings_usd=0.001,
            cache_invalidation_usd=0.01,
            net_savings_usd=-0.009,
        )
        assert result.passed is False
        assert result.net_savings_usd < 0


class TestTurnCompactionDetailDataclass:
    """TurnCompactionDetail can be created and has the expected fields."""

    def test_creation_with_rule(self):
        detail = TurnCompactionDetail(
            turn_number=3,
            role="assistant",
            content_type="tool_output",
            rule_applied="schema_and_sample",
            original_tokens=500,
            compacted_tokens=100,
            original_content="long content here",
            compacted_content="short",
        )
        assert detail.turn_number == 3
        assert detail.rule_applied == "schema_and_sample"
        assert detail.original_tokens == 500
        assert detail.compacted_tokens == 100

    def test_creation_blocked_by_cost_gate(self):
        detail = TurnCompactionDetail(
            turn_number=5,
            role="tool_result",
            content_type="tool_output",
            rule_applied=None,
            original_tokens=200,
            compacted_tokens=200,
            original_content="unchanged",
            compacted_content="unchanged",
        )
        assert detail.rule_applied is None
        assert detail.original_tokens == detail.compacted_tokens


class TestCompactionResultBackwardCompatibility:
    """CompactionResult can be created without the new fields."""

    def test_without_new_fields(self):
        result = CompactionResult(
            turns=[],
            original_tokens=100,
            compacted_tokens=80,
            turns_compacted=1,
        )
        assert result.details == []
        assert result.cost_gate_result is None

    def test_with_analysis_only(self):
        result = CompactionResult(
            turns=[],
            original_tokens=100,
            compacted_tokens=80,
            turns_compacted=1,
            analysis={"key": "value"},
        )
        assert result.analysis == {"key": "value"}
        assert result.details == []
        assert result.cost_gate_result is None

    def test_with_details_populated(self):
        detail = TurnCompactionDetail(
            turn_number=0,
            role="assistant",
            content_type="tool_output",
            rule_applied="schema_and_sample",
            original_tokens=500,
            compacted_tokens=100,
            original_content="original",
            compacted_content="compacted",
        )
        gate = CostGateResult(
            passed=True,
            token_savings_usd=0.01,
            cache_invalidation_usd=0.002,
            net_savings_usd=0.008,
        )
        result = CompactionResult(
            turns=[],
            original_tokens=500,
            compacted_tokens=100,
            turns_compacted=1,
            details=[detail],
            cost_gate_result=gate,
        )
        assert len(result.details) == 1
        assert result.details[0].rule_applied == "schema_and_sample"
        assert result.cost_gate_result is not None
        assert result.cost_gate_result.passed is True


class TestCompactPopulatesDetails:
    """Integration: CompactionEngine.compact() populates details correctly."""

    def test_single_turn_compacted_has_detail(self):
        """A single compacted turn produces one detail entry."""
        config = _make_config(raw_window=1)
        engine = CompactionEngine(config)
        original_content = _long_content()
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=original_content,
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        assert len(result.details) == 1

        detail = result.details[0]
        assert detail.turn_number == 0
        assert detail.role == "assistant"
        assert detail.content_type == "tool_output"
        assert detail.rule_applied == "schema_and_sample"
        assert detail.original_tokens == len(original_content) // 4
        assert detail.compacted_tokens < detail.original_tokens
        assert detail.original_content == original_content
        assert detail.compacted_content == result.turns[0].content

    def test_multiple_turns_compacted(self):
        """Multiple compacted turns produce multiple detail entries."""
        config = _make_config(raw_window=1)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(
                turn_number=1, role="assistant", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 2
        assert len(result.details) == 2
        assert result.details[0].turn_number == 0
        assert result.details[1].turn_number == 1

    def test_skipped_turns_have_no_detail(self):
        """Turns skipped (user message, already compacted, no rule) produce no detail."""
        config = _make_config(raw_window=1)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="user", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 0
        assert len(result.details) == 0

    def test_within_raw_window_no_details(self):
        """Turns entirely within raw_window produce no details."""
        config = _make_config(raw_window=5)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(turn_number=i, role="assistant", content="short")
            for i in range(3)
        ]
        result = engine.compact(turns)

        assert len(result.details) == 0

    def test_no_cost_gate_result_when_gate_disabled(self):
        """cost_gate_result is None when cost gate is not enabled."""
        config = _make_config(raw_window=1, cost_gate=False)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        assert result.cost_gate_result is None

    def test_cost_gate_allowed_populates_result(self):
        """When cost gate allows compaction, cost_gate_result is populated with passed=True."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=10,
            cost_gate=CostGateConfig(
                enabled=True,
                fallback_model="claude-sonnet-4-20250514",
                min_net_savings_usd=0.0,  # Very permissive so it always allows
            ),
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
            ],
        )
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(50),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        assert result.cost_gate_result is not None
        assert result.cost_gate_result.passed is True
        assert len(result.details) == 1
        assert result.details[0].rule_applied == "schema_and_sample"

    def test_cost_gate_blocked_populates_result_and_detail(self):
        """When cost gate blocks compaction, detail has rule_applied=None and cost_gate_result.passed=False."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=10,
            cost_gate=CostGateConfig(
                enabled=True,
                fallback_model="claude-sonnet-4-20250514",
                min_net_savings_usd=999999.0,  # Absurdly high threshold blocks everything
            ),
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
            ],
        )
        engine = CompactionEngine(config)
        original_content = _long_content(50)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=original_content,
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 0
        assert result.cost_gate_result is not None
        assert result.cost_gate_result.passed is False
        assert len(result.details) == 1

        detail = result.details[0]
        assert detail.rule_applied is None
        assert detail.original_tokens == detail.compacted_tokens
        assert detail.original_content == original_content
        assert detail.compacted_content == original_content
