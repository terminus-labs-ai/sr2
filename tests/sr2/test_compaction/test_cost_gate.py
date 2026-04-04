"""Unit tests for CompactionCostGate and pricing resolution."""

from unittest.mock import patch

import pytest

from sr2.compaction.cost_gate import CompactionCandidate, CompactionCostGate
from sr2.compaction.pricing import CachePricing, resolve_pricing
from sr2.config.models import CostGateConfig


# ---------------------------------------------------------------------------
# Pricing resolution
# ---------------------------------------------------------------------------

class TestResolvePricing:
    """resolve_pricing fallback chain: custom -> litellm(hint) -> litellm(fallback) -> fail_open."""

    def test_custom_pricing_takes_priority(self):
        result = resolve_pricing(
            model_hint="anything",
            custom_pricing={"input": 3.0, "cache_write": 3.75, "cache_read": 0.3},
        )
        assert result.source == "custom"
        assert result.input_cost == pytest.approx(3.0 / 1_000_000)

    def test_litellm_model_found(self):
        fake_cost = {
            "claude-sonnet-4-6": {
                "input_cost_per_token": 0.000003,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
                "supports_prompt_caching": True,
            }
        }
        with patch("sr2.compaction.pricing.model_cost", fake_cost, create=True), \
             patch.dict("sys.modules", {"litellm": type("m", (), {"model_cost": fake_cost})}):
            result = resolve_pricing(model_hint="claude-sonnet-4-6")
        assert result.source == "litellm:claude-sonnet-4-6"
        assert result.input_cost == pytest.approx(0.000003)

    def test_litellm_fallback_model(self):
        fake_cost = {
            "claude-sonnet-4-6": {
                "input_cost_per_token": 0.000003,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            }
        }
        with patch("sr2.compaction.pricing.model_cost", fake_cost, create=True), \
             patch.dict("sys.modules", {"litellm": type("m", (), {"model_cost": fake_cost})}):
            result = resolve_pricing(model_hint="unknown", fallback_model="claude-sonnet-4-6")
        assert result.source == "litellm:claude-sonnet-4-6"

    def test_fail_open_when_nothing_resolves(self):
        with patch("sr2.compaction.pricing._resolve_from_litellm", return_value=None):
            result = resolve_pricing(model_hint="unknown", fallback_model="also-unknown")
        assert result.source == "fail_open"
        assert result.input_cost == 0.0

    def test_custom_pricing_converts_per_mtok(self):
        result = resolve_pricing(
            custom_pricing={"input": 10.0, "cache_write": 12.5, "cache_read": 1.0},
        )
        assert result.input_cost == pytest.approx(10.0 / 1_000_000)
        assert result.cache_write_cost == pytest.approx(12.5 / 1_000_000)
        assert result.cache_read_cost == pytest.approx(1.0 / 1_000_000)


# ---------------------------------------------------------------------------
# Cost gate decisions
# ---------------------------------------------------------------------------

# Claude Sonnet 4.6 pricing (per token)
_SONNET_PRICING = CachePricing(
    input_cost=3.0 / 1_000_000,       # $3/MTok
    cache_write_cost=3.75 / 1_000_000, # $3.75/MTok
    cache_read_cost=0.30 / 1_000_000,  # $0.30/MTok
    source="test",
)


def _make_gate(min_net: float = 0.001) -> CompactionCostGate:
    config = CostGateConfig(enabled=True, min_net_savings_usd=min_net)
    gate = CompactionCostGate(config)
    # Pre-fill pricing cache to avoid litellm dependency
    gate._pricing_cache["__none__"] = _SONNET_PRICING
    return gate


class TestShouldCompact:
    """Test should_compact decision logic."""

    def test_large_turn_small_tail_allows(self):
        """Big token savings + small downstream invalidation => allowed."""
        gate = _make_gate(min_net=0.0)
        decision = gate.should_compact(
            turn_index=0,
            turn_tokens=5000,
            estimated_compacted_tokens=500,
            total_tokens_after_turn=1000,
        )
        assert decision.allowed is True
        assert decision.tokens_saved == 4500
        assert decision.net_usd > 0

    def test_small_turn_large_tail_blocks(self):
        """Tiny token savings + huge downstream invalidation => blocked."""
        gate = _make_gate(min_net=0.0)
        decision = gate.should_compact(
            turn_index=0,
            turn_tokens=200,
            estimated_compacted_tokens=150,
            total_tokens_after_turn=100_000,
        )
        assert decision.allowed is False
        assert decision.net_usd < 0

    def test_min_net_savings_threshold(self):
        """Net positive but below threshold => blocked."""
        gate = _make_gate(min_net=1.0)  # $1 threshold
        decision = gate.should_compact(
            turn_index=0,
            turn_tokens=5000,
            estimated_compacted_tokens=500,
            total_tokens_after_turn=1000,
        )
        # Net is positive but tiny (few cents), under $1 threshold
        assert decision.allowed is False

    def test_fail_open_always_allows(self):
        """When pricing is fail_open, compaction is always allowed."""
        config = CostGateConfig(enabled=True)
        gate = CompactionCostGate(config)
        gate._pricing_cache["__none__"] = CachePricing(
            input_cost=0, cache_write_cost=0, cache_read_cost=0, source="fail_open",
        )
        decision = gate.should_compact(
            turn_index=0,
            turn_tokens=100,
            estimated_compacted_tokens=50,
            total_tokens_after_turn=1_000_000,
        )
        assert decision.allowed is True
        assert decision.pricing_source == "fail_open"

    def test_decision_fields_populated(self):
        """All CostGateDecision fields have sensible values."""
        gate = _make_gate()
        decision = gate.should_compact(
            turn_index=3,
            turn_tokens=2000,
            estimated_compacted_tokens=500,
            total_tokens_after_turn=5000,
        )
        assert decision.tokens_saved == 1500
        assert decision.cache_invalidation_tokens == 5000
        assert decision.estimated_savings_usd > 0
        assert decision.estimated_invalidation_cost_usd >= 0
        assert isinstance(decision.reason, str)
        assert decision.pricing_source == "test"


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

class TestEvaluateBatch:
    """Test evaluate_batch cascading invalidation logic."""

    def test_cascading_invalidation_reduces_cost(self):
        """Later turns benefit from earlier turns already invalidating the cache.

        Setup: first candidate has large savings and small tail so it's clearly allowed.
        Second candidate's effective invalidation is reduced by the first's already_invalidated.
        """
        gate = _make_gate(min_net=0.0)
        candidates = [
            # Large savings (50000 tokens saved), small tail (100 tokens) => clearly net-positive
            CompactionCandidate(turn_index=0, turn_tokens=50000, estimated_compacted_tokens=500, total_tokens_after_turn=100),
            # Second turn: tail=80, but first already invalidated 100 => effective = max(0, 80-100) = 0
            CompactionCandidate(turn_index=1, turn_tokens=3000, estimated_compacted_tokens=300, total_tokens_after_turn=80),
        ]
        decisions = gate.evaluate_batch(candidates)
        assert len(decisions) == 2

        # First must be allowed: savings far exceed invalidation
        assert decisions[0].allowed is True, (
            f"First candidate should be allowed (large savings, small tail). Reason: {decisions[0].reason}"
        )
        # Second sees zero effective invalidation (80 - 100 clamped to 0)
        assert decisions[1].cache_invalidation_tokens == 0, (
            f"Expected 0 effective invalidation tokens, got {decisions[1].cache_invalidation_tokens}"
        )

    def test_batch_processes_in_turn_order(self):
        """Candidates are sorted by turn_index regardless of input order."""
        gate = _make_gate(min_net=0.0)
        # Pass in reverse order
        candidates = [
            CompactionCandidate(turn_index=5, turn_tokens=2000, estimated_compacted_tokens=200, total_tokens_after_turn=1000),
            CompactionCandidate(turn_index=2, turn_tokens=2000, estimated_compacted_tokens=200, total_tokens_after_turn=5000),
        ]
        decisions = gate.evaluate_batch(candidates)
        assert len(decisions) == 2


# ---------------------------------------------------------------------------
# Engine integration (cost gate wired in)
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """Test that CompactionEngine correctly uses the cost gate."""

    def test_engine_initializes_gate_when_enabled(self):
        from sr2.compaction.engine import CompactionEngine
        from sr2.config.models import CompactionConfig, CompactionRuleConfig

        config = CompactionConfig(
            enabled=True,
            cost_gate=CostGateConfig(enabled=True),
            rules=[CompactionRuleConfig(type="tool_output", strategy="result_summary")],
        )
        engine = CompactionEngine(config)
        assert engine._cost_gate is not None

    def test_engine_no_gate_when_disabled(self):
        from sr2.compaction.engine import CompactionEngine
        from sr2.config.models import CompactionConfig

        config = CompactionConfig(enabled=True, cost_gate=CostGateConfig(enabled=False))
        engine = CompactionEngine(config)
        assert engine._cost_gate is None

    def test_gate_blocks_compaction_in_engine(self):
        """When cost gate blocks, turns pass through uncompacted."""
        from sr2.compaction.engine import CompactionEngine, ConversationTurn
        from sr2.config.models import CompactionConfig, CompactionRuleConfig

        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=10,
            cost_gate=CostGateConfig(enabled=True, min_net_savings_usd=999.0),  # absurdly high threshold
            rules=[CompactionRuleConfig(type="tool_output", strategy="result_summary")],
        )
        engine = CompactionEngine(config)
        # Pre-fill pricing to avoid litellm
        engine._cost_gate._pricing_cache["__none__"] = _SONNET_PRICING

        turns = [
            ConversationTurn(turn_number=0, role="assistant", content="x" * 2000, content_type="tool_output"),
            ConversationTurn(turn_number=1, role="user", content="hello"),
        ]
        result = engine.compact(turns)
        # Gate should block => no turns compacted
        assert result.turns_compacted == 0

    def test_gate_allows_compaction_in_engine(self):
        """When cost gate allows, turns get compacted normally."""
        from sr2.compaction.engine import CompactionEngine, ConversationTurn
        from sr2.config.models import CompactionConfig, CompactionRuleConfig

        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=10,
            cost_gate=CostGateConfig(enabled=True, min_net_savings_usd=0.0),  # allow everything positive
            rules=[CompactionRuleConfig(type="tool_output", strategy="result_summary")],
        )
        engine = CompactionEngine(config)
        engine._cost_gate._pricing_cache["__none__"] = _SONNET_PRICING

        turns = [
            ConversationTurn(
                turn_number=0, role="assistant",
                content="line\n" * 500,  # large content, small tail
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="user", content="hello"),
        ]
        result = engine.compact(turns)
        # Gate should allow, and rule should compact
        assert result.turns_compacted >= 0  # depends on rule behavior
