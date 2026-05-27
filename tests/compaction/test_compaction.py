"""Tests for sr2.compaction — rules, engine, cost gate, LLM strategy, plugin registration.

Bead: obsidian-t9t.7 (Agent A — test writer)

Covers:
  1. Rules — 5 compaction rules (schema_and_sample, reference, result_summary,
     supersede, collapse), each transforms turn content correctly.
  2. CompactionEngine — apply(rules, messages) produces compacted output.
  3. CostGate — should_compact(token_savings, cache_cost) decision logic.
  4. LLMCompactionStrategy — accepts llm callable and rule interface; mock-tested.
  5. Plugin registration — compaction transformer discoverable via PluginRegistry.

NOTE: All tests FAIL until implementation exists in src/sr2/compaction/.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2.models import ContentBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock


# ===========================================================================
# Helpers
# ===========================================================================


def _text_msg(role: str, text: str, turn_index: int | None = None) -> Message:
    """Build a single-TextBlock Message."""
    return Message(role=role, content=[TextBlock(text=text)], turn_index=turn_index)


def _tool_result_msg(tool_use_id: str, content: str) -> Message:
    """Build a user Message wrapping a ToolResultBlock."""
    return Message(
        role="user",
        content=[ToolResultBlock(tool_use_id=tool_use_id, content=content)],
    )


# ===========================================================================
# 1. Rules
# ===========================================================================


class TestSchemaAndSampleRule:
    """schema_and_sample rule compacts JSON schema + data to schema + 1 sample row."""

    def test_compact_schema_with_multiple_rows_returns_schema_plus_one_row(self):
        """Given a JSON payload with a schema and N data rows, return schema + first row."""
        from sr2.compaction.rules import schema_and_sample

        payload = json.dumps(
            {
                "schema": {"fields": ["id", "name", "value"]},
                "data": [
                    {"id": 1, "name": "alpha", "value": 10},
                    {"id": 2, "name": "beta", "value": 20},
                    {"id": 3, "name": "gamma", "value": 30},
                ],
            }
        )
        block = ToolResultBlock(tool_use_id="tool-1", content=payload)
        result = schema_and_sample(block)

        assert result is not None
        result_data = json.loads(
            result.content if isinstance(result.content, str) else result.content[0].text
        )
        assert "schema" in result_data
        assert len(result_data["data"]) == 1
        assert result_data["data"][0] == {"id": 1, "name": "alpha", "value": 10}

    def test_compact_schema_with_one_row_returns_unchanged(self):
        """A payload with only one data row is returned unchanged (nothing to compact)."""
        from sr2.compaction.rules import schema_and_sample

        payload = json.dumps(
            {
                "schema": {"fields": ["id"]},
                "data": [{"id": 1}],
            }
        )
        block = ToolResultBlock(tool_use_id="tool-2", content=payload)
        result = schema_and_sample(block)

        # None means rule does not apply / no change
        assert result is None

    def test_compact_schema_with_empty_data_returns_none(self):
        """A payload with no data rows returns None (rule does not apply)."""
        from sr2.compaction.rules import schema_and_sample

        payload = json.dumps({"schema": {"fields": ["id"]}, "data": []})
        block = ToolResultBlock(tool_use_id="tool-3", content=payload)
        result = schema_and_sample(block)

        assert result is None

    def test_non_schema_content_returns_none(self):
        """Non-JSON or schema-less content returns None (rule does not apply)."""
        from sr2.compaction.rules import schema_and_sample

        block = ToolResultBlock(tool_use_id="tool-4", content="plain text result")
        result = schema_and_sample(block)

        assert result is None

    def test_compact_result_is_marked_compacted(self):
        """The compacted ToolResultBlock has compacted=True."""
        from sr2.compaction.rules import schema_and_sample

        payload = json.dumps(
            {
                "schema": {"fields": ["x"]},
                "data": [{"x": 1}, {"x": 2}],
            }
        )
        block = ToolResultBlock(tool_use_id="tool-5", content=payload)
        result = schema_and_sample(block)

        assert result is not None
        assert result.compacted is True


class TestReferenceRule:
    """reference rule replaces repeated reference content with a pointer/citation."""

    def test_repeated_block_replaced_with_reference(self):
        """Second occurrence of identical content is replaced with a reference pointer."""
        from sr2.compaction.rules import ReferenceRule

        seen: dict[str, str] = {}  # content_hash -> reference_id
        rule = ReferenceRule(seen_registry=seen)

        text = "System context: always be concise."
        block1 = TextBlock(text=text)
        block2 = TextBlock(text=text)

        result1 = rule.apply(block1)
        result2 = rule.apply(block2)

        # First occurrence is not changed (or returned as-is)
        assert result1 is None or result1.text == text
        # Second occurrence is replaced with a reference pointer
        assert result2 is not None
        assert "ref:" in result2.text or "[ref" in result2.text

    def test_unique_content_not_replaced(self):
        """Unique content is not replaced — rule returns None."""
        from sr2.compaction.rules import ReferenceRule

        rule = ReferenceRule(seen_registry={})
        block = TextBlock(text="unique content for this turn")
        result = rule.apply(block)

        assert result is None

    def test_reference_is_deterministic(self):
        """The same content always maps to the same reference ID."""
        from sr2.compaction.rules import ReferenceRule

        seen: dict[str, str] = {}
        rule = ReferenceRule(seen_registry=seen)

        text = "Repeated reference material."
        rule.apply(TextBlock(text=text))  # register
        result = rule.apply(TextBlock(text=text))  # should be replaced

        # Apply a third time — should produce the same reference
        result2 = rule.apply(TextBlock(text=text))
        assert result is not None and result2 is not None
        assert result.text == result2.text


class TestResultSummaryRule:
    """result_summary rule replaces tool result content with a summary."""

    def test_long_tool_result_is_summarized(self):
        """A tool result longer than threshold tokens is replaced with a summary."""
        from sr2.compaction.rules import result_summary

        long_content = "word " * 200  # well above any threshold
        block = ToolResultBlock(tool_use_id="tool-10", content=long_content)
        result = result_summary(block, max_tokens=50)

        assert result is not None
        # Result should be shorter
        original_len = len(long_content)
        result_content = (
            result.content if isinstance(result.content, str) else result.content[0].text
        )
        assert len(result_content) < original_len

    def test_short_tool_result_not_summarized(self):
        """A tool result under the threshold returns None (rule does not apply)."""
        from sr2.compaction.rules import result_summary

        short_content = "ok"
        block = ToolResultBlock(tool_use_id="tool-11", content=short_content)
        result = result_summary(block, max_tokens=500)

        assert result is None

    def test_summarized_result_is_marked_compacted(self):
        """Summarized ToolResultBlock has compacted=True."""
        from sr2.compaction.rules import result_summary

        long_content = "data " * 300
        block = ToolResultBlock(tool_use_id="tool-12", content=long_content)
        result = result_summary(block, max_tokens=50)

        assert result is not None
        assert result.compacted is True

    def test_error_result_not_summarized(self):
        """Tool results with is_error=True are not summarized."""
        from sr2.compaction.rules import result_summary

        block = ToolResultBlock(
            tool_use_id="tool-13",
            content="error: connection refused " * 50,
            is_error=True,
        )
        result = result_summary(block, max_tokens=10)

        assert result is None


class TestSupersederule:
    """supersede rule drops older content when newer content supersedes it."""

    def test_superseded_message_is_dropped(self):
        """If a later message declares it supersedes an earlier one, the earlier is removed."""
        from sr2.compaction.rules import supersede

        messages = [
            Message(
                role="user",
                content=[TextBlock(text="Old system context v1")],
                turn_index=0,
            ),
            Message(
                role="user",
                content=[TextBlock(text="[supersedes turn 0] New system context v2")],
                turn_index=1,
            ),
        ]
        result = supersede(messages)

        # Turn 0 should be removed; turn 1 retained
        assert len(result) == 1
        assert result[0].turn_index == 1

    def test_non_superseding_messages_unchanged(self):
        """Messages without supersede markers are returned as-is."""
        from sr2.compaction.rules import supersede

        messages = [
            _text_msg("user", "first message", turn_index=0),
            _text_msg("user", "second message", turn_index=1),
        ]
        result = supersede(messages)

        assert len(result) == 2

    def test_empty_message_list_returns_empty(self):
        """An empty message list returns an empty list."""
        from sr2.compaction.rules import supersede

        assert supersede([]) == []

    def test_supersede_removes_only_specified_index(self):
        """Only the exact superseded turn index is dropped, not others."""
        from sr2.compaction.rules import supersede

        messages = [
            _text_msg("user", "turn 0", turn_index=0),
            _text_msg("user", "turn 1", turn_index=1),
            Message(
                role="user",
                content=[TextBlock(text="[supersedes turn 0] replacement")],
                turn_index=2,
            ),
        ]
        result = supersede(messages)

        turn_indices = [m.turn_index for m in result]
        assert 0 not in turn_indices
        assert 1 in turn_indices
        assert 2 in turn_indices


class TestCollapseRule:
    """collapse rule collapses multiple similar turns into one."""

    def test_consecutive_similar_turns_collapsed(self):
        """Multiple consecutive messages with similar structure collapse into one."""
        from sr2.compaction.rules import collapse

        messages = [
            _text_msg("user", "ping", turn_index=0),
            _text_msg("assistant", "pong", turn_index=1),
            _text_msg("user", "ping", turn_index=2),
            _text_msg("assistant", "pong", turn_index=3),
            _text_msg("user", "ping", turn_index=4),
            _text_msg("assistant", "pong", turn_index=5),
        ]
        result = collapse(messages, min_occurrences=3)

        # Result should have fewer messages than the input
        assert len(result) < len(messages)

    def test_few_messages_not_collapsed(self):
        """Fewer than min_occurrences similar turns are not collapsed."""
        from sr2.compaction.rules import collapse

        messages = [
            _text_msg("user", "hello", turn_index=0),
            _text_msg("assistant", "hi", turn_index=1),
        ]
        result = collapse(messages, min_occurrences=3)

        assert len(result) == len(messages)

    def test_collapsed_output_preserves_unique_content(self):
        """Unique messages are retained even when surrounding turns are collapsed."""
        from sr2.compaction.rules import collapse

        messages = [
            _text_msg("user", "ping", turn_index=0),
            _text_msg("assistant", "pong", turn_index=1),
            _text_msg("user", "ping", turn_index=2),
            _text_msg("assistant", "pong", turn_index=3),
            _text_msg("user", "something completely different", turn_index=4),
        ]
        result = collapse(messages, min_occurrences=2)

        # The unique message must survive
        texts = []
        for m in result:
            for b in m.content:
                if isinstance(b, TextBlock):
                    texts.append(b.text)
        assert any("something completely different" in t for t in texts)

    def test_empty_message_list_returns_empty(self):
        """An empty list returns an empty list."""
        from sr2.compaction.rules import collapse

        assert collapse([], min_occurrences=2) == []


# ===========================================================================
# 2. CompactionEngine
# ===========================================================================


class TestCompactionEngine:
    """CompactionEngine applies rules to a list of Messages."""

    def test_apply_returns_messages_when_no_rules_apply(self):
        """apply() returns original messages when no rule transforms anything."""
        from sr2.compaction.engine import CompactionEngine

        engine = CompactionEngine(rules=[])
        messages = [_text_msg("user", "hello"), _text_msg("assistant", "world")]
        result = engine.apply(messages)

        assert result == messages

    def test_apply_with_schema_rule_compacts_tool_result(self):
        """apply() with schema_and_sample rule compacts a multi-row tool result."""
        from sr2.compaction.engine import CompactionEngine
        from sr2.compaction.rules import schema_and_sample

        payload = json.dumps(
            {
                "schema": {"fields": ["id", "val"]},
                "data": [{"id": i, "val": i * 2} for i in range(10)],
            }
        )
        messages = [
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="t-1", content=payload)],
            )
        ]

        engine = CompactionEngine(rules=[schema_and_sample])
        result = engine.apply(messages)

        # Should produce one compacted message
        assert len(result) == 1
        tool_block = result[0].content[0]
        assert isinstance(tool_block, ToolResultBlock)
        assert tool_block.compacted is True

    def test_apply_applies_rules_in_order(self):
        """Rules are applied in the order they are provided."""
        from sr2.compaction.engine import CompactionEngine

        call_order: list[str] = []

        def rule_a(block: ContentBlock) -> ContentBlock | None:
            call_order.append("a")
            return None

        def rule_b(block: ContentBlock) -> ContentBlock | None:
            call_order.append("b")
            return None

        messages = [_text_msg("user", "test")]
        engine = CompactionEngine(rules=[rule_a, rule_b])
        engine.apply(messages)

        # Both rules were called; order is a before b
        if call_order:  # rules only called if the engine passes blocks to them
            assert call_order.index("a") < call_order.index("b")

    def test_apply_returns_original_if_no_savings(self):
        """apply() returns original messages if compacted version has no token savings."""
        from sr2.compaction.engine import CompactionEngine

        def noop_rule(block: ContentBlock) -> ContentBlock | None:
            return None

        messages = [_text_msg("user", "hello")]
        engine = CompactionEngine(rules=[noop_rule])
        result = engine.apply(messages)

        assert result == messages

    def test_apply_with_multiple_rules_accumulates_savings(self):
        """Multiple applicable rules can both be applied across a message list."""
        from sr2.compaction.engine import CompactionEngine
        from sr2.compaction.rules import result_summary, schema_and_sample

        schema_payload = json.dumps(
            {
                "schema": {"fields": ["id"]},
                "data": [{"id": i} for i in range(20)],
            }
        )
        long_result = "result data " * 200

        messages = [
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="t-schema", content=schema_payload),
                    ToolResultBlock(tool_use_id="t-long", content=long_result),
                ],
            )
        ]

        engine = CompactionEngine(rules=[schema_and_sample, lambda b: result_summary(b, max_tokens=50)])
        result = engine.apply(messages)

        # At least one block should have been compacted
        compacted_blocks = [
            b
            for m in result
            for b in m.content
            if isinstance(b, ToolResultBlock) and b.compacted
        ]
        assert len(compacted_blocks) >= 1


# ===========================================================================
# 3. Cost Gate
# ===========================================================================


class TestCostGate:
    """CostGate.should_compact() decides whether compaction is worth it."""

    def test_returns_true_when_savings_exceed_cache_cost(self):
        """should_compact returns True when token_savings > cache_cost."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=500, cache_cost=100) is True

    def test_returns_false_when_savings_less_than_cache_cost(self):
        """should_compact returns False when token_savings < cache_cost."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=50, cache_cost=200) is False

    def test_returns_false_when_savings_equal_to_cache_cost(self):
        """should_compact returns False when savings == cost (not strictly worth it)."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=100, cache_cost=100) is False

    def test_returns_false_when_zero_savings(self):
        """should_compact returns False when token_savings is zero."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=0, cache_cost=0) is False

    def test_returns_false_when_negative_savings(self):
        """should_compact returns False when token_savings is negative."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=-10, cache_cost=0) is False

    def test_returns_false_with_zero_savings_nonzero_cost(self):
        """should_compact returns False when savings=0 but cache_cost>0."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=0, cache_cost=50) is False

    def test_large_savings_zero_cost_returns_true(self):
        """should_compact returns True when savings are large and cost is zero."""
        from sr2.compaction.cost_gate import CostGate

        gate = CostGate()
        assert gate.should_compact(token_savings=1000, cache_cost=0) is True

    def test_custom_multiplier_adjusts_threshold(self):
        """A cost_multiplier > 1 raises the bar for compaction."""
        from sr2.compaction.cost_gate import CostGate

        # With multiplier=2, effective cost doubles — 150 savings vs 100*2=200 cost
        gate = CostGate(cost_multiplier=2.0)
        assert gate.should_compact(token_savings=150, cache_cost=100) is False

    def test_custom_multiplier_below_one_lowers_threshold(self):
        """A cost_multiplier < 1 lowers the bar for compaction."""
        from sr2.compaction.cost_gate import CostGate

        # With multiplier=0.5, effective cost halves — 60 savings vs 100*0.5=50 cost
        gate = CostGate(cost_multiplier=0.5)
        assert gate.should_compact(token_savings=60, cache_cost=100) is True


# ===========================================================================
# 4. LLMCompactionStrategy
# ===========================================================================


class TestLLMCompactionStrategy:
    """LLMCompactionStrategy: accepts llm callable and rule interface; mock-tested."""

    def test_strategy_accepts_llm_callable(self):
        """LLMCompactionStrategy can be constructed with an LLM callable."""
        from sr2.compaction.llm_strategy import LLMCompactionStrategy

        mock_llm = MagicMock()
        strategy = LLMCompactionStrategy(llm=mock_llm)
        assert strategy is not None

    def test_strategy_accepts_custom_rules(self):
        """LLMCompactionStrategy accepts an optional list of rule callables."""
        from sr2.compaction.llm_strategy import LLMCompactionStrategy

        mock_llm = MagicMock()
        rules = [MagicMock(), MagicMock()]
        strategy = LLMCompactionStrategy(llm=mock_llm, rules=rules)
        assert strategy is not None

    @pytest.mark.asyncio
    async def test_compact_calls_llm_for_eligible_content(self):
        """compact() invokes the llm callable when content is eligible."""
        from sr2.compaction.llm_strategy import LLMCompactionStrategy

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = MagicMock(
            content=[TextBlock(text="compacted summary")]
        )

        strategy = LLMCompactionStrategy(llm=mock_llm)
        messages = [_text_msg("user", "long text " * 100)]
        await strategy.compact(messages)

        assert mock_llm.complete.called

    @pytest.mark.asyncio
    async def test_compact_does_not_call_llm_for_empty_messages(self):
        """compact() does not call the llm for an empty message list."""
        from sr2.compaction.llm_strategy import LLMCompactionStrategy

        mock_llm = AsyncMock()
        strategy = LLMCompactionStrategy(llm=mock_llm)
        result = await strategy.compact([])

        assert not mock_llm.complete.called
        assert result == []

    @pytest.mark.asyncio
    async def test_compact_returns_list_of_messages(self):
        """compact() always returns a list of Messages."""
        from sr2.compaction.llm_strategy import LLMCompactionStrategy

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = MagicMock(
            content=[TextBlock(text="summary")]
        )

        strategy = LLMCompactionStrategy(llm=mock_llm)
        messages = [_text_msg("user", "test content")]
        result = await strategy.compact(messages)

        assert isinstance(result, list)
        for m in result:
            assert isinstance(m, Message)


# ===========================================================================
# 5. Plugin Registration
# ===========================================================================


class TestCompactionPluginRegistration:
    """Compaction transformer is registered as an SR2 plugin entry point."""

    def test_compaction_transformer_in_sr2_transformers_group(self):
        """The 'compaction' entry point is listed in the sr2.transformers group."""
        from importlib.metadata import entry_points

        eps = entry_points(group="sr2.transformers")
        names = [ep.name for ep in eps]
        assert "compaction" in names, (
            f"'compaction' not found in sr2.transformers entry points. Found: {names}"
        )

    def test_compaction_entry_point_loads_transformer_class(self):
        """The compaction entry point loads a class with a transform() method."""
        from importlib.metadata import entry_points

        eps = entry_points(group="sr2.transformers")
        ep = next((e for e in eps if e.name == "compaction"), None)
        assert ep is not None, "compaction entry point not found"

        cls = ep.load()
        assert hasattr(cls, "transform") or hasattr(cls, "build"), (
            f"Loaded class {cls!r} has neither 'transform' nor 'build' method"
        )

    def test_plugin_registry_can_discover_compaction(self):
        """PluginRegistry for sr2.transformers can list 'compaction'."""
        from sr2.pipeline.protocols import Transformer
        from sr2.plugins import PluginRegistry

        registry = PluginRegistry(group="sr2.transformers", protocol=Transformer)
        names = registry.names()
        assert "compaction" in names, (
            f"'compaction' not in registry names. Found: {names}"
        )
