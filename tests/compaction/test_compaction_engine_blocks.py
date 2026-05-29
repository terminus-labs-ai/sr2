"""Tests for CompactionEngine.apply_to_blocks() and the updated CompactionTransformer.

Bead: sr2-61 (Agent A — test writer)

These tests verify the fix for the wrap/unwrap impedance between CompactionTransformer
and CompactionEngine:

  1. apply_to_blocks returns blocks unchanged when no rules apply.
  2. apply_to_blocks returns transformed blocks when a rule applies.
  3. apply_to_blocks handles an empty block list without error.
  4. CompactionTransformer.transform returns TransformationResult(content=None) when
     blocks are unchanged.
  5. CompactionTransformer.transform returns TransformationResult with compacted content
     when blocks change.
  6. CompactionTransformer.transform does NOT perform Message wrapping internally —
     it calls apply_to_blocks on the engine (observable via interface).

All tests FAIL until the implementation adds apply_to_blocks to CompactionEngine and
updates CompactionTransformer.transform to use it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sr2.models import ContentBlock, TextBlock, ToolResultBlock


# ===========================================================================
# Helpers
# ===========================================================================


def _make_schema_block(tool_use_id: str, num_rows: int) -> ToolResultBlock:
    """Build a ToolResultBlock containing a JSON schema+data payload."""
    payload = json.dumps(
        {
            "schema": {"fields": ["id", "val"]},
            "data": [{"id": i, "val": i * 10} for i in range(num_rows)],
        }
    )
    return ToolResultBlock(tool_use_id=tool_use_id, content=payload)


def _always_applies_rule(block: ContentBlock) -> ContentBlock | None:
    """A rule that always replaces any block with a sentinel TextBlock."""
    return TextBlock(text="[compacted]")


def _never_applies_rule(block: ContentBlock) -> ContentBlock | None:
    """A rule that never applies."""
    return None


# ===========================================================================
# 1-3: CompactionEngine.apply_to_blocks
# ===========================================================================


class TestCompactionEngineApplyToBlocks:
    """CompactionEngine.apply_to_blocks(blocks) processes content blocks directly."""

    def test_returns_original_blocks_when_no_rules_apply(self):
        """When no rule matches, apply_to_blocks returns the original block list unchanged."""
        from sr2.compaction.engine import CompactionEngine

        engine = CompactionEngine(rules=[_never_applies_rule])
        blocks: list[ContentBlock] = [
            TextBlock(text="hello"),
            TextBlock(text="world"),
        ]
        result = engine.apply_to_blocks(blocks)

        # Identity: same object or equal content, no transformation
        assert result == blocks

    def test_returns_transformed_blocks_when_rule_applies(self):
        """When a rule matches, apply_to_blocks returns a new list with the replacement block."""
        from sr2.compaction.engine import CompactionEngine

        engine = CompactionEngine(rules=[_always_applies_rule])
        blocks: list[ContentBlock] = [TextBlock(text="original content")]
        result = engine.apply_to_blocks(blocks)

        assert len(result) == 1
        assert isinstance(result[0], TextBlock)
        assert result[0].text == "[compacted]"  # type: ignore[union-attr]

    def test_returns_transformed_blocks_using_real_schema_rule(self):
        """apply_to_blocks compacts a multi-row schema block via the schema_and_sample rule."""
        from sr2.compaction.engine import CompactionEngine
        from sr2.compaction.rules import schema_and_sample

        engine = CompactionEngine(rules=[schema_and_sample])
        blocks: list[ContentBlock] = [_make_schema_block("t-1", num_rows=5)]
        result = engine.apply_to_blocks(blocks)

        assert len(result) == 1
        block = result[0]
        assert isinstance(block, ToolResultBlock)
        assert block.compacted is True
        data = json.loads(block.content)  # type: ignore[arg-type]
        assert len(data["data"]) == 1

    def test_empty_block_list_returns_empty_without_error(self):
        """apply_to_blocks([]) returns an empty list and does not raise."""
        from sr2.compaction.engine import CompactionEngine

        engine = CompactionEngine(rules=[_always_applies_rule])
        result = engine.apply_to_blocks([])

        assert result == []

    def test_multiple_blocks_partial_match(self):
        """Only matching blocks are transformed; non-matching blocks are passed through."""
        from sr2.compaction.engine import CompactionEngine
        from sr2.compaction.rules import schema_and_sample

        # First block: compactable (5 rows). Second block: plain text, not compactable.
        blocks: list[ContentBlock] = [
            _make_schema_block("t-2", num_rows=5),
            TextBlock(text="plain text, not a schema block"),
        ]
        engine = CompactionEngine(rules=[schema_and_sample])
        result = engine.apply_to_blocks(blocks)

        assert len(result) == 2
        # First block was transformed
        assert isinstance(result[0], ToolResultBlock)
        assert result[0].compacted is True  # type: ignore[union-attr]
        # Second block is unchanged
        assert isinstance(result[1], TextBlock)
        assert result[1].text == "plain text, not a schema block"  # type: ignore[union-attr]


# ===========================================================================
# 4-5: CompactionTransformer.transform integration
# ===========================================================================


class TestCompactionTransformerUsesApplyToBlocks:
    """CompactionTransformer.transform() uses apply_to_blocks, not Message wrapping."""

    @pytest.mark.asyncio
    async def test_transform_returns_none_content_when_blocks_unchanged(self):
        """transform() returns TransformationResult(content=None) when no rules fire."""
        from sr2.compaction.transformer import CompactionTransformer
        from sr2.config.models import TransformerConfig

        config = TransformerConfig(
            type="sr2.compaction.transformer:CompactionTransformer",
            name="compaction",
            config={"max_result_tokens": 500},
        )
        transformer = CompactionTransformer(config)

        # A plain TextBlock — no compaction rule matches this
        blocks: list[ContentBlock] = [TextBlock(text="short plain text")]
        result = await transformer.transform(blocks, events=[])

        assert result.content is None

    @pytest.mark.asyncio
    async def test_transform_returns_compacted_content_when_rules_fire(self):
        """transform() returns TransformationResult with new content when compaction applies."""
        from sr2.compaction.transformer import CompactionTransformer
        from sr2.config.models import TransformerConfig

        config = TransformerConfig(
            type="sr2.compaction.transformer:CompactionTransformer",
            name="compaction",
            config={"max_result_tokens": 500},
        )
        transformer = CompactionTransformer(config)

        # A multi-row schema block — schema_and_sample rule will fire
        blocks: list[ContentBlock] = [_make_schema_block("t-3", num_rows=10)]
        result = await transformer.transform(blocks, events=[])

        assert result.content is not None
        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolResultBlock)
        assert result.content[0].compacted is True  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_transform_result_carries_transformer_name(self):
        """TransformationResult.transformer_name is 'compaction'."""
        from sr2.compaction.transformer import CompactionTransformer
        from sr2.config.models import TransformerConfig

        config = TransformerConfig(
            type="sr2.compaction.transformer:CompactionTransformer",
            name="compaction",
        )
        transformer = CompactionTransformer(config)
        result = await transformer.transform([TextBlock(text="x")], events=[])

        assert result.transformer_name == "compaction"


# ===========================================================================
# 6: Interface contract — transformer does NOT do Message wrapping internally
# ===========================================================================


class TestTransformerDoesNotWrapMessages:
    """The transformer's transform() must not construct Message objects internally.

    The behavioral contract: after the fix, CompactionTransformer calls
    engine.apply_to_blocks() rather than wrapping blocks into a Message and
    calling engine.apply(). We verify this at the interface level by confirming
    apply_to_blocks is called (and apply is NOT called) on the engine.
    """

    @pytest.mark.asyncio
    async def test_transform_calls_apply_to_blocks_not_apply(self):
        """transform() invokes engine.apply_to_blocks(), not engine.apply()."""
        from sr2.compaction.engine import CompactionEngine
        from sr2.compaction.transformer import CompactionTransformer
        from sr2.config.models import TransformerConfig

        config = TransformerConfig(
            type="sr2.compaction.transformer:CompactionTransformer",
            name="compaction",
        )
        transformer = CompactionTransformer(config)

        # Replace the engine with a spy (no spec: apply_to_blocks doesn't exist yet pre-fix)
        mock_engine = MagicMock()
        mock_engine.apply_to_blocks.return_value = [TextBlock(text="unchanged")]
        transformer._engine = mock_engine  # noqa: SLF001  # direct injection for test spy

        blocks: list[ContentBlock] = [TextBlock(text="test")]
        await transformer.transform(blocks, events=[])

        # apply_to_blocks must have been called with the blocks directly
        mock_engine.apply_to_blocks.assert_called_once_with(blocks)

        # apply() (the Message-based method) must NOT have been called
        mock_engine.apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_transform_passes_blocks_directly_to_engine(self):
        """transform() passes the content list directly to the engine, not wrapped in a Message.

        Verified by confirming the argument received by apply_to_blocks is the
        original blocks list (same identity), not a Message's .content attribute.
        """
        from sr2.compaction.transformer import CompactionTransformer
        from sr2.config.models import TransformerConfig

        config = TransformerConfig(
            type="sr2.compaction.transformer:CompactionTransformer",
            name="compaction",
        )
        transformer = CompactionTransformer(config)

        mock_engine = MagicMock()
        mock_engine.apply_to_blocks.return_value = [TextBlock(text="hello")]
        transformer._engine = mock_engine  # noqa: SLF001

        blocks: list[ContentBlock] = [TextBlock(text="hello")]
        await transformer.transform(blocks, events=[])

        # The argument must equal the original blocks, not something extracted from a
        # Message wrapper. We check equality (not identity) so a correct implementation
        # that passes `content` directly is not penalised if it happens to copy the list.
        call_args = mock_engine.apply_to_blocks.call_args
        assert call_args is not None, "apply_to_blocks was never called"
        passed_blocks = call_args[0][0]
        assert passed_blocks == blocks, (
            "transform() must pass the content blocks directly to apply_to_blocks, "
            "not wrap them in a Message first"
        )
