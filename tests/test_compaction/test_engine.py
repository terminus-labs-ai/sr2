"""Tests for compaction engine."""

import pytest

from sr2.compaction.engine import CompactionEngine, CompactionResult, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig


def _make_config(raw_window: int = 3, min_content_size: int = 10) -> CompactionConfig:
    """Create a CompactionConfig with schema_and_sample rule."""
    return CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=min_content_size,
        rules=[
            CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
        ],
    )


def _long_content(lines: int = 20) -> str:
    """Generate content long enough to trigger compaction."""
    return "\n".join(f"data line {i}: some verbose output" for i in range(lines))


class TestCompactionEngine:
    """Tests for CompactionEngine."""

    def test_within_raw_window_none_compacted(self):
        """Turns within raw window are not compacted."""
        config = _make_config(raw_window=5)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(turn_number=i, role="assistant", content="short")
            for i in range(3)
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 0

    def test_user_messages_never_compacted(self):
        """User messages are never compacted regardless of position."""
        config = _make_config(raw_window=2)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="user", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert turns[0].compacted is False

    def test_already_compacted_skipped(self):
        """Already-compacted turns are skipped."""
        config = _make_config(raw_window=2)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(),
                content_type="tool_output", compacted=True,
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 0

    def test_below_min_content_size_skipped(self):
        """Content below min_content_size is skipped."""
        config = _make_config(raw_window=2, min_content_size=1000)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content="short",
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 0

    def test_tool_output_with_matching_rule_compacted(self):
        """Tool output turn with matching rule is compacted."""
        config = _make_config(raw_window=2)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        # Original turn is not mutated; compacted turn is in result.turns
        assert turns[0].compacted is False
        assert result.turns[0].compacted is True
        assert "lines" in result.turns[0].content

    def test_unknown_content_type_not_compacted(self):
        """Turn with unknown content_type is not compacted."""
        config = _make_config(raw_window=2)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(),
                content_type="unknown_type",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
            ConversationTurn(turn_number=2, role="assistant", content="done"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 0

    def test_correct_token_counts(self):
        """CompactionResult reports correct token counts."""
        config = _make_config(raw_window=1)
        engine = CompactionEngine(config)
        long = _long_content()
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=long,
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)

        assert result.original_tokens > 0
        assert result.compacted_tokens < result.original_tokens

    def test_turns_compacted_count(self):
        """turns_compacted count is accurate."""
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

    def test_idempotent(self):
        """Compaction is idempotent — running twice produces same result."""
        config = _make_config(raw_window=1)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=_long_content(),
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="done"),
        ]

        result1 = engine.compact(turns)
        result2 = engine.compact(result1.turns)

        assert result2.turns_compacted == 0  # Already compacted
        assert result1.compacted_tokens == result2.compacted_tokens
