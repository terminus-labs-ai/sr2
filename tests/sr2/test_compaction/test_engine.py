"""Tests for compaction engine."""

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, CostGateConfig


def _make_config(raw_window: int = 3, min_content_size: int = 10) -> CompactionConfig:
    """Create a CompactionConfig with schema_and_sample rule."""
    return CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=min_content_size,
        cost_gate=CostGateConfig(enabled=False),
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
        engine.compact(turns)

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
        """CompactionResult reports exact token counts based on len(content) // 4."""
        config = _make_config(raw_window=1)
        engine = CompactionEngine(config)
        long = _long_content()  # 20 lines of "data line N: some verbose output"
        protected_content = "ok"
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=long,
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content=protected_content),
        ]
        result = engine.compact(turns)

        # Token estimation is len(content) // 4 per the engine source
        expected_original_tokens = len(long) // 4 + len(protected_content) // 4
        assert result.original_tokens == expected_original_tokens, (
            f"Expected original_tokens={expected_original_tokens}, got {result.original_tokens}"
        )

        # Turn 0 was compacted; its compacted content is in result.turns[0]
        assert result.turns_compacted == 1
        compacted_turn_tokens = len(result.turns[0].content) // 4
        protected_tokens = len(protected_content) // 4
        expected_compacted_tokens = compacted_turn_tokens + protected_tokens
        assert result.compacted_tokens == expected_compacted_tokens, (
            f"Expected compacted_tokens={expected_compacted_tokens}, got {result.compacted_tokens}"
        )
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


class TestRawWindowBoundary:
    """Tests that the raw_window boundary is respected.

    Docs: "The last `raw_window` turns are never touched by compaction."
    """

    @pytest.mark.parametrize("raw_window", [1, 3, 5, 10])
    def test_oldest_turn_compacted_raw_window_untouched(self, raw_window: int):
        """With raw_window+1 turns, the oldest is compacted, the rest are untouched."""
        config = _make_config(raw_window=raw_window)
        engine = CompactionEngine(config)

        total = raw_window + 1
        turns = [
            ConversationTurn(
                turn_number=i,
                role="assistant",
                content=_long_content(),
                content_type="tool_output",
            )
            for i in range(total)
        ]

        result = engine.compact(turns)

        # The first turn (outside raw window) should be compacted
        assert result.turns[0].compacted is True
        # All turns in the raw window should be untouched
        for turn in result.turns[-raw_window:]:
            assert turn.compacted is False, (
                f"Turn {turn.turn_number} is in raw window but was compacted"
            )

    def test_raw_window_turns_untouched_even_when_matching_rule(self):
        """Raw window turns are never compacted, even if they match a compaction rule."""
        raw_window = 3
        config = _make_config(raw_window=raw_window)
        engine = CompactionEngine(config)

        # ALL turns are tool_output — every one matches the rule
        total = raw_window + 2  # 2 outside, 3 inside
        turns = [
            ConversationTurn(
                turn_number=i,
                role="assistant",
                content=_long_content(),
                content_type="tool_output",
            )
            for i in range(total)
        ]

        result = engine.compact(turns)

        # Turns outside raw window should be compacted
        assert result.turns_compacted == 2

        # Raw window turns must remain untouched despite matching the rule
        for turn in result.turns[-raw_window:]:
            assert turn.compacted is False, (
                f"Turn {turn.turn_number} is in raw window but was compacted"
            )
            # Content should be unchanged
            assert turn.content == _long_content(), (
                f"Turn {turn.turn_number} content was modified despite being in raw window"
            )

    def test_raw_window_untouched_after_repeated_compaction(self):
        """Running compaction twice: raw window turns stay untouched both times."""
        raw_window = 3
        config = _make_config(raw_window=raw_window)
        engine = CompactionEngine(config)

        total = raw_window + 3
        turns = [
            ConversationTurn(
                turn_number=i,
                role="assistant",
                content=_long_content(),
                content_type="tool_output",
            )
            for i in range(total)
        ]

        original_raw_contents = [t.content for t in turns[-raw_window:]]

        result1 = engine.compact(turns)
        result2 = engine.compact(result1.turns)

        # Raw window turns unchanged after both runs
        for i, turn in enumerate(result2.turns[-raw_window:]):
            assert turn.compacted is False, (
                f"Turn {turn.turn_number} compacted after second pass"
            )
            assert turn.content == original_raw_contents[i], (
                f"Turn {turn.turn_number} content changed after second pass"
            )


class TestMinContentSizeBoundary:
    """Boundary tests for min_content_size.

    Engine uses strict less-than: `if est_tokens < min_size` → skip.
    Token estimation: len(content) // 4.
    """

    def test_at_min_content_size_is_compacted(self):
        """Content with exactly min_content_size tokens IS compacted (not skipped)."""
        min_size = 50
        # Create content whose len // 4 == exactly min_size
        content = "x" * (min_size * 4)  # 200 chars → 200 // 4 = 50 tokens
        assert len(content) // 4 == min_size  # Sanity check

        config = _make_config(raw_window=1, min_content_size=min_size)
        # Need a rule that matches; content needs >3 lines for schema_and_sample
        # Use multiline content that still has the right total length
        lines = ["line"] * 10
        content = "\n".join(lines)
        # Pad to exactly min_size * 4 chars
        content = content.ljust(min_size * 4, "z")
        assert len(content) // 4 == min_size

        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=content,
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 1, (
            f"Content at exactly min_content_size ({min_size} tokens) should be compacted"
        )

    def test_below_min_content_size_is_skipped(self):
        """Content with min_content_size - 1 tokens is NOT compacted."""
        min_size = 50
        # Create content whose len // 4 == min_size - 1
        content_len = (min_size - 1) * 4  # 196 chars → 196 // 4 = 49 tokens
        lines = ["line"] * 10
        content = "\n".join(lines)
        content = content.ljust(content_len, "z")
        assert len(content) // 4 == min_size - 1  # Sanity check

        config = _make_config(raw_window=1, min_content_size=min_size)
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="assistant", content=content,
                content_type="tool_output",
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 0, (
            f"Content below min_content_size ({min_size - 1} < {min_size} tokens) should NOT be compacted"
        )


class TestRecoveryHintInCompactedContent:
    """Engine appends '\\n  Recovery: {hint}' to compacted content.

    Each rule produces a different recovery_hint format:
    - SchemaAndSampleRule (tool_output): 'Re-fetch with {tool_name}'
    - ReferenceRule (file_content): 'read_file("{path}")'
    - ResultSummaryRule (code_execution): metadata.result_path or None

    The engine is responsible for appending the hint to the content string.
    These tests verify the engine-level behavior, not the rule-level hint generation.
    """

    def test_tool_output_recovery_hint_format(self):
        """Tool output compaction appends 'Re-fetch with {tool_name}' recovery hint."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(
                    type="tool_output",
                    strategy="schema_and_sample",
                    recovery_hint=True,
                ),
            ],
        )
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content=_long_content(20),
                content_type="tool_output",
                metadata={"tool_name": "list_directory"},
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        compacted = result.turns[0]
        assert compacted.content.endswith("\n  Recovery: Re-fetch with list_directory"), (
            f"Expected tool output recovery hint format, got: {compacted.content!r}"
        )

    def test_file_content_recovery_hint_format(self):
        """File content compaction appends 'read_file(\"{path}\")' recovery hint."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(
                    type="file_content",
                    strategy="reference",
                ),
            ],
        )
        engine = CompactionEngine(config)
        file_path = "/src/models/user.py"
        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content="class User:\n" + "\n".join(f"    field_{i} = {i}" for i in range(20)),
                content_type="file_content",
                metadata={"file_path": file_path, "line_count": 21, "language": "python"},
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        compacted = result.turns[0]
        expected_suffix = f'\n  Recovery: read_file("{file_path}")'
        assert compacted.content.endswith(expected_suffix), (
            f"Expected file content recovery hint format ending with {expected_suffix!r}, "
            f"got: {compacted.content!r}"
        )

    def test_code_execution_recovery_hint_with_result_path(self):
        """Code execution with result_path appends it as recovery hint."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(
                    type="code_execution",
                    strategy="result_summary",
                ),
            ],
        )
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content="Running tests...\n" + "\n".join(f"test_{i} passed" for i in range(20)),
                content_type="code_execution",
                metadata={"exit_code": 0, "result_path": "/tmp/exec_42.log"},
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        compacted = result.turns[0]
        assert "\n  Recovery: /tmp/exec_42.log" in compacted.content, (
            f"Expected code execution recovery hint with result_path, got: {compacted.content!r}"
        )

    def test_code_execution_no_recovery_hint_without_result_path(self):
        """Code execution without result_path has no recovery hint appended."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(
                    type="code_execution",
                    strategy="result_summary",
                ),
            ],
        )
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content="Running tests...\n" + "\n".join(f"test_{i} passed" for i in range(20)),
                content_type="code_execution",
                metadata={"exit_code": 0},
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        compacted = result.turns[0]
        assert "Recovery:" not in compacted.content, (
            f"No recovery hint expected when result_path is absent, got: {compacted.content!r}"
        )

    def test_tool_output_no_recovery_hint_when_disabled(self):
        """Tool output with recovery_hint=False has no hint appended."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(
                    type="tool_output",
                    strategy="schema_and_sample",
                    recovery_hint=False,
                ),
            ],
        )
        engine = CompactionEngine(config)
        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content=_long_content(20),
                content_type="tool_output",
                metadata={"tool_name": "search_files"},
            ),
            ConversationTurn(turn_number=1, role="assistant", content="ok"),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        compacted = result.turns[0]
        assert "Recovery:" not in compacted.content, (
            f"No recovery hint expected when recovery_hint=False, got: {compacted.content!r}"
        )
