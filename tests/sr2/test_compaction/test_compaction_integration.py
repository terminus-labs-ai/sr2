"""Integration tests for compaction pipeline — no mocks (except LLM callable).

These tests verify that compaction actually runs during real pipeline execution,
that the right strategies are applied, and that compaction interacts correctly
with summarization.
"""

import json
import logging

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import (
    CompactionConfig,
    CompactionRuleConfig,
    SummarizationConfig,
)
from sr2.pipeline.conversation import ConversationManager
from sr2.summarization.engine import SummarizationEngine


def _long_tool_output(lines: int = 20, prefix: str = "result") -> str:
    """Generate realistic tool output that should trigger compaction."""
    return "\n".join(f"{prefix}_line_{i}: value_{i} status=ok data={{x: {i}}}" for i in range(lines))


def _make_tool_turn(
    turn_num: int,
    content: str | None = None,
    tool_name: str = "search_files",
) -> ConversationTurn:
    """Create a tool result turn with content_type set."""
    return ConversationTurn(
        turn_number=turn_num,
        role="tool_result",
        content=content or _long_tool_output(),
        content_type="tool_output",
        metadata={"tool_name": tool_name},
    )


def _make_user_turn(turn_num: int, content: str = "Do something") -> ConversationTurn:
    return ConversationTurn(turn_number=turn_num, role="user", content=content)


def _make_assistant_turn(turn_num: int, content: str = "Done.") -> ConversationTurn:
    return ConversationTurn(turn_number=turn_num, role="assistant", content=content)


class TestCompactionOnlyIntegration:
    """Integration test: compaction enabled, summarization disabled.

    Verifies:
    - Compaction actually runs on tool outputs
    - Tool outputs older than raw_window are compacted
    - Compacted content includes recovery hints
    - Metrics/state reflect compaction activity
    """

    RAW_WINDOW = 5

    def _build_pipeline(self) -> ConversationManager:
        config = CompactionConfig(
            enabled=True,
            raw_window=self.RAW_WINDOW,
            min_content_size=10,
            rules=[
                CompactionRuleConfig(
                    type="tool_output",
                    strategy="schema_and_sample",
                    max_compacted_tokens=80,
                    recovery_hint=True,
                ),
                CompactionRuleConfig(
                    type="code_execution",
                    strategy="result_summary",
                    max_output_lines=3,
                ),
            ],
        )
        engine = CompactionEngine(config)
        # Summarization disabled
        return ConversationManager(
            compaction_engine=engine,
            summarization_engine=None,
            raw_window=self.RAW_WINDOW,
            compacted_max_tokens=10000,
        )

    def test_20_turns_compaction_runs(self, caplog):
        """20 turns with tool outputs → compaction must actually run."""
        mgr = self._build_pipeline()

        # Simulate 20 turns: user -> tool_result -> assistant (repeated)
        turn_num = 0
        for i in range(20):
            if i % 3 == 0:
                mgr.add_turn(_make_user_turn(turn_num, f"Search for item {i}"))
            elif i % 3 == 1:
                mgr.add_turn(_make_tool_turn(turn_num, tool_name=f"search_{i}"))
            else:
                mgr.add_turn(_make_assistant_turn(turn_num, f"Found results for item {i}"))
            turn_num += 1

        # Run compaction
        result = mgr.run_compaction()

        # Compaction must have run (not None)
        assert result is not None, "Compaction should have triggered with 20 turns"

        # At least some tool outputs should be compacted
        assert result.turns_compacted > 0, (
            f"Expected tool outputs to be compacted, but turns_compacted={result.turns_compacted}"
        )

        # Token count should decrease
        assert result.compacted_tokens < result.original_tokens, (
            f"Compaction should reduce tokens: {result.original_tokens} -> {result.compacted_tokens}"
        )

        # Raw zone should have exactly raw_window turns
        zones = mgr.zones()
        assert len(zones.raw) == self.RAW_WINDOW

    def test_tool_outputs_outside_raw_window_compacted(self):
        """Tool outputs older than raw_window turns must be compacted."""
        mgr = self._build_pipeline()

        # Add enough turns to push tool outputs outside raw window
        turns = []
        for i in range(12):
            if i % 2 == 0:
                turn = _make_tool_turn(i, tool_name=f"tool_{i}")
            else:
                turn = _make_assistant_turn(i)
            mgr.add_turn(turn)
            turns.append(turn)

        mgr.run_compaction()

        # Check compacted zone: tool outputs should be compacted
        zones = mgr.zones()
        for turn in zones.compacted:
            if turn.content_type == "tool_output":
                assert turn.compacted, (
                    f"Turn {turn.turn_number} is in compacted zone with content_type=tool_output "
                    f"but compacted=False"
                )
                # Content should be the compacted form, not the original
                assert "lines" in turn.content or "→" in turn.content, (
                    f"Turn {turn.turn_number} content doesn't look compacted: {turn.content[:100]}"
                )

    def test_compacted_content_includes_recovery_hints(self):
        """Compacted tool outputs must include recovery hints."""
        mgr = self._build_pipeline()

        for i in range(10):
            if i % 2 == 0:
                mgr.add_turn(_make_tool_turn(i, tool_name=f"fetch_data_{i}"))
            else:
                mgr.add_turn(_make_assistant_turn(i))

        mgr.run_compaction()

        zones = mgr.zones()
        compacted_tools = [t for t in zones.compacted if t.compacted and t.content_type == "tool_output"]
        assert len(compacted_tools) > 0, "Should have at least one compacted tool output"

        for turn in compacted_tools:
            assert "Recovery:" in turn.content, (
                f"Turn {turn.turn_number} missing recovery hint. Content: {turn.content}"
            )
            assert "Re-fetch" in turn.content, (
                f"Turn {turn.turn_number} recovery hint doesn't contain re-fetch instruction"
            )

    def test_zone_transitions_tracked(self):
        """Zone transition metrics should be updated after compaction."""
        mgr = self._build_pipeline()

        for i in range(10):
            mgr.add_turn(_make_tool_turn(i))

        mgr.run_compaction()

        transitions = mgr.get_zone_transitions()
        assert "raw_to_compacted" in transitions, "Should track raw->compacted transitions"
        assert transitions["raw_to_compacted"] > 0

    def test_compaction_idempotent_across_runs(self):
        """Running compaction multiple times should be idempotent."""
        mgr = self._build_pipeline()

        for i in range(10):
            mgr.add_turn(_make_tool_turn(i))

        result1 = mgr.run_compaction()
        result2 = mgr.run_compaction()

        # Second run should compact nothing (all eligible turns already compacted)
        if result2 is not None:
            assert result2.turns_compacted == 0, (
                f"Second compaction should be a no-op, but compacted {result2.turns_compacted} turns"
            )

    def test_incremental_compaction(self):
        """Adding turns incrementally and compacting should work correctly."""
        mgr = self._build_pipeline()

        # Phase 1: Add 8 turns, compact
        for i in range(8):
            mgr.add_turn(_make_tool_turn(i))
        result1 = mgr.run_compaction()
        assert result1 is not None
        first_compacted = result1.turns_compacted

        # Phase 2: Add 8 more turns, compact again
        for i in range(8, 16):
            mgr.add_turn(_make_tool_turn(i))
        result2 = mgr.run_compaction()
        assert result2 is not None
        # New turns should get compacted, old ones already done
        assert result2.turns_compacted > 0

        # Raw window still respected
        assert len(mgr.zones().raw) == self.RAW_WINDOW

    def test_no_summarization_without_engine(self):
        """With summarization engine=None, summarization never triggers."""
        mgr = self._build_pipeline()

        # Fill up well beyond any threshold
        for i in range(30):
            mgr.add_turn(_make_tool_turn(i, content=_long_tool_output(50)))

        mgr.run_compaction()

        zones = mgr.zones()
        # Summarized zone must be empty
        assert len(zones.summarized) == 0, "Summarization should not trigger when engine is None"


class TestCompactionPlusSummarizationIntegration:
    """Integration test: both compaction and summarization enabled.

    Verifies:
    - Compaction runs first on tool outputs
    - Summarization triggers when compacted zone exceeds threshold
    - Proper interaction between the two systems
    """

    RAW_WINDOW = 3

    def _build_pipeline(self, compacted_max_tokens: int = 500) -> ConversationManager:
        compaction_config = CompactionConfig(
            enabled=True,
            raw_window=self.RAW_WINDOW,
            min_content_size=10,
            rules=[
                CompactionRuleConfig(
                    type="tool_output",
                    strategy="schema_and_sample",
                    max_compacted_tokens=80,
                    recovery_hint=True,
                ),
            ],
        )
        compaction_engine = CompactionEngine(compaction_config)

        summarization_config = SummarizationConfig(
            enabled=True,
            trigger="token_threshold",
            threshold=0.5,  # Low threshold to trigger sooner
            preserve_recent_turns=2,
            output_format="structured",
        )

        async def mock_llm(system: str, prompt: str) -> str:
            return json.dumps({
                "summary_of_turns": "0-20",
                "key_decisions": ["Used search tools to find items"],
                "unresolved": [],
                "facts": ["Multiple searches performed"],
                "user_preferences": [],
                "errors_encountered": [],
            })

        summarization_engine = SummarizationEngine(
            config=summarization_config,
            llm_callable=mock_llm,
        )

        return ConversationManager(
            compaction_engine=compaction_engine,
            summarization_engine=summarization_engine,
            raw_window=self.RAW_WINDOW,
            compacted_max_tokens=compacted_max_tokens,
        )

    @pytest.mark.asyncio
    async def test_30_turns_compaction_then_summarization(self):
        """30+ turns: compaction runs on tool outputs, summarization triggers on threshold."""
        mgr = self._build_pipeline(compacted_max_tokens=300)

        # Add 30+ turns with tool outputs
        turn_num = 0
        for i in range(35):
            if i % 3 == 0:
                mgr.add_turn(_make_user_turn(turn_num, f"Query {i}"))
            elif i % 3 == 1:
                mgr.add_turn(_make_tool_turn(turn_num, tool_name=f"search_{i}"))
            else:
                mgr.add_turn(_make_assistant_turn(turn_num, f"Result {i}"))
            turn_num += 1

        # Step 1: Run compaction
        compaction_result = mgr.run_compaction()
        assert compaction_result is not None
        assert compaction_result.turns_compacted > 0, "Compaction should have compacted tool outputs"

        # Step 2: Run summarization
        summ_result = await mgr.run_summarization()

        # With 35 turns and low threshold, summarization should trigger
        if summ_result is not None:
            # Verify summarization ran
            zones = mgr.zones()
            assert len(zones.summarized) > 0, "Summary should be stored"
            # Compacted zone should be reduced (some turns moved to summary)
            assert len(zones.compacted) < compaction_result.turns_compacted

    @pytest.mark.asyncio
    async def test_compaction_before_summarization_order(self):
        """Compaction must run before summarization — compacted tokens determine summary trigger."""
        mgr = self._build_pipeline(compacted_max_tokens=200)

        # Add many verbose tool outputs
        for i in range(20):
            mgr.add_turn(_make_tool_turn(i, content=_long_tool_output(25), tool_name=f"tool_{i}"))

        # Run compaction first
        compaction_result = mgr.run_compaction()
        assert compaction_result is not None, "Compaction should run"

        compacted_zone_tokens_before_summ = sum(
            len(t.content) // 4 for t in mgr.zones().compacted
        )

        # Now check if summarization triggers based on compacted tokens
        summ_result = await mgr.run_summarization()

        if compacted_zone_tokens_before_summ > 100:  # 0.5 * 200
            assert summ_result is not None, (
                f"Summarization should trigger: compacted_zone={compacted_zone_tokens_before_summ} > threshold=100"
            )

    @pytest.mark.asyncio
    async def test_preserved_turns_survive_summarization(self):
        """preserve_recent_turns should keep N recent turns in compacted zone after summarization."""
        mgr = self._build_pipeline(compacted_max_tokens=200)

        for i in range(20):
            mgr.add_turn(_make_tool_turn(i, content=_long_tool_output(25)))

        mgr.run_compaction()

        # Record the last turns in compacted zone before summarization
        compacted_before = list(mgr.zones().compacted)

        summ_result = await mgr.run_summarization()

        if summ_result is not None:
            zones = mgr.zones()
            # preserve_recent_turns=2, so 2 turns should remain in compacted zone
            assert len(zones.compacted) == 2, (
                f"Expected 2 preserved turns, got {len(zones.compacted)}"
            )
            # These should be the most recent turns from the compacted zone
            assert zones.compacted[-1].turn_number == compacted_before[-1].turn_number

    @pytest.mark.asyncio
    async def test_zone_transitions_tracked_through_both(self):
        """Zone transitions should track both raw→compacted and compacted→summarized."""
        mgr = self._build_pipeline(compacted_max_tokens=200)

        for i in range(20):
            mgr.add_turn(_make_tool_turn(i, content=_long_tool_output(25)))

        mgr.run_compaction()
        await mgr.run_summarization()

        transitions = mgr.get_zone_transitions()
        assert "raw_to_compacted" in transitions
        assert transitions["raw_to_compacted"] > 0

        zones = mgr.zones()
        if len(zones.summarized) > 0:
            assert "compacted_to_summarized" in transitions
            assert transitions["compacted_to_summarized"] > 0


class TestCompactionRuleMatching:
    """Integration test: verify YAML-style rule configs are loaded and matched."""

    def test_multiple_rules_match_correct_content_types(self):
        """Each content_type should match its configured strategy."""
        config = CompactionConfig(
            enabled=True,
            raw_window=2,
            min_content_size=5,
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample", recovery_hint=True),
                CompactionRuleConfig(type="file_content", strategy="reference"),
                CompactionRuleConfig(type="code_execution", strategy="result_summary"),
                CompactionRuleConfig(type="confirmation", strategy="collapse"),
            ],
        )
        engine = CompactionEngine(config)

        tool_turn = ConversationTurn(
            turn_number=0, role="tool_result",
            content=_long_tool_output(20),
            content_type="tool_output",
            metadata={"tool_name": "search"},
        )
        file_turn = ConversationTurn(
            turn_number=1, role="tool_result",
            content="def main():\n" + "\n".join(f"    line_{i} = {i}" for i in range(30)),
            content_type="file_content",
            metadata={"file_path": "/src/main.py", "line_count": 30, "language": "python"},
        )
        exec_turn = ConversationTurn(
            turn_number=2, role="tool_result",
            content="Running tests...\n" + "\n".join(f"test_{i} passed" for i in range(20)),
            content_type="code_execution",
            metadata={"exit_code": 0},
        )
        confirm_turn = ConversationTurn(
            turn_number=3, role="tool_result",
            content="File written successfully to /src/main.py with 30 lines of Python code.",
            content_type="confirmation",
            metadata={"tool_name": "write_file", "args_summary": "main.py"},
        )
        # Protected turns (in raw window)
        protected = [
            _make_assistant_turn(4),
            _make_assistant_turn(5),
        ]

        turns = [tool_turn, file_turn, exec_turn, confirm_turn] + protected
        result = engine.compact(turns)

        assert result.turns_compacted == 4, f"Expected all 4 content types compacted, got {result.turns_compacted}"

        # Check each was compacted with the right strategy
        compacted_turns = [t for t in result.turns if t.compacted]
        assert any("lines" in t.content for t in compacted_turns), "schema_and_sample should produce 'lines'"
        assert any("Saved to" in t.content for t in compacted_turns), "reference should produce 'Saved to'"
        assert any("Exit" in t.content for t in compacted_turns), "result_summary should produce 'Exit'"
        assert any("✓" in t.content and "write_file" in t.content for t in compacted_turns), "collapse should produce checkmark + tool name"

    def test_unmatched_content_type_passes_through(self):
        """Content types not in rules should pass through uncompacted."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
            ],
        )
        engine = CompactionEngine(config)

        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content=_long_tool_output(20),
                content_type="custom_type_not_in_rules",
            ),
            _make_assistant_turn(1),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 0

    def test_none_content_type_passes_through(self):
        """Turns with content_type=None should never match any rule."""
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
            ],
        )
        engine = CompactionEngine(config)

        turns = [
            ConversationTurn(
                turn_number=0, role="assistant",
                content=_long_tool_output(20),
                content_type=None,  # Explicitly None — the old bug
            ),
            _make_assistant_turn(1),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 0
