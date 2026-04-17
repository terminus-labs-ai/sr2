"""Integration tests for compaction pipeline — no mocks (except LLM callable).

These tests verify that compaction actually runs during real pipeline execution,
that the right strategies are applied, and that compaction interacts correctly
with summarization.
"""

import json

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import (
    CompactionConfig,
    CompactionRuleConfig,
    CostGateConfig,
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
            cost_gate=CostGateConfig(enabled=False),
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

        # Turns 0-14 are compactable (20 - RAW_WINDOW=5 = 15 turns outside raw window).
        # Of those, tool_output turns are at i%3==1: i=1,4,7,10,13 → 5 tool outputs.
        # User turns (i%3==0) are never compacted. Assistant turns have no content_type.
        assert result.turns_compacted == 5, (
            f"Expected 5 tool_output turns compacted, got {result.turns_compacted}"
        )

        # Verify exact original_tokens: engine sums len(content)//4 for all turns
        # using their pre-compaction content. For the 5 compacted tool turns, the
        # original content was _long_tool_output() (identical for each).
        tool_content = _long_tool_output()
        tool_tokens = len(tool_content) // 4
        non_compacted = [t for t in result.turns if not t.compacted]
        expected_original = (
            sum(len(t.content) // 4 for t in non_compacted) + 5 * tool_tokens
        )
        assert result.original_tokens == expected_original, (
            f"Expected original_tokens={expected_original}, got {result.original_tokens}"
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
        """Compacted tool outputs must include recovery hints with tool name.

        Engine appends: '\\n  Recovery: {hint}'
        SchemaAndSampleRule with recovery_hint=True produces: 'Re-fetch with {tool_name}'
        So final format is: '\\n  Recovery: Re-fetch with {tool_name}'
        """
        mgr = self._build_pipeline()

        for i in range(10):
            if i % 2 == 0:
                mgr.add_turn(_make_tool_turn(i, tool_name=f"fetch_data_{i}"))
            else:
                mgr.add_turn(_make_assistant_turn(i))

        mgr.run_compaction()

        zones = mgr.zones()
        compacted_tools = [t for t in zones.compacted if t.compacted and t.content_type == "tool_output"]
        # 10 turns, RAW_WINDOW=5: compactable = turns 0-4.
        # Tool turns at i=0,2,4 (even indices) → 3 tool outputs in compactable zone.
        assert len(compacted_tools) == 3, (
            f"Expected 3 compacted tool outputs, got {len(compacted_tools)}"
        )

        for turn in compacted_tools:
            tool_name = turn.metadata.get("tool_name", "the tool")
            expected_hint = f"\n  Recovery: Re-fetch with {tool_name}"
            assert expected_hint in turn.content, (
                f"Turn {turn.turn_number} missing expected recovery hint "
                f"'{expected_hint}'. Content: {turn.content}"
            )

    def test_file_content_recovery_hint_format(self):
        """File content compacted with ReferenceRule must include read_file recovery hint.

        ReferenceRule produces recovery_hint='read_file("{path}")'.
        Engine appends: '\\n  Recovery: read_file("{path}")'.
        """
        config = CompactionConfig(
            enabled=True,
            raw_window=2,
            min_content_size=10,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(
                    type="file_content",
                    strategy="reference",
                    recovery_hint=True,
                ),
            ],
        )
        engine = CompactionEngine(config)
        mgr = ConversationManager(
            compaction_engine=engine,
            summarization_engine=None,
            raw_window=2,
            compacted_max_tokens=10000,
        )

        file_path = "/src/app/main.py"
        file_content = "def main():\n" + "\n".join(f"    x_{i} = {i}" for i in range(30))
        file_turn = ConversationTurn(
            turn_number=0,
            role="tool_result",
            content=file_content,
            content_type="file_content",
            metadata={"file_path": file_path, "line_count": 31, "language": "python"},
        )
        mgr.add_turn(file_turn)
        mgr.add_turn(_make_assistant_turn(1, "Here's the file."))
        mgr.add_turn(_make_user_turn(2, "Thanks"))
        mgr.add_turn(_make_assistant_turn(3, "You're welcome"))

        result = mgr.run_compaction()
        assert result is not None
        assert result.turns_compacted == 1

        compacted_turn = [t for t in mgr.zones().compacted if t.compacted][0]
        # ReferenceRule replaces content with "→ Saved to {path} ({metadata})"
        assert f"Saved to {file_path}" in compacted_turn.content
        # Recovery hint format: read_file("{path}")
        expected_recovery = f'\n  Recovery: read_file("{file_path}")'
        assert expected_recovery in compacted_turn.content, (
            f"Expected recovery hint '{expected_recovery}' in content: {compacted_turn.content}"
        )

    def test_zone_transitions_tracked(self):
        """Zone transition metrics should be updated after compaction."""
        mgr = self._build_pipeline()

        for i in range(10):
            mgr.add_turn(_make_tool_turn(i))

        mgr.run_compaction()

        transitions = mgr.get_zone_transitions()
        assert "raw_to_compacted" in transitions, "Should track raw->compacted transitions"
        # 10 turns started in raw, 5 remain (RAW_WINDOW=5) → 5 transitioned
        assert transitions["raw_to_compacted"] == 5, (
            f"Expected 5 raw->compacted transitions, got {transitions['raw_to_compacted']}"
        )

    def test_compaction_idempotent_across_runs(self):
        """Running compaction multiple times should be idempotent."""
        mgr = self._build_pipeline()

        for i in range(10):
            mgr.add_turn(_make_tool_turn(i))

        mgr.run_compaction()
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
        # 8 turns - RAW_WINDOW(5) = 3 compactable, all tool_output
        assert result1.turns_compacted == 3, (
            f"Phase 1: expected 3 turns compacted, got {result1.turns_compacted}"
        )

        # Phase 2: Add 8 more turns, compact again
        for i in range(8, 16):
            mgr.add_turn(_make_tool_turn(i))
        result2 = mgr.run_compaction()
        assert result2 is not None
        # 16 total turns - RAW_WINDOW(5) = 11 compactable.
        # Turns 0-2 already compacted (skipped), turns 3-10 are new → 8 compacted.
        assert result2.turns_compacted == 8, (
            f"Phase 2: expected 8 turns compacted, got {result2.turns_compacted}"
        )

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
            cost_gate=CostGateConfig(enabled=False),
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
        # 35 turns - RAW_WINDOW(3) = 32 compactable. Tool turns (i%3==1):
        # i=1,4,7,10,13,16,19,22,25,28,31 → 11 tool outputs in compactable zone.
        assert compaction_result.turns_compacted == 11, (
            f"Expected 11 tool outputs compacted, got {compaction_result.turns_compacted}"
        )

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
            # Preserved turns must be the MOST RECENT from the compacted zone,
            # not arbitrary ones. Verify identity of both preserved turns.
            expected_preserved = compacted_before[-2:]
            for i, (actual, expected) in enumerate(zip(zones.compacted, expected_preserved)):
                assert actual.turn_number == expected.turn_number, (
                    f"Preserved turn {i}: expected turn_number={expected.turn_number}, "
                    f"got {actual.turn_number}. Preserved turns must be the most recent."
                )
                assert actual.content == expected.content, (
                    f"Preserved turn {i} (turn_number={actual.turn_number}): "
                    f"content changed after summarization"
                )
            # Also verify ordering: preserved turns should be in ascending turn order
            assert zones.compacted[0].turn_number < zones.compacted[1].turn_number, (
                f"Preserved turns should be in ascending order: "
                f"{zones.compacted[0].turn_number} >= {zones.compacted[1].turn_number}"
            )

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
        # 20 turns started in raw, 3 remain (RAW_WINDOW=3) → 17 transitioned
        assert transitions["raw_to_compacted"] == 17, (
            f"Expected 17 raw->compacted transitions, got {transitions['raw_to_compacted']}"
        )

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
            cost_gate=CostGateConfig(enabled=False),
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
            cost_gate=CostGateConfig(enabled=False),
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
            cost_gate=CostGateConfig(enabled=False),
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


class TestCompactionDisabled:
    """When enabled=False, compaction must never run regardless of content."""

    def test_compaction_disabled_no_turns_compacted(self):
        """With enabled=False, turns matching rules are never compacted."""
        config = CompactionConfig(
            enabled=False,
            raw_window=2,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
                CompactionRuleConfig(type="file_content", strategy="reference"),
                CompactionRuleConfig(type="code_execution", strategy="result_summary"),
            ],
        )
        engine = CompactionEngine(config)
        mgr = ConversationManager(
            compaction_engine=engine,
            summarization_engine=None,
            raw_window=2,
            compacted_max_tokens=10000,
        )

        # Add turns that WOULD match every configured rule if enabled
        mgr.add_turn(ConversationTurn(
            turn_number=0, role="tool_result",
            content=_long_tool_output(20),
            content_type="tool_output",
            metadata={"tool_name": "search"},
        ))
        mgr.add_turn(ConversationTurn(
            turn_number=1, role="tool_result",
            content="def main():\n" + "\n".join(f"    line_{i} = {i}" for i in range(30)),
            content_type="file_content",
            metadata={"file_path": "/src/main.py", "line_count": 30},
        ))
        mgr.add_turn(ConversationTurn(
            turn_number=2, role="tool_result",
            content="Running tests...\n" + "\n".join(f"test_{i} PASSED" for i in range(20)),
            content_type="code_execution",
            metadata={"exit_code": 0},
        ))
        # Padding turns to push others outside raw window
        mgr.add_turn(_make_assistant_turn(3, "ok"))
        mgr.add_turn(_make_assistant_turn(4, "done"))

        result = mgr.run_compaction()

        # run_compaction returns None when disabled
        assert result is None, (
            "run_compaction should return None when compaction is disabled"
        )

        # Verify all turns are untouched — none have compacted=True
        zones = mgr.zones()
        all_turns = zones.compacted + zones.raw
        for turn in all_turns:
            assert turn.compacted is False, (
                f"Turn {turn.turn_number} (content_type={turn.content_type}) "
                f"was compacted despite enabled=False"
            )


class TestRuleMatchingBehavior:
    """Tests for rule matching mechanics in CompactionEngine._build_rule_map.

    The docs say 'if a turn matches multiple types, the first matching rule wins.'
    However, the implementation uses a dict keyed by content_type. Each turn has
    exactly one content_type, so there's no ambiguity at match time. But if two
    rules are configured with the SAME type, the dict overwrites — meaning the
    LAST rule definition wins, not the first.

    This test documents that behavior.
    """

    def test_duplicate_type_last_rule_wins(self):
        """When two rules share the same type, the last one in the list is used.

        This is a dict-overwrite behavior, not 'first match wins'. The docs are
        misleading — there's no list-scan; it's a dict lookup by content_type.
        """
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                # First rule for tool_output: schema_and_sample
                CompactionRuleConfig(
                    type="tool_output",
                    strategy="schema_and_sample",
                    max_compacted_tokens=80,
                ),
                # Second rule for tool_output: collapse (overwrites the first)
                CompactionRuleConfig(
                    type="tool_output",
                    strategy="collapse",
                ),
            ],
        )
        engine = CompactionEngine(config)

        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content=_long_tool_output(20),
                content_type="tool_output",
                metadata={"tool_name": "search_files", "args_summary": "*.py"},
            ),
            _make_assistant_turn(1),
        ]
        result = engine.compact(turns)

        assert result.turns_compacted == 1
        compacted = result.turns[0]
        # CollapseRule produces "→ ✓ {tool_name}({args_summary})"
        assert "✓" in compacted.content and "search_files" in compacted.content, (
            f"Expected collapse strategy output (last rule wins), got: {compacted.content}"
        )
        # Verify it did NOT use schema_and_sample (which produces "X lines. Sample:")
        assert "lines" not in compacted.content, (
            f"First rule (schema_and_sample) was used instead of last rule (collapse): {compacted.content}"
        )

    def test_each_content_type_maps_to_one_rule(self):
        """A turn's content_type is looked up in a dict — exactly one rule applies.

        This confirms there's no ambiguity: each content_type has at most one
        rule, and matching is O(1) dict lookup, not a list scan.
        """
        config = CompactionConfig(
            enabled=True,
            raw_window=1,
            min_content_size=5,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
                CompactionRuleConfig(type="file_content", strategy="reference"),
            ],
        )
        engine = CompactionEngine(config)

        # Verify the internal rule map has exactly one entry per type
        assert len(engine._rule_map) == 2
        assert "tool_output" in engine._rule_map
        assert "file_content" in engine._rule_map

        # A turn with content_type="tool_output" gets schema_and_sample, not reference
        turns = [
            ConversationTurn(
                turn_number=0, role="tool_result",
                content=_long_tool_output(20),
                content_type="tool_output",
                metadata={"tool_name": "search"},
            ),
            _make_assistant_turn(1),
        ]
        result = engine.compact(turns)
        assert result.turns_compacted == 1
        assert "lines" in result.turns[0].content, (
            "tool_output should use schema_and_sample strategy"
        )
