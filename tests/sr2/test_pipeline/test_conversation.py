"""Tests for three-zone conversation manager."""

import json

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, CostGateConfig, SummarizationConfig
from sr2.pipeline.conversation import ConversationManager, ConversationZones
from sr2.summarization.engine import SummarizationEngine


def _make_compaction_engine(raw_window: int = 3) -> CompactionEngine:
    config = CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=10,
        cost_gate=CostGateConfig(enabled=False),
        rules=[
            CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
        ],
    )
    return CompactionEngine(config)


def _make_turn(num: int, role: str = "assistant", content: str = "turn content") -> ConversationTurn:
    return ConversationTurn(turn_number=num, role=role, content=content)


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_add_turn_to_raw_zone(self):
        """add_turn() adds to raw zone."""
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        turn = _make_turn(0)
        mgr.add_turn(turn)

        assert len(mgr.zones().raw) == 1
        assert mgr.zones().raw[0] is turn

    def test_compaction_disabled_returns_none(self):
        """run_compaction() returns None immediately when compaction is disabled."""
        config = CompactionConfig(
            enabled=False,
            raw_window=2,
            min_content_size=10,
            cost_gate=CostGateConfig(enabled=False),
            rules=[
                CompactionRuleConfig(type="tool_output", strategy="schema_and_sample"),
            ],
        )
        engine = CompactionEngine(config)
        mgr = ConversationManager(compaction_engine=engine, raw_window=2)
        # Add enough turns to normally trigger compaction
        for i in range(10):
            mgr.add_turn(_make_turn(i))

        result = mgr.run_compaction()

        assert result is None
        # Verify turns were NOT moved — all still in raw zone
        assert len(mgr.zones().raw) == 10
        assert len(mgr.zones().compacted) == 0

    def test_compaction_within_raw_window_returns_none(self):
        """run_compaction() with turns <= raw_window returns None."""
        engine = _make_compaction_engine(raw_window=5)
        mgr = ConversationManager(compaction_engine=engine, raw_window=5)
        for i in range(3):
            mgr.add_turn(_make_turn(i))

        result = mgr.run_compaction()
        assert result is None

    def test_compaction_moves_to_compacted_zone(self):
        """run_compaction() with excess turns moves to compacted zone."""
        engine = _make_compaction_engine(raw_window=2)
        mgr = ConversationManager(compaction_engine=engine, raw_window=2)
        for i in range(5):
            mgr.add_turn(_make_turn(i))

        result = mgr.run_compaction()

        assert result is not None
        assert len(mgr.zones().compacted) > 0

    def test_raw_zone_limited_after_compaction(self):
        """Raw zone always has <= raw_window turns after compaction."""
        engine = _make_compaction_engine(raw_window=2)
        mgr = ConversationManager(compaction_engine=engine, raw_window=2)
        for i in range(10):
            mgr.add_turn(_make_turn(i))

        mgr.run_compaction()

        assert len(mgr.zones().raw) <= 2

    @pytest.mark.asyncio
    async def test_summarization_below_threshold_returns_none(self):
        """run_summarization() with tokens below threshold returns None."""
        engine = _make_compaction_engine(raw_window=2)
        summ_config = SummarizationConfig(trigger="token_threshold", threshold=0.75)
        summ_engine = SummarizationEngine(config=summ_config)

        mgr = ConversationManager(
            compaction_engine=engine,
            summarization_engine=summ_engine,
            raw_window=2,
            compacted_max_tokens=10000,
        )
        # Add small content to compacted zone
        mgr.zones().compacted = [_make_turn(0, content="short")]

        result = await mgr.run_summarization()
        assert result is None

    @pytest.mark.asyncio
    async def test_summarization_above_threshold_summarizes(self):
        """run_summarization() with tokens above threshold summarizes and clears compacted zone."""
        async def mock_llm(system: str, prompt: str) -> str:
            return json.dumps({
                "summary_of_turns": "0-5",
                "key_decisions": ["Use Python"],
                "unresolved": [],
                "facts": [],
                "user_preferences": [],
                "errors_encountered": [],
            })

        engine = _make_compaction_engine(raw_window=2)
        summ_config = SummarizationConfig(enabled=True, trigger="token_threshold", threshold=0.5, preserve_recent_turns=0)
        summ_engine = SummarizationEngine(config=summ_config, llm_callable=mock_llm)

        mgr = ConversationManager(
            compaction_engine=engine,
            summarization_engine=summ_engine,
            raw_window=2,
            compacted_max_tokens=100,  # Small threshold
        )
        # Fill compacted zone with enough tokens to trigger
        long_content = "x" * 800  # ~200 tokens, above 0.5 * 100
        for i in range(5):
            mgr.zones().compacted.append(_make_turn(i, content=long_content))

        result = await mgr.run_summarization()

        assert result is not None
        assert len(mgr.zones().compacted) == 0  # Cleared after summarization

    @pytest.mark.asyncio
    async def test_summary_stored_in_summarized_zone(self):
        """Summary stored in summarized zone after summarization."""
        async def mock_llm(system: str, prompt: str) -> str:
            return json.dumps({
                "summary_of_turns": "0-2",
                "key_decisions": ["Deploy"],
                "unresolved": [],
                "facts": [],
                "user_preferences": [],
                "errors_encountered": [],
            })

        engine = _make_compaction_engine(raw_window=2)
        summ_config = SummarizationConfig(enabled=True, trigger="token_threshold", threshold=0.5, preserve_recent_turns=0)
        summ_engine = SummarizationEngine(config=summ_config, llm_callable=mock_llm)

        mgr = ConversationManager(
            compaction_engine=engine,
            summarization_engine=summ_engine,
            raw_window=2,
            compacted_max_tokens=50,
        )
        mgr.zones().compacted = [_make_turn(i, content="x" * 400) for i in range(3)]

        await mgr.run_summarization()

        assert len(mgr.zones().summarized) == 1
        assert "Deploy" in mgr.zones().summarized[0]

    def test_total_tokens(self):
        """total_tokens sums across all zones correctly."""
        zones = ConversationZones(
            summarized=["A" * 40],  # 10 tokens
            compacted=[_make_turn(0, content="B" * 80)],  # 20 tokens
            raw=[_make_turn(1, content="C" * 120)],  # 30 tokens
        )
        assert zones.total_tokens == 60

    def test_get_all_turns(self):
        """get_all_turns() returns compacted + raw (not summarized)."""
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.zones().summarized = ["old summary"]
        mgr.zones().compacted = [_make_turn(0)]
        mgr.zones().raw = [_make_turn(1), _make_turn(2)]

        all_turns = mgr.get_all_turns()

        assert len(all_turns) == 3
        assert all_turns[0].turn_number == 0
        assert all_turns[2].turn_number == 2

    @pytest.mark.asyncio
    async def test_three_zone_end_to_end(self):
        """All three zones populated: raw (recent), compacted (compressed), summarized (oldest).

        Workflow: add many turns -> compact -> add more -> compact -> summarize.
        Verifies that turns flow correctly through the zone pipeline:
        raw -> compacted -> summarized.
        """
        async def mock_llm(system: str, prompt: str) -> str:
            return json.dumps({
                "summary_of_turns": "0-4",
                "key_decisions": ["Adopted Python 3.12"],
                "unresolved": [],
                "facts": ["Server runs on port 8008"],
                "user_preferences": [],
                "errors_encountered": [],
            })

        compaction_engine = _make_compaction_engine(raw_window=3)
        summ_config = SummarizationConfig(
            enabled=True, trigger="token_threshold", threshold=0.3, preserve_recent_turns=1
        )
        summ_engine = SummarizationEngine(config=summ_config, llm_callable=mock_llm)

        mgr = ConversationManager(
            compaction_engine=compaction_engine,
            summarization_engine=summ_engine,
            raw_window=3,
            compacted_max_tokens=100,  # Low threshold so summarization triggers easily
        )

        # Phase 1: Add enough turns to force compaction
        for i in range(8):
            content = f"Turn {i} content with enough text to be meaningful: {'x' * 200}"
            mgr.add_turn(_make_turn(i, role="user" if i % 2 == 0 else "assistant", content=content))

        # All 8 turns in raw, nothing compacted yet
        assert len(mgr.zones().raw) == 8
        assert len(mgr.zones().compacted) == 0

        # Run compaction: overflow moves older turns to compacted zone
        result = mgr.run_compaction()
        assert result is not None
        assert len(mgr.zones().raw) <= 3  # raw_window=3
        assert len(mgr.zones().compacted) > 0  # Older turns moved here

        # Phase 2: Run summarization on compacted zone
        summ_result = await mgr.run_summarization()
        assert summ_result is not None

        # Verify all three zones are populated
        zones = mgr.zones()
        assert len(zones.summarized) >= 1, "Summarized zone should have at least one summary"
        # preserve_recent_turns=1 keeps 1 turn in compacted
        assert len(zones.compacted) <= 1, "Compacted zone should have at most 1 preserved turn"
        assert len(zones.raw) <= 3, "Raw zone should have at most raw_window turns"

        # Verify summary content is present
        assert "Python 3.12" in zones.summarized[0]

        # Verify zone transitions were tracked
        transitions = mgr.get_zone_transitions()
        assert "raw_to_compacted" in transitions
        assert transitions["raw_to_compacted"] > 0
        assert "compacted_to_summarized" in transitions
        assert transitions["compacted_to_summarized"] > 0

    @pytest.mark.asyncio
    async def test_three_zone_full_lifecycle(self):
        """Full lifecycle: turns flow raw -> compacted -> summarized across multiple cycles.

        Verifies:
        - Specific turn numbers land in the correct zones
        - Oldest turns get summarized, newest stay in raw
        - Multiple summarization cycles accumulate summaries
        - Zone ordering invariant: summarized < compacted < raw (by turn number)
        """
        call_count = 0

        async def mock_llm(system: str, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "summary_of_turns": f"batch-{call_count}",
                "key_decisions": [f"Decision from batch {call_count}"],
                "unresolved": [],
                "facts": [f"Fact from batch {call_count}"],
                "user_preferences": [],
                "errors_encountered": [],
            })

        raw_window = 3
        compaction_engine = _make_compaction_engine(raw_window=raw_window)
        summ_config = SummarizationConfig(
            enabled=True,
            trigger="token_threshold",
            threshold=0.3,
            preserve_recent_turns=1,
        )
        summ_engine = SummarizationEngine(config=summ_config, llm_callable=mock_llm)

        mgr = ConversationManager(
            compaction_engine=compaction_engine,
            summarization_engine=summ_engine,
            raw_window=raw_window,
            compacted_max_tokens=80,  # Low so summarization triggers easily
        )

        # --- Cycle 1: Add turns 0-7, compact, summarize ---
        for i in range(8):
            content = f"Turn {i}: {'x' * 200}"  # ~50+ tokens each
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_turn(_make_turn(i, role=role, content=content))

        assert len(mgr.zones().raw) == 8
        assert len(mgr.zones().compacted) == 0

        # Compact: older turns move to compacted, last 3 stay in raw
        mgr.run_compaction()
        zones = mgr.zones()
        assert len(zones.raw) == raw_window
        raw_turn_nums_1 = [t.turn_number for t in zones.raw]
        compacted_turn_nums_1 = [t.turn_number for t in zones.compacted]
        assert raw_turn_nums_1 == [5, 6, 7], f"Raw should have newest turns, got {raw_turn_nums_1}"
        assert compacted_turn_nums_1 == [0, 1, 2, 3, 4], (
            f"Compacted should have older turns, got {compacted_turn_nums_1}"
        )

        # Summarize: compacted turns (except preserve_recent=1) become summary
        summ_result = await mgr.run_summarization()
        assert summ_result is not None
        zones = mgr.zones()
        assert len(zones.summarized) == 1
        assert "batch 1" in zones.summarized[0].lower() or "batch-1" in zones.summarized[0].lower() or "Decision from batch 1" in zones.summarized[0]
        # preserve_recent_turns=1 keeps the last compacted turn
        assert len(zones.compacted) == 1
        assert zones.compacted[0].turn_number == 4, (
            "Preserved compacted turn should be the most recent one (turn 4)"
        )
        assert len(zones.raw) == raw_window

        # --- Cycle 2: Add turns 8-13, compact again, summarize again ---
        for i in range(8, 14):
            content = f"Turn {i}: {'y' * 200}"
            role = "user" if i % 2 == 0 else "assistant"
            mgr.add_turn(_make_turn(i, role=role, content=content))

        # raw now has 3 (from before) + 6 new = 9 turns
        assert len(mgr.zones().raw) == 9

        mgr.run_compaction()
        zones = mgr.zones()
        assert len(zones.raw) == raw_window
        raw_turn_nums_2 = [t.turn_number for t in zones.raw]
        assert raw_turn_nums_2 == [11, 12, 13], (
            f"Raw should have newest turns after 2nd compaction, got {raw_turn_nums_2}"
        )
        # Compacted should have: preserved turn 4 + turns 5-10
        compacted_turn_nums_2 = [t.turn_number for t in zones.compacted]
        assert 4 in compacted_turn_nums_2, "Previously preserved turn 4 should still be in compacted"

        # Second summarization
        summ_result_2 = await mgr.run_summarization()
        assert summ_result_2 is not None
        zones = mgr.zones()
        assert len(zones.summarized) == 2, "Should have 2 accumulated summaries"
        assert len(zones.compacted) == 1, "preserve_recent_turns=1 keeps one compacted turn"

        # --- Verify zone ordering invariant ---
        # Compacted turns should have higher turn numbers than summarized content
        # Raw turns should have higher turn numbers than compacted turns
        if zones.compacted:
            max_compacted = max(t.turn_number for t in zones.compacted)
            min_raw = min(t.turn_number for t in zones.raw)
            assert max_compacted < min_raw, (
                f"Compacted turns ({max_compacted}) must precede raw turns ({min_raw})"
            )

        # Verify raw zone always has the absolute newest turns
        assert zones.raw[-1].turn_number == 13
        assert zones.raw[0].turn_number == 11

        # --- Verify cumulative zone transitions ---
        transitions = mgr.get_zone_transitions()
        assert transitions["raw_to_compacted"] > 5, (
            f"Expected many raw->compacted transitions, got {transitions['raw_to_compacted']}"
        )
        assert transitions["compacted_to_summarized"] > 3, (
            f"Expected multiple compacted->summarized transitions, got {transitions['compacted_to_summarized']}"
        )

    def test_run_compaction_passes_prefix_budget(self):
        """prefix_budget flows through to CompactionEngine.compact()."""
        from unittest.mock import patch

        engine = _make_compaction_engine(raw_window=2)
        mgr = ConversationManager(compaction_engine=engine, raw_window=2)
        for i in range(5):
            mgr.add_turn(_make_turn(i))

        with patch.object(engine, "compact", wraps=engine.compact) as mock_compact:
            mgr.run_compaction(prefix_budget=42)
            mock_compact.assert_called_once()
            _, kwargs = mock_compact.call_args
            assert kwargs.get("prefix_budget") == 42


class TestSeedFromHistory:
    """Tests for ConversationManager.seed_from_history()."""

    def _make_mgr(self) -> ConversationManager:
        engine = _make_compaction_engine()
        return ConversationManager(compaction_engine=engine, raw_window=5)

    # --- Basic seeding ---

    def test_basic_seeding_populates_raw_zone(self):
        """seed_from_history() creates ConversationTurn objects in raw zone."""
        mgr = self._make_mgr()
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        count = mgr.seed_from_history(history)

        assert count == 2
        assert len(mgr.zones().raw) == 2
        assert mgr.zones().raw[0].role == "user"
        assert mgr.zones().raw[0].content == "hello"
        assert mgr.zones().raw[1].role == "assistant"
        assert mgr.zones().raw[1].content == "hi"

    def test_returns_count_of_turns_seeded(self):
        """Return value equals the number of turns added."""
        mgr = self._make_mgr()
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]

        assert mgr.seed_from_history(history) == 3

    def test_turn_numbers_assigned_sequentially(self):
        """Seeded turns get sequential turn_number starting from 0."""
        mgr = self._make_mgr()
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]

        mgr.seed_from_history(history)

        assert [t.turn_number for t in mgr.zones().raw] == [0, 1, 2]

    def test_custom_session_id(self):
        """seed_from_history() respects session_id parameter."""
        mgr = self._make_mgr()
        history = [{"role": "user", "content": "hello"}]

        count = mgr.seed_from_history(history, session_id="session-42")

        assert count == 1
        assert len(mgr.zones("session-42").raw) == 1
        assert len(mgr.zones().raw) == 0  # default session untouched

    # --- Idempotent ---

    def test_idempotent_with_existing_raw(self):
        """Second call returns 0 and doesn't modify zones when raw has content."""
        mgr = self._make_mgr()
        history = [{"role": "user", "content": "hello"}]

        mgr.seed_from_history(history)
        count = mgr.seed_from_history(history)

        assert count == 0
        assert len(mgr.zones().raw) == 1  # unchanged

    def test_idempotent_with_existing_compacted(self):
        """Returns 0 when compacted zone already has content."""
        mgr = self._make_mgr()
        mgr.zones().compacted = [_make_turn(0)]

        count = mgr.seed_from_history([{"role": "user", "content": "hello"}])

        assert count == 0
        assert len(mgr.zones().raw) == 0  # not seeded

    # --- Tool metadata preservation ---

    def test_tool_calls_preserved_in_metadata(self):
        """Assistant tool_calls are stored in ConversationTurn.metadata."""
        mgr = self._make_mgr()
        tool_calls = [{"id": "tc1", "function": {"name": "search"}}]
        history = [
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]

        mgr.seed_from_history(history)

        turn = mgr.zones().raw[0]
        assert turn.metadata is not None
        assert turn.metadata["tool_calls"] == tool_calls

    def test_tool_call_id_preserved_in_metadata(self):
        """Tool result tool_call_id is stored in ConversationTurn.metadata."""
        mgr = self._make_mgr()
        history = [
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]

        mgr.seed_from_history(history)

        turn = mgr.zones().raw[0]
        assert turn.metadata is not None
        assert turn.metadata["tool_call_id"] == "tc1"

    def test_nested_metadata_preserved(self):
        """Existing metadata dict from session history is preserved."""
        mgr = self._make_mgr()
        history = [
            {
                "role": "tool",
                "content": "result",
                "tool_call_id": "tc1",
                "metadata": {"tool_name": "search", "latency_ms": 42},
            },
        ]

        mgr.seed_from_history(history)

        turn = mgr.zones().raw[0]
        assert turn.metadata["tool_call_id"] == "tc1"
        assert turn.metadata["tool_name"] == "search"
        assert turn.metadata["latency_ms"] == 42

    # --- Content type inference ---

    def test_tool_role_gets_tool_output_content_type(self):
        """Role 'tool' gets content_type='tool_output'."""
        mgr = self._make_mgr()
        history = [{"role": "tool", "content": "result", "tool_call_id": "tc1"}]

        mgr.seed_from_history(history)

        assert mgr.zones().raw[0].content_type == "tool_output"

    def test_tool_result_role_gets_tool_output_content_type(self):
        """Role 'tool_result' gets content_type='tool_output'."""
        mgr = self._make_mgr()
        history = [{"role": "tool_result", "content": "result"}]

        mgr.seed_from_history(history)

        assert mgr.zones().raw[0].content_type == "tool_output"

    def test_user_role_gets_none_content_type(self):
        """Non-tool roles get content_type=None."""
        mgr = self._make_mgr()
        history = [{"role": "user", "content": "hello"}]

        mgr.seed_from_history(history)

        assert mgr.zones().raw[0].content_type is None

    def test_assistant_role_gets_none_content_type(self):
        """Assistant role gets content_type=None."""
        mgr = self._make_mgr()
        history = [{"role": "assistant", "content": "hi"}]

        mgr.seed_from_history(history)

        assert mgr.zones().raw[0].content_type is None

    # --- Edge cases ---

    def test_empty_history_returns_zero(self):
        """Empty list returns 0 and zones stay empty."""
        mgr = self._make_mgr()

        count = mgr.seed_from_history([])

        assert count == 0
        assert len(mgr.zones().raw) == 0

    def test_none_content_becomes_empty_string(self):
        """Message with None content is handled — content becomes ''."""
        mgr = self._make_mgr()
        history = [{"role": "assistant", "content": None}]

        mgr.seed_from_history(history)

        assert mgr.zones().raw[0].content == ""

    def test_missing_content_key_becomes_empty_string(self):
        """Message with no 'content' key is handled — content becomes ''."""
        mgr = self._make_mgr()
        history = [{"role": "user"}]

        mgr.seed_from_history(history)

        assert mgr.zones().raw[0].content == ""
