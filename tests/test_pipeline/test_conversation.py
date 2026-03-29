"""Tests for three-zone conversation manager."""

import json

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, SummarizationConfig
from sr2.pipeline.conversation import ConversationManager, ConversationZones
from sr2.summarization.engine import SummarizationEngine


def _make_compaction_engine(raw_window: int = 3) -> CompactionEngine:
    config = CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=10,
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
        summ_config = SummarizationConfig(trigger="token_threshold", threshold=0.5, preserve_recent_turns=0)
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
        summ_config = SummarizationConfig(trigger="token_threshold", threshold=0.5, preserve_recent_turns=0)
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
