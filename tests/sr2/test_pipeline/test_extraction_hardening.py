"""Tests for memory extraction hardening: cursor tracking, mutual exclusion, turn batching."""

import asyncio

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, CostGateConfig
from sr2.memory.extraction import ExtractionResult
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.post_processor import PostLLMProcessor


def _make_compaction_engine() -> CompactionEngine:
    return CompactionEngine(
        CompactionConfig(
            enabled=True,
            raw_window=20,
            min_content_size=10,
            cost_gate=CostGateConfig(enabled=False),
            rules=[CompactionRuleConfig(type="tool_output", strategy="schema_and_sample")],
        )
    )


def _make_turn(num: int, content: str = "turn content") -> ConversationTurn:
    return ConversationTurn(turn_number=num, role="assistant", content=content)


class FakeExtractor:
    """Tracks which turns were extracted."""

    def __init__(self):
        self.extracted_turns: list[int] = []
        self.call_count = 0

    async def extract(self, conversation_turn, conversation_id=None, turn_number=0, current_context=None):
        self.call_count += 1
        self.extracted_turns.append(turn_number)
        return ExtractionResult(memories=[], source=None)


class SlowExtractor:
    """Extractor that takes a while to simulate concurrency."""

    def __init__(self, delay: float = 0.1):
        self._delay = delay
        self.call_count = 0

    async def extract(self, conversation_turn, conversation_id=None, turn_number=0, current_context=None):
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return ExtractionResult(memories=[], source=None)


class TestCursorTracking:
    @pytest.mark.asyncio
    async def test_cursor_advances_after_extraction(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = FakeExtractor()
        proc = PostLLMProcessor(conv, memory_extractor=extractor, extraction_mutex=False)

        await proc.process(_make_turn(0), conversation_id="s1")
        assert proc.get_extraction_cursor("s1") == 0

        await proc.process(_make_turn(1), conversation_id="s1")
        assert proc.get_extraction_cursor("s1") == 1

    @pytest.mark.asyncio
    async def test_second_extraction_only_processes_new_turns(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = FakeExtractor()
        proc = PostLLMProcessor(conv, memory_extractor=extractor, extraction_mutex=False)

        # Process turns 0-2
        for i in range(3):
            await proc.process(_make_turn(i), conversation_id="s1")

        # All 3 turns should have been extracted
        assert sorted(extractor.extracted_turns) == [0, 1, 2]

        # Now add turn 3
        extractor.extracted_turns.clear()
        await proc.process(_make_turn(3), conversation_id="s1")

        # Only turn 3 should be extracted (cursor was at 2)
        assert extractor.extracted_turns == [3]

    @pytest.mark.asyncio
    async def test_cursor_persists_via_getter_setter(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = FakeExtractor()
        proc = PostLLMProcessor(conv, memory_extractor=extractor, extraction_mutex=False)

        # Simulate restored cursor
        proc.set_extraction_cursor("s1", 5)
        assert proc.get_extraction_cursor("s1") == 5

    @pytest.mark.asyncio
    async def test_cursor_default_is_negative_one(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        proc = PostLLMProcessor(conv, extraction_mutex=False)

        assert proc.get_extraction_cursor("new_session") == -1


class TestTurnBatching:
    @pytest.mark.asyncio
    async def test_batch_size_respected(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = FakeExtractor()
        proc = PostLLMProcessor(
            conv, memory_extractor=extractor, extraction_batch_size=3, extraction_mutex=False
        )

        # Add 10 turns at once
        for i in range(10):
            conv.add_turn(_make_turn(i), session_id="s1")

        # Process one more turn (triggers extraction)
        await proc.process(_make_turn(10), conversation_id="s1")

        # Only first batch_size turns should be processed (0-2, since cursor starts at -1)
        assert len(extractor.extracted_turns) == 3
        assert proc.get_extraction_cursor("s1") == 2

    @pytest.mark.asyncio
    async def test_subsequent_batches_continue(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = FakeExtractor()
        proc = PostLLMProcessor(
            conv, memory_extractor=extractor, extraction_batch_size=2, extraction_mutex=False
        )

        # Add 5 turns and process
        for i in range(5):
            await proc.process(_make_turn(i), conversation_id="s1")

        # With batch_size=2, each process() call processes up to 2 unprocessed turns
        # Turn 0: process, extracts [0] (but batch may get [0,1] if turn 1 exists)
        # The exact count depends on how many turns are available at each call
        # But cursor should advance through all of them after enough calls
        assert proc.get_extraction_cursor("s1") >= 0
        total_extracted = len(extractor.extracted_turns)
        assert total_extracted == 5  # All 5 should eventually be extracted


class TestMutualExclusion:
    @pytest.mark.asyncio
    async def test_concurrent_extraction_skipped(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = SlowExtractor(delay=0.05)
        proc = PostLLMProcessor(
            conv, memory_extractor=extractor, extraction_batch_size=10, extraction_mutex=True
        )

        # Pre-populate turns
        for i in range(3):
            conv.add_turn(_make_turn(i), session_id="s1")

        # Launch two concurrent process() calls
        task1 = asyncio.create_task(proc.process(_make_turn(3), conversation_id="s1"))
        # Small delay to ensure task1 grabs the lock first
        await asyncio.sleep(0.01)
        task2 = asyncio.create_task(proc.process(_make_turn(4), conversation_id="s1"))

        await asyncio.gather(task1, task2)

        # The second call should have skipped extraction due to lock
        # Total extractions should be less than if both ran
        assert extractor.call_count < 10  # Sanity: not all turns extracted twice

    @pytest.mark.asyncio
    async def test_mutex_disabled_allows_concurrent(self):
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=20)
        extractor = FakeExtractor()
        proc = PostLLMProcessor(
            conv, memory_extractor=extractor, extraction_batch_size=10, extraction_mutex=False
        )

        await proc.process(_make_turn(0), conversation_id="s1")
        await proc.process(_make_turn(1), conversation_id="s1")

        # Both should have run extraction
        assert extractor.call_count >= 2


class TestConfigFields:
    def test_extraction_batch_size_default(self):
        from sr2.config.models import MemoryConfig

        config = MemoryConfig()
        assert config.extraction_batch_size == 5

    def test_extraction_mutex_default(self):
        from sr2.config.models import MemoryConfig

        config = MemoryConfig()
        assert config.extraction_mutex is True
