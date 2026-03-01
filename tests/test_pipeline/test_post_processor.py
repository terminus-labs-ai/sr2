"""Tests for post-LLM async processor."""

import json

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.store import InMemoryMemoryStore
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.post_processor import PostLLMProcessor


def _make_compaction_engine(raw_window: int = 3) -> CompactionEngine:
    config = CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=10,
        rules=[],
    )
    return CompactionEngine(config)


def _make_turn(num: int = 0, content: str = "I work at Anthropic") -> ConversationTurn:
    return ConversationTurn(turn_number=num, role="assistant", content=content)


class TestPostLLMProcessor:
    """Tests for PostLLMProcessor."""

    @pytest.mark.asyncio
    async def test_all_stages_succeed(self):
        """Process with all components -> all stages succeed."""
        store = InMemoryMemoryStore()

        async def mock_llm(prompt: str) -> str:
            return json.dumps([
                {"key": "user.employer", "value": "Anthropic", "memory_type": "identity"}
            ])

        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)
        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        detector = ConflictDetector(store=store)
        resolver = ConflictResolver(store=store)

        processor = PostLLMProcessor(
            conversation_manager=conv,
            memory_extractor=extractor,
            conflict_detector=detector,
            conflict_resolver=resolver,
        )

        turn = _make_turn()
        result = await processor.process(turn, conversation_id="conv_1")

        assert len(result.stages) == 3
        assert all(s.status == "success" for s in result.stages)

    @pytest.mark.asyncio
    async def test_without_memory_extractor(self):
        """Process without memory extractor -> extraction skipped, others run."""
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)
        processor = PostLLMProcessor(conversation_manager=conv)

        turn = _make_turn()
        result = await processor.process(turn)

        assert len(result.stages) == 3
        # All should succeed (extraction skips gracefully)
        assert all(s.status == "success" for s in result.stages)

    @pytest.mark.asyncio
    async def test_extraction_failure_doesnt_block_compaction(self):
        """Memory extraction failure -> logged, compaction still runs."""
        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        store = InMemoryMemoryStore()
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)
        extractor = MemoryExtractor(llm_callable=failing_llm, store=store)

        processor = PostLLMProcessor(
            conversation_manager=conv,
            memory_extractor=extractor,
        )

        turn = _make_turn()
        result = await processor.process(turn)

        assert result.stages[0].stage_name == "memory_extraction"
        assert result.stages[0].status == "failed"
        assert result.stages[1].stage_name == "compaction"
        assert result.stages[1].status == "success"

    @pytest.mark.asyncio
    async def test_compaction_failure_doesnt_block_summarization(self):
        """Compaction failure -> logged, summarization still runs."""
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)

        # Monkey-patch compaction to fail
        original_run = conv.run_compaction
        def failing_compaction():
            raise RuntimeError("Compaction broke")
        conv.run_compaction = failing_compaction

        processor = PostLLMProcessor(conversation_manager=conv)
        turn = _make_turn()
        result = await processor.process(turn)

        assert result.stages[1].stage_name == "compaction"
        assert result.stages[1].status == "failed"
        assert result.stages[2].stage_name == "summarization"
        assert result.stages[2].status == "success"

    @pytest.mark.asyncio
    async def test_all_stages_failed(self):
        """All stages failed -> PipelineResult has 3 failed stages."""
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)

        async def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM down")

        store = InMemoryMemoryStore()
        extractor = MemoryExtractor(llm_callable=failing_llm, store=store)

        def failing_compaction():
            raise RuntimeError("Compaction error")
        conv.run_compaction = failing_compaction

        async def failing_summarization():
            raise RuntimeError("Summarization error")
        conv.run_summarization = failing_summarization

        processor = PostLLMProcessor(
            conversation_manager=conv,
            memory_extractor=extractor,
        )

        turn = _make_turn()
        result = await processor.process(turn)

        assert len(result.stages) == 3
        assert all(s.status == "failed" for s in result.stages)

    @pytest.mark.asyncio
    async def test_turn_added_to_conversation(self):
        """Turn added to conversation manager during process."""
        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)
        processor = PostLLMProcessor(conversation_manager=conv)

        turn = _make_turn(num=5, content="Hello")
        await processor.process(turn)

        assert len(conv.zones().raw) == 1
        assert conv.zones().raw[0].turn_number == 5

    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self):
        """Extracted memories go through conflict detection + resolution."""
        store = InMemoryMemoryStore()

        # Pre-populate with conflicting memory
        from sr2.memory.schema import Memory
        existing = Memory(key="user.employer", value="Google", memory_type="identity")
        await store.save(existing)

        async def mock_llm(prompt: str) -> str:
            return json.dumps([
                {"key": "user.employer", "value": "Anthropic", "memory_type": "identity"}
            ])

        engine = _make_compaction_engine()
        conv = ConversationManager(compaction_engine=engine, raw_window=3)
        extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
        detector = ConflictDetector(store=store)
        resolver = ConflictResolver(store=store)

        processor = PostLLMProcessor(
            conversation_manager=conv,
            memory_extractor=extractor,
            conflict_detector=detector,
            conflict_resolver=resolver,
        )

        turn = _make_turn(content="I now work at Anthropic")
        await processor.process(turn, conversation_id="conv_1")

        # The old memory should be archived (identity -> latest_wins_archive)
        old = await store.get(existing.id)
        assert old.archived is True
