"""Post-LLM async processor for memory extraction, compaction, and summarization."""

import asyncio
import logging

from sr2.compaction.engine import CompactionResult, ConversationTurn
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.retrieval import HybridRetriever
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.result import PipelineResult, StageResult
from sr2.summarization.engine import SummarizationResult

logger = logging.getLogger(__name__)


class PostLLMProcessor:
    """Runs async after the LLM response is sent to the user.

    Steps (all non-blocking, all tolerant of individual failures):
    1. Memory extraction -> extract facts from the latest turn
    2. Conflict detection + resolution -> check and resolve conflicts
    3. Compaction -> compact turns outside raw window
    4. Summarization check -> trigger if needed
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        memory_extractor: MemoryExtractor | None = None,
        conflict_detector: ConflictDetector | None = None,
        conflict_resolver: ConflictResolver | None = None,
        retriever: HybridRetriever | None = None,
        extraction_batch_size: int = 5,
        extraction_mutex: bool = True,
        trace_collector=None,
    ):
        self._conv = conversation_manager
        self._extractor = memory_extractor
        self._detector = conflict_detector
        self._resolver = conflict_resolver
        self._retriever = retriever
        self._trace = trace_collector
        self._extraction_batch_size = extraction_batch_size
        # Cursor tracking: last turn index processed for extraction, per session
        self._extraction_cursor: dict[str, int] = {}
        # Mutual exclusion: prevent concurrent extraction per session
        self._extraction_locks: dict[str, asyncio.Lock] = {}
        self._extraction_mutex_enabled = extraction_mutex
        # Counters for memory metrics (per-invocation, reset on each process call)
        self.last_memories_extracted: int = 0
        self.last_conflicts_detected: int = 0
        # Last compaction/summarization results for metrics collection
        self.last_compaction_result: "CompactionResult | None" = None
        self.last_summarization_result: "SummarizationResult | None" = None

    async def process(
        self,
        latest_turn: ConversationTurn,
        conversation_id: str | None = None,
        current_context: dict | None = None,
        extract_only: bool = False,
    ) -> PipelineResult:
        """Run all post-LLM steps. Each step is independent — failures don't block others.

        Args:
            extract_only: When True, only run memory extraction (skip compaction
                and summarization). Used by ephemeral agents that run one task
                and exit — they need to persist memories but don't need
                conversation management.
        """
        self._current_context = current_context
        result = PipelineResult(config_used="post_llm")
        session_id = conversation_id or "default"

        # Reset per-invocation counters
        self.last_memories_extracted = 0
        self.last_conflicts_detected = 0
        self.last_compaction_result = None
        self.last_summarization_result = None

        # 1. Add turn to conversation
        self._conv.add_turn(latest_turn, session_id=session_id)

        # 2. Flush deferred memory touches from retrieval
        await self._run_stage(
            result,
            "memory_touch",
            self._flush_touches,
        )

        # 3. Memory extraction
        await self._run_stage(
            result,
            "memory_extraction",
            self._extract_memories,
            latest_turn,
            conversation_id,
        )

        if not extract_only:
            # 4. Compaction
            await self._run_stage(
                result,
                "compaction",
                self._run_compaction,
                session_id,
            )

            # 5. Summarization
            await self._run_stage(
                result,
                "summarization",
                self._run_summarization,
                session_id,
            )

        return result

    async def _run_stage(self, result: PipelineResult, name: str, coro, *args) -> None:
        """Run a stage, catching any errors."""
        try:
            await coro(*args)
            result.add_stage(StageResult(stage_name=name, status="success"))
        except Exception as e:
            logger.warning(f"Post-LLM stage '{name}' failed: {e}", exc_info=True)
            result.add_stage(StageResult(stage_name=name, status="failed", error=str(e)))

    async def _flush_touches(self) -> None:
        if self._retriever:
            await self._retriever.flush_touches()

    async def _extract_memories(
        self,
        turn: ConversationTurn,
        conversation_id: str | None,
    ) -> None:
        if not self._extractor:
            return

        session_id = conversation_id or "default"

        # Mutual exclusion: skip if another extraction is already running
        if self._extraction_mutex_enabled:
            lock = self._extraction_locks.setdefault(session_id, asyncio.Lock())
            if lock.locked():
                logger.debug(
                    "Extraction skipped for session=%s (another extraction in progress)",
                    session_id,
                )
                return
            async with lock:
                await self._do_extract(turn, conversation_id, session_id)
        else:
            await self._do_extract(turn, conversation_id, session_id)

    async def _do_extract(
        self,
        turn: ConversationTurn,
        conversation_id: str | None,
        session_id: str,
    ) -> None:
        """Execute extraction with cursor tracking and batching."""
        # Get all turns from conversation manager
        all_turns = self._conv.get_all_turns(session_id=session_id)

        # Cursor: only process turns after the last extracted index
        cursor = self._extraction_cursor.get(session_id, -1)
        unprocessed = [t for t in all_turns if t.turn_number > cursor]

        if not unprocessed:
            return

        # Batch: limit to batch_size turns
        batch = unprocessed[: self._extraction_batch_size]
        logger.debug(
            "Extraction: session=%s, cursor=%d, unprocessed=%d, batch=%d",
            session_id,
            cursor,
            len(unprocessed),
            len(batch),
        )

        for t in batch:
            extraction = await self._extractor.extract(
                conversation_turn=t.content,
                conversation_id=conversation_id,
                turn_number=t.turn_number,
                current_context=self._current_context,
            )
            self.last_memories_extracted += len(extraction.memories)
            if self._detector and self._resolver:
                for mem in extraction.memories:
                    conflicts = await self._detector.detect(mem)
                    if conflicts:
                        self.last_conflicts_detected += len(conflicts)
                        await self._resolver.resolve_all(conflicts)

        # Advance cursor to the last processed turn
        self._extraction_cursor[session_id] = batch[-1].turn_number

    def get_extraction_cursor(self, session_id: str = "default") -> int:
        """Get the current extraction cursor for a session."""
        return self._extraction_cursor.get(session_id, -1)

    def set_extraction_cursor(self, session_id: str, cursor: int) -> None:
        """Restore extraction cursor (e.g. after deserialization)."""
        self._extraction_cursor[session_id] = cursor

    async def _run_compaction(self, session_id: str) -> None:
        self.last_compaction_result = self._conv.run_compaction(session_id=session_id)

    async def _run_summarization(self, session_id: str) -> None:
        self.last_summarization_result = await self._conv.run_summarization(session_id=session_id)
