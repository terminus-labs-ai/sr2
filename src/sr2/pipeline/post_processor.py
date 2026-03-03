"""Post-LLM async processor for memory extraction, compaction, and summarization."""

import logging

from sr2.compaction.engine import ConversationTurn
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.pipeline.conversation import ConversationManager
from sr2.pipeline.result import PipelineResult, StageResult

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
    ):
        self._conv = conversation_manager
        self._extractor = memory_extractor
        self._detector = conflict_detector
        self._resolver = conflict_resolver
        # Counters for memory metrics (per-invocation, reset on each process call)
        self.last_memories_extracted: int = 0
        self.last_conflicts_detected: int = 0

    async def process(
        self,
        latest_turn: ConversationTurn,
        conversation_id: str | None = None,
    ) -> PipelineResult:
        """Run all post-LLM steps. Each step is independent — failures don't block others."""
        result = PipelineResult(config_used="post_llm")
        session_id = conversation_id or "default"

        # Reset per-invocation counters
        self.last_memories_extracted = 0
        self.last_conflicts_detected = 0

        # 1. Add turn to conversation
        self._conv.add_turn(latest_turn, session_id=session_id)

        # 2. Memory extraction
        await self._run_stage(
            result,
            "memory_extraction",
            self._extract_memories,
            latest_turn,
            conversation_id,
        )

        # 3. Compaction
        await self._run_stage(
            result,
            "compaction",
            self._run_compaction,
            session_id,
        )

        # 4. Summarization
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
            logger.warning(f"Post-LLM stage '{name}' failed: {e}")
            result.add_stage(StageResult(stage_name=name, status="failed", error=str(e)))

    async def _extract_memories(
        self,
        turn: ConversationTurn,
        conversation_id: str | None,
    ) -> None:
        if not self._extractor:
            return
        extraction = await self._extractor.extract(
            conversation_turn=turn.content,
            conversation_id=conversation_id,
            turn_number=turn.turn_number,
        )
        self.last_memories_extracted = len(extraction.memories)
        if self._detector and self._resolver:
            for mem in extraction.memories:
                conflicts = await self._detector.detect(mem)
                if conflicts:
                    self.last_conflicts_detected += len(conflicts)
                    await self._resolver.resolve_all(conflicts)

    async def _run_compaction(self, session_id: str) -> None:
        self._conv.run_compaction(session_id=session_id)

    async def _run_summarization(self, session_id: str) -> None:
        await self._conv.run_summarization(session_id=session_id)
