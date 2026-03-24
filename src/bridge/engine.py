"""BridgeEngine — format-agnostic context optimization for proxied requests."""

from __future__ import annotations

import logging

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, PipelineConfig, SummarizationConfig
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.degradation.ladder import DegradationLadder
from sr2.memory.conflicts import ConflictDetector
from sr2.memory.extraction import MemoryExtractor
from sr2.memory.resolution import ConflictResolver
from sr2.memory.retrieval import HybridRetriever
from sr2.memory.store import SQLiteMemoryStore
from sr2.pipeline.conversation import ConversationManager
from sr2.summarization.engine import SummarizationEngine

from bridge.adapters.base import BridgeAdapter
from bridge.config import BridgeConfig
from bridge.llm import (
    APIKeyCache,
    make_embedding_callable,
    make_extraction_callable,
    make_summarization_callable,
)
from bridge.session_tracker import BridgeSession

logger = logging.getLogger(__name__)

# Default compaction rules for bridge proxy conversations.
# Claude Code traffic is heavy on tool outputs and file content — these
# rules compress older tool results while keeping recent ones in full.
BRIDGE_DEFAULT_COMPACTION_RULES = [
    CompactionRuleConfig(
        type="tool_output",
        strategy="schema_and_sample",
        max_compacted_tokens=80,
        recovery_hint=True,
    ),
    CompactionRuleConfig(
        type="file_content",
        strategy="reference",
        recovery_hint=True,
    ),
    CompactionRuleConfig(
        type="code_execution",
        strategy="result_summary",
        max_output_lines=5,
        recovery_hint=True,
    ),
]


class BridgeEngine:
    """Applies SR2 context optimization to proxied LLM requests.

    On each request (Claude Code sends full history every time), the engine:
    1. Compares incoming message count to last known count
    2. Delegates wire-format conversion to the adapter
    3. Runs compaction on turns outside raw_window
    4. Checks summarization trigger (guarded by circuit breaker)
    5. Asks the adapter to rebuild the message array from zones
    6. Post-process: extract memories from assistant response
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        bridge_config: BridgeConfig | None = None,
        key_cache: APIKeyCache | None = None,
    ):
        self._config = pipeline_config
        self._bridge_config = bridge_config or BridgeConfig()
        self._key_cache = key_cache or APIKeyCache()
        self._memory_initialized = False

        # Build compaction engine — apply bridge defaults if no rules configured
        compaction_config = pipeline_config.compaction or CompactionConfig()
        if not compaction_config.rules:
            compaction_config = compaction_config.model_copy(
                update={"rules": BRIDGE_DEFAULT_COMPACTION_RULES}
            )
        self._compaction = CompactionEngine(compaction_config)

        # Build summarization callable from bridge LLM config
        llm_callable = None
        if self._bridge_config.llm.summarization:
            llm_callable = make_summarization_callable(
                self._bridge_config.llm.summarization,
                self._key_cache,
                self._bridge_config.forwarding.upstream_url,
            )

        # Build summarization engine (optional — requires llm_callable)
        summarization_config = pipeline_config.summarization or SummarizationConfig()
        self._summarization: SummarizationEngine | None = None
        if summarization_config.enabled and llm_callable:
            self._summarization = SummarizationEngine(
                config=summarization_config,
                llm_callable=llm_callable,
            )

        # Build conversation manager
        self._conversation = ConversationManager(
            compaction_engine=self._compaction,
            summarization_engine=self._summarization,
            raw_window=compaction_config.raw_window,
        )

        # Degradation: circuit breaker + ladder
        deg = self._bridge_config.degradation
        self._breaker = CircuitBreaker(
            threshold=deg.circuit_breaker_threshold,
            cooldown_seconds=deg.circuit_breaker_cooldown_seconds,
        )
        self._ladder = DegradationLadder()

        # Memory system (deferred init — SQLite connect is async)
        mem_cfg = self._bridge_config.memory
        self._memory_store: SQLiteMemoryStore | None = None
        self._memory_extractor: MemoryExtractor | None = None
        self._conflict_detector: ConflictDetector | None = None
        self._conflict_resolver: ConflictResolver | None = None
        self._retriever: HybridRetriever | None = None

        if mem_cfg.enabled and self._bridge_config.llm.extraction:
            self._memory_store = SQLiteMemoryStore(db_path=mem_cfg.db_path)

    async def _ensure_memory_initialized(self) -> None:
        """Lazily connect to SQLite on first use."""
        if self._memory_initialized or not self._memory_store:
            return

        await self._memory_store.connect()

        extraction_callable = make_extraction_callable(
            self._bridge_config.llm.extraction,
            self._key_cache,
            self._bridge_config.forwarding.upstream_url,
        )

        mem_cfg = self._bridge_config.memory
        self._memory_extractor = MemoryExtractor(
            llm_callable=extraction_callable,
            store=self._memory_store,
            max_memories_per_turn=mem_cfg.max_memories_per_turn,
        )
        self._conflict_detector = ConflictDetector(store=self._memory_store)
        self._conflict_resolver = ConflictResolver(store=self._memory_store)

        # Build retriever
        embed_callable = None
        if self._bridge_config.llm.embedding:
            embed_callable = make_embedding_callable(
                self._bridge_config.llm.embedding,
                self._key_cache,
                self._bridge_config.forwarding.upstream_url,
            )
        self._retriever = HybridRetriever(
            store=self._memory_store,
            embedding_callable=embed_callable,
            strategy=mem_cfg.retrieval_strategy,
            top_k=mem_cfg.retrieval_top_k,
        )

        self._memory_initialized = True
        logger.info("Memory system initialized (db=%s)", mem_cfg.db_path)

    async def optimize(
        self,
        system: str | None,
        messages: list[dict],
        session: BridgeSession,
        adapter: BridgeAdapter,
    ) -> tuple[str | None, list[dict]]:
        """Optimize messages using SR2 pipeline.

        Returns (system_injection_or_None, optimized_messages).
        """
        session_id = session.session_id
        current_count = len(messages)

        if current_count <= session.last_message_count:
            if current_count < session.last_message_count:
                logger.info(
                    "Session %s: message count decreased (%d -> %d), resetting",
                    session_id, session.last_message_count, current_count,
                )
                self._reset_session(session)

        # Convert new messages to turns via adapter (engine never sees wire format)
        new_messages = messages[session.last_message_count:]
        new_turns = adapter.messages_to_turns(new_messages, session.turn_counter)
        for turn in new_turns:
            self._conversation.add_turn(turn, session_id)
        session.turn_counter += len(new_turns)
        session.last_message_count = current_count

        # Run compaction (not gated — compaction is local, no external calls)
        if not self._ladder.should_skip("compaction"):
            compaction_result = self._conversation.run_compaction(session_id)
            if compaction_result and compaction_result.turns_compacted > 0:
                logger.info(
                    "Session %s: compacted %d turns (%d -> %d tokens)",
                    session_id,
                    compaction_result.turns_compacted,
                    compaction_result.original_tokens,
                    compaction_result.compacted_tokens,
                )

        # Run summarization (guarded by circuit breaker + degradation ladder)
        if not self._ladder.should_skip("summarization") and not self._breaker.is_open(
            "summarization"
        ):
            try:
                summarization_result = await self._conversation.run_summarization(session_id)
                if summarization_result:
                    logger.info(
                        "Session %s: summarized turns %s (%d -> %d tokens)",
                        session_id,
                        summarization_result.turn_range,
                        summarization_result.original_tokens,
                        summarization_result.summary_tokens,
                    )
                    self._breaker.record_success("summarization")
            except Exception:
                logger.warning(
                    "Session %s: summarization failed", session_id, exc_info=True
                )
                self._breaker.record_failure("summarization")
                self._ladder.degrade()

        # Memory retrieval (guarded by circuit breaker + degradation)
        memory_injection: str | None = None
        if (
            not self._ladder.should_skip("memory")
            and not self._breaker.is_open("memory_retrieval")
            and self._memory_store
        ):
            await self._ensure_memory_initialized()
            if self._retriever:
                try:
                    query = self._extract_retrieval_query(messages)
                    if query:
                        mem_cfg = self._bridge_config.memory
                        results = await self._retriever.retrieve(
                            query,
                            top_k=mem_cfg.retrieval_top_k,
                            max_tokens=mem_cfg.retrieval_max_tokens,
                        )
                        if results:
                            memory_lines = [
                                f"- {r.memory.key}: {r.memory.value}"
                                for r in results
                            ]
                            memory_injection = (
                                "[Relevant memories from previous sessions]\n"
                                + "\n".join(memory_lines)
                                + "\n[End of memories]"
                            )
                            logger.info(
                                "Session %s: retrieved %d memories",
                                session_id, len(results),
                            )
                        self._breaker.record_success("memory_retrieval")
                except Exception:
                    logger.warning(
                        "Session %s: memory retrieval failed",
                        session_id, exc_info=True,
                    )
                    self._breaker.record_failure("memory_retrieval")

        # Rebuild message list from zones
        zones = self._conversation.zones(session_id)

        # Inject summaries and memories as system prompt context
        system_injection: str | None = None
        injection_parts: list[str] = []
        if zones.summarized:
            summary_text = "\n\n".join(zones.summarized)
            injection_parts.append(
                f"[Previous conversation summary]\n{summary_text}\n"
                f"[End of summary — recent conversation follows]"
            )
        if memory_injection:
            injection_parts.append(memory_injection)
        if injection_parts:
            system_injection = "\n\n".join(injection_parts)

        # Convert turns back to wire format via adapter
        all_turns = list(zones.compacted) + list(zones.raw)
        optimized = adapter.turns_to_messages(all_turns, messages)

        # If we have no optimized messages, fall through to original
        if not optimized:
            return None, messages

        return system_injection, optimized

    async def post_process(self, session: BridgeSession, assistant_text: str) -> None:
        """Post-process after a response stream completes.

        1. Track assistant turn
        2. Extract memories (guarded by circuit breaker)
        """
        turn = ConversationTurn(
            turn_number=session.turn_counter,
            role="assistant",
            content=assistant_text,
        )
        session.turn_counter += 1

        # Memory extraction (guarded)
        if (
            not self._ladder.should_skip("memory")
            and not self._breaker.is_open("memory_extraction")
        ):
            await self._ensure_memory_initialized()
            if self._memory_extractor:
                try:
                    result = await self._memory_extractor.extract(
                        conversation_turn=assistant_text,
                        conversation_id=session.session_id,
                        turn_number=turn.turn_number,
                    )
                    if result.memories:
                        logger.info(
                            "Session %s: extracted %d memories",
                            session.session_id, len(result.memories),
                        )
                        # Conflict detection + resolution
                        if self._conflict_detector and self._conflict_resolver:
                            for mem in result.memories:
                                conflicts = await self._conflict_detector.detect(mem)
                                if conflicts:
                                    await self._conflict_resolver.resolve_all(conflicts)
                    self._breaker.record_success("memory_extraction")
                except Exception:
                    logger.warning(
                        "Session %s: memory extraction failed",
                        session.session_id, exc_info=True,
                    )
                    self._breaker.record_failure("memory_extraction")

    @staticmethod
    def _extract_retrieval_query(messages: list[dict]) -> str | None:
        """Extract text from the latest user message for memory retrieval."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:500]  # cap query length
                if isinstance(content, list):
                    # Extract text blocks from content array
                    texts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if texts:
                        return " ".join(texts)[:500]
        return None

    def _reset_session(self, session: BridgeSession) -> None:
        """Reset conversation state for a session."""
        self._conversation.destroy_session(session.session_id)
        session.last_message_count = 0
        session.turn_counter = 0
        session.turns = []

    def destroy_session(self, session_id: str) -> None:
        """Clean up ConversationManager state for a session."""
        self._conversation.destroy_session(session_id)

    async def shutdown(self) -> None:
        """Clean up resources (close DB connections)."""
        if self._memory_store:
            await self._memory_store.disconnect()

    def get_session_metrics(self, session: BridgeSession) -> dict:
        """Return metrics for a specific session."""
        zones = self._conversation.zones(session.session_id)
        return {
            "message_count": session.last_message_count,
            "turn_counter": session.turn_counter,
            "summarized_count": len(zones.summarized),
            "compacted_count": len(zones.compacted),
            "raw_count": len(zones.raw),
            "total_tokens": zones.total_tokens,
            "zone_transitions": self._conversation.get_zone_transitions(session.session_id),
        }

    @property
    def degradation_level(self) -> str:
        return self._ladder.level

    @property
    def circuit_breaker_status(self) -> dict:
        return self._breaker.status()
