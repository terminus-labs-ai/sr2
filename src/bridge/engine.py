"""BridgeEngine — format-agnostic context optimization for proxied requests."""

from __future__ import annotations

import logging

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, PipelineConfig, SummarizationConfig
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.degradation.ladder import DegradationLadder
from sr2.pipeline.conversation import ConversationManager
from sr2.summarization.engine import SummarizationEngine

from bridge.adapters.base import BridgeAdapter
from bridge.config import BridgeDegradationConfig
from bridge.session_tracker import BridgeSession

logger = logging.getLogger(__name__)


class BridgeEngine:
    """Applies SR2 context optimization to proxied LLM requests.

    On each request (Claude Code sends full history every time), the engine:
    1. Compares incoming message count to last known count
    2. Delegates wire-format conversion to the adapter
    3. Runs compaction on turns outside raw_window
    4. Checks summarization trigger (guarded by circuit breaker)
    5. Asks the adapter to rebuild the message array from zones
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        llm_callable=None,
        degradation_config: BridgeDegradationConfig | None = None,
    ):
        self._config = pipeline_config
        self._llm_callable = llm_callable

        # Build compaction engine
        compaction_config = pipeline_config.compaction or CompactionConfig()
        self._compaction = CompactionEngine(compaction_config)

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
        deg = degradation_config or BridgeDegradationConfig()
        self._breaker = CircuitBreaker(
            threshold=deg.circuit_breaker_threshold,
            cooldown_seconds=deg.circuit_breaker_cooldown_seconds,
        )
        self._ladder = DegradationLadder()

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

        # Rebuild message list from zones
        zones = self._conversation.zones(session_id)

        # Inject summaries as system prompt context
        system_injection: str | None = None
        if zones.summarized:
            summary_text = "\n\n".join(zones.summarized)
            system_injection = (
                f"[Previous conversation summary]\n{summary_text}\n"
                f"[End of summary — recent conversation follows]"
            )

        # Convert turns back to wire format via adapter
        all_turns = list(zones.compacted) + list(zones.raw)
        optimized = adapter.turns_to_messages(all_turns, messages)

        # If we have no optimized messages, fall through to original
        if not optimized:
            return None, messages

        return system_injection, optimized

    async def post_process(self, session: BridgeSession, assistant_text: str) -> None:
        """Post-process after a response stream completes.

        Adds the assistant response as a turn so it's tracked for future
        compaction/summarization.
        """
        turn = ConversationTurn(
            turn_number=session.turn_counter,
            role="assistant",
            content=assistant_text,
        )
        session.turn_counter += 1

    def _reset_session(self, session: BridgeSession) -> None:
        """Reset conversation state for a session."""
        self._conversation.destroy_session(session.session_id)
        session.last_message_count = 0
        session.turn_counter = 0
        session.turns = []

    def destroy_session(self, session_id: str) -> None:
        """Clean up ConversationManager state for a session."""
        self._conversation.destroy_session(session_id)

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
