"""BridgeEngine — format-agnostic context optimization for proxied requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, PipelineConfig, SummarizationConfig
from sr2.pipeline.conversation import ConversationManager
from sr2.summarization.engine import SummarizationEngine

from runtime.bridge.adapters.base import BridgeAdapter

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Per-session tracking for the bridge engine."""

    last_message_count: int = 0
    turn_counter: int = 0
    turns: list[ConversationTurn] = field(default_factory=list)


class BridgeEngine:
    """Applies SR2 context optimization to proxied LLM requests.

    On each request (Claude Code sends full history every time), the engine:
    1. Compares incoming message count to last known count
    2. Converts new messages to SR2 ConversationTurns
    3. Runs compaction on turns outside raw_window
    4. Checks summarization trigger
    5. Rebuilds the message array: [summary exchange] + [compacted] + [raw recent]
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        llm_callable=None,
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

        # Per-session state (message count tracking)
        self._session_states: dict[str, SessionState] = {}

    def _get_state(self, session_id: str) -> SessionState:
        if session_id not in self._session_states:
            self._session_states[session_id] = SessionState()
        return self._session_states[session_id]

    async def optimize(
        self,
        system: str | None,
        messages: list[dict],
        session_id: str,
        adapter: BridgeAdapter,
    ) -> tuple[str | None, list[dict]]:
        """Optimize messages using SR2 pipeline.

        Returns (system_injection_or_None, optimized_messages).
        """
        state = self._get_state(session_id)
        current_count = len(messages)

        if current_count <= state.last_message_count:
            # No new messages (or history was reset). If shorter, reset session.
            if current_count < state.last_message_count:
                logger.info(
                    "Session %s: message count decreased (%d -> %d), resetting",
                    session_id, state.last_message_count, current_count,
                )
                self._reset_session(session_id)
                state = self._get_state(session_id)

        # Ingest new messages as turns
        new_messages = messages[state.last_message_count:]
        for msg in new_messages:
            turn = self._message_to_turn(msg, state)
            self._conversation.add_turn(turn, session_id)

        state.last_message_count = current_count

        # Run compaction
        compaction_result = self._conversation.run_compaction(session_id)
        if compaction_result and compaction_result.turns_compacted > 0:
            logger.info(
                "Session %s: compacted %d turns (%d -> %d tokens)",
                session_id,
                compaction_result.turns_compacted,
                compaction_result.original_tokens,
                compaction_result.compacted_tokens,
            )

        # Run summarization (async)
        summarization_result = await self._conversation.run_summarization(session_id)
        if summarization_result:
            logger.info(
                "Session %s: summarized turns %s (%d -> %d tokens)",
                session_id,
                summarization_result.turn_range,
                summarization_result.original_tokens,
                summarization_result.summary_tokens,
            )

        # Rebuild message list from zones
        zones = self._conversation.zones(session_id)
        optimized: list[dict] = []

        # Inject summaries as a synthetic user/assistant exchange at the start
        system_injection: str | None = None
        if zones.summarized:
            summary_text = "\n\n".join(zones.summarized)
            system_injection = (
                f"[Previous conversation summary]\n{summary_text}\n"
                f"[End of summary — recent conversation follows]"
            )

        # Add compacted turns
        for turn in zones.compacted:
            optimized.append(self._turn_to_message(turn))

        # Add raw turns
        for turn in zones.raw:
            optimized.append(self._turn_to_message(turn))

        # If we have no optimized messages, fall through to original
        if not optimized:
            return None, messages

        return system_injection, optimized

    async def post_process(self, session_id: str, assistant_text: str) -> None:
        """Post-process after a response stream completes.

        Adds the assistant response as a turn so it's tracked for future
        compaction/summarization.
        """
        state = self._get_state(session_id)
        turn = ConversationTurn(
            turn_number=state.turn_counter,
            role="assistant",
            content=assistant_text,
        )
        state.turn_counter += 1
        # Don't add to conversation manager — Claude Code will send the full
        # history including this response on the next request. We just increment
        # last_message_count so we know to expect it.
        # Actually, the external caller sends full history, so we track via
        # last_message_count. The assistant response will come back as part of
        # the next request's messages array.

    def _message_to_turn(self, msg: dict, state: SessionState) -> ConversationTurn:
        """Convert an API message dict to a ConversationTurn."""
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle Anthropic content blocks
        if isinstance(content, list):
            content_type = self._detect_content_type(content)
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        text_parts.append(
                            f"[tool_use: {block.get('name', '?')}({_truncate(str(block.get('input', '')), 200)})]"
                        )
                    elif block.get("type") == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = " ".join(
                                b.get("text", "") for b in result_content if isinstance(b, dict)
                            )
                        text_parts.append(f"[tool_result]\n{result_content}")
                    else:
                        text_parts.append(str(block))
                else:
                    text_parts.append(str(block))
            content_str = "\n".join(text_parts)
        else:
            content_str = str(content)
            content_type = None

        turn = ConversationTurn(
            turn_number=state.turn_counter,
            role=role,
            content=content_str,
            content_type=content_type,
        )
        state.turn_counter += 1
        return turn

    def _detect_content_type(self, content_blocks: list) -> str | None:
        """Detect the dominant content type from Anthropic content blocks."""
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                return "tool_output"
            if block_type == "tool_use":
                return "tool_output"
        return None

    def _turn_to_message(self, turn: ConversationTurn) -> dict:
        """Convert a ConversationTurn back to an API message dict.

        For compacted turns, we emit plain text content. For raw turns that
        still have their original structure, we'd ideally preserve it — but
        since we've already flattened to text in _message_to_turn, we
        reconstruct as simple text messages.
        """
        return {"role": turn.role, "content": turn.content}

    def _reset_session(self, session_id: str) -> None:
        """Reset all state for a session."""
        self._conversation.destroy_session(session_id)
        self._session_states.pop(session_id, None)

    def destroy_session(self, session_id: str) -> None:
        """Public interface to clean up a session."""
        self._reset_session(session_id)

    def get_metrics(self) -> dict:
        """Return bridge engine metrics."""
        metrics = {}
        for sid, state in self._session_states.items():
            zones = self._conversation.zones(sid)
            metrics[sid] = {
                "message_count": state.last_message_count,
                "turn_counter": state.turn_counter,
                "summarized_count": len(zones.summarized),
                "compacted_count": len(zones.compacted),
                "raw_count": len(zones.raw),
                "total_tokens": zones.total_tokens,
                "zone_transitions": self._conversation.get_zone_transitions(sid),
            }
        return metrics


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
