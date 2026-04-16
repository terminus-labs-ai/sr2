"""BridgeEngine — format-agnostic context optimization for proxied requests.

Delegates all compaction/summarization/memory/scope/degradation to SR2's
unified orchestrator via ``proxy_optimize()`` and ``proxy_post_process()``.
Bridge keeps only what's bridge-specific: session locking, message diffing,
adapter coordination, and session persistence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

from sr2.config.models import PipelineConfig
from sr2.pipeline.conversation import ConversationManager
from sr2.sr2 import SR2

from sr2_bridge.adapters.base import BridgeAdapter
from sr2_bridge.config import BridgeConfig
from sr2_bridge.persistence import BridgeSessionStore
from sr2_bridge.session_tracker import BridgeSession

logger = logging.getLogger(__name__)


class BridgeEngine:
    """Applies SR2 context optimization to proxied LLM requests.

    On each request (Claude Code sends full history every time), the engine:
    1. Compares incoming message count to last known count
    2. Delegates wire-format conversion to the adapter
    3. Delegates compaction/summarization/memory/scope to SR2.proxy_optimize()
    4. Asks the adapter to rebuild the message array from zones
    5. Post-process: delegates memory extraction to SR2.proxy_post_process()
    """

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        bridge_config: BridgeConfig | None = None,
        sr2: SR2 | None = None,
        key_cache=None,
    ):
        self._config = pipeline_config
        self._bridge_config = bridge_config or BridgeConfig()

        # Create SR2 instance if not provided (e.g. tests, simple usage)
        if sr2 is None:
            from sr2.sr2 import SR2Config

            sr2 = SR2(SR2Config(
                config_dir="",
                agent_yaml={},
                preloaded_config=pipeline_config,
            ))
        self._sr2 = sr2

        # Per-session lock to serialize concurrent requests from Claude Code.
        self._session_locks: dict[str, asyncio.Lock] = {}

        # Session persistence (optional)
        self._session_store: BridgeSessionStore | None = None
        if self._bridge_config.session.persistence:
            self._session_store = BridgeSessionStore(db_path=self._bridge_config.memory.db_path)

        # Metrics tracking
        self._postprocess_error_count: int = 0
        self._last_summarization_duration: float | None = None
        self._last_request_tokens: dict[str, tuple[int, int]] = {}

    async def optimize(
        self,
        system: str | None,
        messages: list[dict],
        session: BridgeSession,
        adapter: BridgeAdapter,
        agent_name: str | None = None,
    ) -> tuple[str | None, list[dict]]:
        """Optimize messages using SR2 pipeline.

        Args:
            agent_name: Optional per-request agent identity (from X-SR2-Agent-Name
                header). When set, private memory scopes use this identity.

        Returns (system_injection_or_None, optimized_messages).
        """
        session_id = session.session_id

        # Serialize concurrent requests for the same session
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()

        async with self._session_locks[session_id]:
            return await self._optimize_locked(
                system, messages, session, adapter, agent_name=agent_name,
            )

    async def _optimize_locked(
        self,
        system: str | None,
        messages: list[dict],
        session: BridgeSession,
        adapter: BridgeAdapter,
        agent_name: str | None = None,
    ) -> tuple[str | None, list[dict]]:
        """Actual optimization logic, called under per-session lock."""
        session_id = session.session_id
        current_count = len(messages)
        current_hash = hashlib.md5(
            json.dumps(messages, sort_keys=True, default=str).encode()
        ).hexdigest()

        if (
            current_count == session.last_message_count
            and current_hash == session.last_message_hash
        ):
            # Parallel request with identical content -- skip optimization
            logger.debug(
                "Session %s: same message count (%d) and hash, passing through",
                session_id,
                current_count,
            )
            return None, messages

        if (
            current_count == session.last_message_count
            and current_hash != session.last_message_hash
        ):
            # Same count but different content — messages were edited.
            logger.info(
                "Session %s: message content changed (count %d, hash %s -> %s), resetting",
                session_id,
                current_count,
                session.last_message_hash,
                current_hash,
            )
            self._reset_session(session)

        if current_count < session.last_message_count:
            logger.info(
                "Session %s: message count decreased (%d -> %d), resetting",
                session_id,
                session.last_message_count,
                current_count,
            )
            self._reset_session(session)

        # Convert new messages to turns via adapter (engine never sees wire format)
        new_messages = messages[session.last_message_count :]
        new_turns = adapter.messages_to_turns(new_messages, session.turn_counter)
        session.turn_counter += len(new_turns)
        session.last_message_count = current_count
        session.last_message_hash = current_hash

        # Extract retrieval query from latest user message
        retrieval_query = self._extract_retrieval_query(messages)

        # Delegate to SR2
        t0 = time.monotonic()
        result = await self._sr2.proxy_optimize(
            new_turns=new_turns,
            session_id=session_id,
            system_prompt=system,
            retrieval_query=retrieval_query,
            agent_name=agent_name,
        )
        elapsed = time.monotonic() - t0

        # Track summarization duration if it happened
        if result.summarization_result:
            self._last_summarization_duration = elapsed

        # Convert turns back to wire format via adapter
        zones = result.zones
        all_turns = list(zones.compacted) + list(zones.raw)
        optimized = adapter.turns_to_messages(all_turns, messages)

        # If we have no optimized messages, fall through to original
        if not optimized:
            return None, messages

        # Track before/after token counts for metrics
        tokens_before = sum(len(str(m.get("content", ""))) // 4 for m in messages)
        tokens_after = sum(len(str(m.get("content", ""))) // 4 for m in optimized)
        if result.system_injection:
            tokens_after += len(result.system_injection) // 4
        self._last_request_tokens[session_id] = (tokens_before, tokens_after)

        # Token budget warning (advisory only)
        max_ctx = self._bridge_config.forwarding.max_context_tokens
        if max_ctx and tokens_after > max_ctx:
            logger.warning(
                "Session %s: optimized request (%d est. tokens) exceeds "
                "max_context_tokens (%d) by %d tokens",
                session_id,
                tokens_after,
                max_ctx,
                tokens_after - max_ctx,
            )

        # Persist session state after optimization
        await self._persist_session(session)

        return result.system_injection, optimized

    async def _persist_session(self, session: BridgeSession) -> None:
        """Persist session state to SQLite if persistence is enabled."""
        if not self._session_store:
            return
        try:
            zones = self._sr2._conversation.zones(session.session_id)
            await self._session_store.save_session(session, zones)
        except Exception:
            logger.warning(
                "Session %s: persistence failed",
                session.session_id,
                exc_info=True,
            )

    async def post_process(
        self,
        session: BridgeSession,
        assistant_text: str,
        agent_name: str | None = None,
    ) -> None:
        """Post-process after a response stream completes.

        Delegates to SR2.proxy_post_process() for memory extraction,
        conflict detection, and compaction.

        Args:
            agent_name: Optional per-request agent identity for memory extraction.
        """
        turn_number = session.turn_counter
        session.turn_counter += 1

        try:
            await self._sr2.proxy_post_process(
                assistant_text=assistant_text,
                session_id=session.session_id,
                turn_number=turn_number,
                agent_name=agent_name,
            )
        except Exception:
            logger.warning(
                "Session %s: post-processing failed",
                session.session_id,
                exc_info=True,
            )
            self._postprocess_error_count += 1

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
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    if texts:
                        return " ".join(texts)[:500]
        return None

    def _reset_session(self, session: BridgeSession) -> None:
        """Reset conversation state for a session."""
        self._sr2.reset_session(session.session_id)
        session.last_message_count = 0
        session.last_message_hash = ""
        session.turn_counter = 0
        session.turns = []

    def destroy_session(self, session_id: str) -> None:
        """Clean up ConversationManager state and session lock for a session."""
        self._sr2.reset_session(session_id)
        self._session_locks.pop(session_id, None)
        if self._session_store:
            # Fire-and-forget delete — best effort
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._session_store.delete_session(session_id))
            except RuntimeError:
                pass  # No event loop — skip persistence cleanup

    async def shutdown(self) -> None:
        """Clean up resources (close DB connections)."""
        if self._session_store:
            await self._session_store.disconnect()

    def get_session_metrics(self, session: BridgeSession) -> dict:
        """Return metrics for a specific session."""
        zones = self._sr2._conversation.zones(session.session_id)
        return {
            "message_count": session.last_message_count,
            "turn_counter": session.turn_counter,
            "summarized_count": len(zones.summarized),
            "compacted_count": len(zones.compacted),
            "raw_count": len(zones.raw),
            "total_tokens": zones.total_tokens,
            "zone_transitions": self._sr2._conversation.get_zone_transitions(
                session.session_id
            ),
        }

    @property
    def postprocess_error_count(self) -> int:
        return self._postprocess_error_count

    @property
    def last_summarization_duration(self) -> float | None:
        return self._last_summarization_duration

    @property
    def last_request_tokens(self) -> dict[str, tuple[int, int]]:
        return self._last_request_tokens

    def is_breaker_open(self, feature: str) -> bool:
        """Check if circuit breaker is open for a feature."""
        return self._sr2._engine._circuit_breaker.is_open(feature)

    @property
    def session_store(self) -> BridgeSessionStore | None:
        return self._session_store

    @property
    def conversation_manager(self) -> ConversationManager:
        return self._sr2._conversation

    @property
    def degradation_level(self) -> str:
        """Degradation level based on circuit breaker state.

        Returns 'full' when all breakers are closed, 'compaction_only' when
        summarization/memory are open, 'passthrough' when all are open.
        """
        breaker = self._sr2._engine._circuit_breaker
        summarization_open = breaker.is_open("summarization")
        memory_open = breaker.is_open("memory_extraction")
        if summarization_open and memory_open:
            return "passthrough"
        if summarization_open or memory_open:
            return "compaction_only"
        return "full"

    @property
    def circuit_breaker_status(self) -> dict:
        return self._sr2._engine._circuit_breaker.status()
