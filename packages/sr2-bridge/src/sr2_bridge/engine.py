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

        # Cache last optimized result per session for duplicate request handling
        self._last_optimized: dict[str, tuple[str | None, list[dict]]] = {}

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

    @staticmethod
    def _normalize_for_hash(msg: dict) -> str:
        """Normalize a message for hashing.

        Strips volatile fields that Claude Code changes between turns:
        - cache_control markers (added/moved between requests)
        - <system-reminder> blocks (content may include timestamps or
          change slightly between turns; tracked separately by the engine)
        """
        import re

        normalized = dict(msg)
        content = normalized.get("content", "")

        if isinstance(content, list):
            clean_blocks = []
            for block in content:
                if isinstance(block, dict):
                    b = {k: v for k, v in block.items() if k != "cache_control"}
                    if b.get("type") == "text" and "text" in b:
                        b["text"] = re.sub(
                            r"<system-reminder>.*?</system-reminder>",
                            "",
                            b["text"],
                            flags=re.DOTALL,
                        ).strip()
                    if b.get("type") == "text" and not b.get("text"):
                        continue
                    clean_blocks.append(b)
                else:
                    clean_blocks.append(block)
            normalized["content"] = clean_blocks
        elif isinstance(content, str):
            normalized["content"] = re.sub(
                r"<system-reminder>.*?</system-reminder>",
                "",
                content,
                flags=re.DOTALL,
            ).strip()

        return json.dumps(normalized, sort_keys=True, default=str)

    @staticmethod
    def _hash_message(msg: dict) -> str:
        """Compute MD5 hash for a normalized message."""
        return hashlib.md5(
            BridgeEngine._normalize_for_hash(msg).encode()
        ).hexdigest()

    def _detect_history_change(
        self,
        incoming_hashes: list[str],
        session: BridgeSession,
    ) -> str:
        """Compare per-message hashes to detect what changed.

        Returns one of:
        - "first_request": no stored hashes yet
        - "normal": prior messages match, new messages appended
        - "compaction": prior message hashes differ (Claude Code compacted)
        - "reset": message count decreased significantly (user did /clear)
        """
        stored = session.message_hashes

        if not stored:
            return "first_request"

        incoming_count = len(incoming_hashes)
        stored_count = len(stored)

        # Significant decrease → reset (user did /clear)
        if incoming_count < stored_count:
            return "reset"

        # Compare all stored hashes against their positions in incoming
        for i in range(stored_count):
            if i >= incoming_count or incoming_hashes[i] != stored[i]:
                return "compaction"

        return "normal"

    async def _optimize_locked(
        self,
        system: str | None,
        messages: list[dict],
        session: BridgeSession,
        adapter: BridgeAdapter,
        agent_name: str | None = None,
    ) -> tuple[str | None, list[dict]]:
        """Actual optimization logic, called under per-session lock.

        Uses per-message hash comparison to detect Claude Code compaction.
        On normal turns: processes only new messages.
        On compaction: preserves ConversationManager history, processes only
        the new user message from the compacted request.
        """
        session_id = session.session_id

        # Hash each incoming message individually
        incoming_hashes = [self._hash_message(m) for m in messages]
        change_type = self._detect_history_change(incoming_hashes, session)

        # Duplicate request detection (all hashes match including last).
        # Return the last optimized result if available, otherwise forward as-is.
        if change_type == "normal" and len(incoming_hashes) == len(session.message_hashes):
            cached = self._last_optimized.get(session_id)
            if cached:
                logger.info(
                    "DECISION: Session %s: DUPLICATE — all %d message hashes match, "
                    "returning cached optimized result.",
                    session_id,
                    len(incoming_hashes),
                )
                return cached
            logger.info(
                "DECISION: Session %s: DUPLICATE — all %d message hashes match, "
                "no cached result, forwarding original.",
                session_id,
                len(incoming_hashes),
            )
            return None, messages

        if change_type == "reset":
            logger.info(
                "DECISION: Session %s: RESET — message count decreased (%d -> %d).",
                session_id,
                len(session.message_hashes),
                len(messages),
            )
            self._reset_session(session)
            # Process all messages as fresh
            new_messages = messages
        elif change_type == "compaction":
            logger.info(
                "DECISION: Session %s: COMPACTION DETECTED — prior message hashes differ. "
                "Preserving ConversationManager history, extracting only new message.",
                session_id,
            )
            # Don't reset ConversationManager — it has clean history.
            # Extract only the LAST message (new user input).
            new_messages = messages[-1:] if messages else []
        else:
            # "first_request" or "normal" — process new messages
            new_messages = messages[len(session.message_hashes):]

        # Convert new messages to turns
        new_turns = adapter.messages_to_turns(new_messages, session.turn_counter)
        session.turn_counter += len(new_turns)

        # Update stored hashes
        if change_type == "reset":
            session.message_hashes = list(incoming_hashes)
        elif change_type == "compaction":
            # Replace stored hashes with incoming (compacted) hashes
            session.message_hashes = list(incoming_hashes)
        else:
            # Normal/first: extend with new hashes
            session.message_hashes = list(incoming_hashes)

        # Backward compat: keep count/hash in sync
        session.last_message_count = len(messages)
        session.last_message_hash = hashlib.md5(
            json.dumps(messages, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Collect and deduplicate system-reminder blocks from extracted turns
        has_new_reminders = False
        total_extracted = 0
        total_deduped = 0
        for turn in new_turns:
            if turn.metadata and "extracted_system_reminders" in turn.metadata:
                blocks = turn.metadata.pop("extracted_system_reminders")
                total_extracted += len(blocks)
                for block in blocks:
                    block_hash = hashlib.sha256(block.encode()).hexdigest()
                    if block_hash not in session.system_reminder_hashes:
                        session.system_reminder_hashes.add(block_hash)
                        session.system_reminder_content.append(block)
                        has_new_reminders = True
                    else:
                        total_deduped += 1
        if total_extracted:
            logger.info(
                "DECISION: Session %s: REMINDER EXTRACTION — found %d block(s), "
                "%d new, %d deduplicated, %d total unique in session.",
                session_id,
                total_extracted,
                total_extracted - total_deduped,
                total_deduped,
                len(session.system_reminder_hashes),
            )

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

        # Check if compaction/summarization happened
        has_compaction = result.compaction_result and result.compaction_result.turns_compacted > 0
        has_summarization = result.summarization_result is not None

        # Deferred injection: only inject stored reminders when compaction or
        # summarization has modified the conversation. Pre-compaction, reminders
        # live naturally in user messages and the system prompt stays stable.
        if (has_compaction or has_summarization) and session.system_reminder_content:
            reminder_text = "\n\n".join(session.system_reminder_content)
            if result.system_injection:
                result.system_injection = f"{result.system_injection}\n\n{reminder_text}"
            else:
                result.system_injection = reminder_text
            logger.info(
                "DECISION: Session %s: DEFERRED INJECTION — compaction/summarization triggered, "
                "injecting %d system-reminder block(s) into system_injection.",
                session_id,
                len(session.system_reminder_content),
            )

        has_injection = result.system_injection is not None

        logger.info(
            "DECISION: Session %s: OPTIMIZE CHECK — compaction=%s, summarization=%s, "
            "injection=%s (reminders=%d). Will %s.",
            session_id,
            has_compaction,
            has_summarization,
            has_injection,
            len(session.system_reminder_content),
            "REBUILD messages" if (has_compaction or has_summarization or has_injection) else "PASSTHROUGH original",
        )

        if not has_compaction and not has_summarization and not has_injection:
            tokens_est = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            self._last_request_tokens[session_id] = (tokens_est, tokens_est)
            await self._persist_session(session)
            self._last_optimized[session_id] = (None, messages)
            return None, messages

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

        self._last_optimized[session_id] = (result.system_injection, optimized)
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
        session.message_hashes = []

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
