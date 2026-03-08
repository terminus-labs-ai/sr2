"""Heartbeat tools — schedule and cancel future agent callbacks."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Callable

from runtime.heartbeat.model import Heartbeat
from runtime.heartbeat.store import HeartbeatStore

logger = logging.getLogger(__name__)


class ScheduleHeartbeatTool:
    """Tool that lets the agent schedule a future callback to itself.

    When executed, captures the last N turns from the current session
    and creates a heartbeat that will fire after the specified delay.
    """

    def __init__(
        self,
        store: HeartbeatStore,
        agent_name: str,
        max_context_turns: int,
        session_resolver: Callable[[], str | None],
        session_turns_resolver: Callable[[], list[dict]],
    ) -> None:
        self._store = store
        self._agent_name = agent_name
        self._max_context_turns = max_context_turns
        self._session_resolver = session_resolver
        self._session_turns_resolver = session_turns_resolver

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "schedule_heartbeat",
            "description": (
                "Schedule a future callback to yourself. "
                "When it fires, you'll get a new session with context from this conversation "
                "and the prompt you specify. Use this to check back on async operations, "
                "retry failed tasks, or schedule follow-ups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "delay_seconds": {
                        "type": "integer",
                        "description": "How many seconds from now to fire the heartbeat.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The prompt you'll receive when the heartbeat fires. "
                            "Be specific about what action to take."
                        ),
                    },
                    "key": {
                        "type": "string",
                        "description": (
                            "Optional unique key for idempotency. "
                            "If a heartbeat with this key already exists, it will be updated. "
                            "Use this to prevent duplicate heartbeats for the same task."
                        ),
                    },
                },
                "required": ["delay_seconds", "prompt"],
            },
        }

    async def execute(
        self,
        delay_seconds: int = 0,
        prompt: str = "",
        key: str | None = None,
        **kwargs,
    ) -> str:
        if delay_seconds <= 0:
            return "Error: delay_seconds must be positive."
        if not prompt.strip():
            return "Error: prompt cannot be empty."

        session_id = self._session_resolver() or "unknown"
        turns = self._session_turns_resolver()
        context_turns = turns[-self._max_context_turns :] if self._max_context_turns > 0 else []

        fire_at = datetime.now(UTC) + timedelta(seconds=delay_seconds)

        heartbeat = Heartbeat(
            id=str(uuid.uuid4()),
            agent_name=self._agent_name,
            source_session=session_id,
            prompt=prompt,
            fire_at=fire_at,
            context_turns=context_turns,
            key=key,
        )

        await self._store.upsert(heartbeat)

        key_info = f" (key={key})" if key else ""
        logger.info(f"Heartbeat scheduled{key_info}: fires at {fire_at.isoformat()}")
        return (
            f"Heartbeat scheduled{key_info}. "
            f"Will fire in {delay_seconds}s at {fire_at.isoformat()}."
        )


class CancelHeartbeatTool:
    """Tool that lets the agent cancel a previously scheduled heartbeat by key."""

    def __init__(self, store: HeartbeatStore) -> None:
        self._store = store

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "cancel_heartbeat",
            "description": (
                "Cancel a previously scheduled heartbeat by its key. "
                "Only cancels pending heartbeats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The key of the heartbeat to cancel.",
                    },
                },
                "required": ["key"],
            },
        }

    async def execute(self, key: str = "", **kwargs) -> str:
        if not key.strip():
            return "Error: key is required."

        cancelled = await self._store.cancel_by_key(key)
        if cancelled:
            logger.info(f"Heartbeat cancelled: key={key}")
            return f"Heartbeat with key '{key}' cancelled."
        return f"No pending heartbeat found with key '{key}'."
