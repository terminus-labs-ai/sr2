"""Heartbeat scanner — polls for due heartbeats and fires them."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine

from runtime.heartbeat.model import Heartbeat, HeartbeatStatus
from runtime.heartbeat.store import HeartbeatStore
from runtime.plugins.base import TriggerContext

logger = logging.getLogger(__name__)


class HeartbeatScanner:
    """Async polling loop that fires due heartbeats.

    Follows the same lifecycle pattern as TimerPlugin:
    start() creates an asyncio task, stop() cancels it.
    """

    def __init__(
        self,
        store: HeartbeatStore,
        agent_callback: Callable[[TriggerContext], Coroutine[Any, Any, str]],
        poll_interval_seconds: int = 30,
        respond_fn: Callable[[str, str, str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._store = store
        self._callback = agent_callback
        self._poll_interval = poll_interval_seconds
        self._respond_fn = respond_fn
        self._running = False
        self._busy = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="heartbeat_scanner")
        logger.info(f"Heartbeat scanner started (poll every {self._poll_interval}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Heartbeat scanner stopped")

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._poll_interval)

            if self._busy:
                logger.debug("Heartbeat scanner skipped — previous scan still running")
                continue

            self._busy = True
            try:
                now = datetime.now(UTC)
                due = await self._store.list_due(now)

                if due:
                    logger.info(f"Heartbeat scanner found {len(due)} due heartbeat(s)")

                for hb in due:
                    if not self._running:
                        break
                    await self._fire(hb)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Heartbeat scanner error: {e}")
            finally:
                self._busy = False

    async def _fire(self, hb: Heartbeat) -> None:
        """Fire a single heartbeat."""
        logger.info(f"Firing heartbeat {hb.id}" + (f" (key={hb.key})" if hb.key else ""))
        await self._store.update_status(hb.id, HeartbeatStatus.firing)

        try:
            trigger = TriggerContext(
                interface_name="heartbeat",
                plugin_name="heartbeat",
                session_name=f"heartbeat_{hb.id}",
                session_lifecycle="ephemeral",
                input_data=hb.prompt,
                metadata={
                    "heartbeat_id": hb.id,
                    "heartbeat_key": hb.key,
                    "source_session": hb.source_session,
                    "context_turns": hb.context_turns,
                },
            )

            response = await self._callback(trigger)
            await self._store.update_status(hb.id, HeartbeatStatus.completed)
            logger.info(f"Heartbeat {hb.id} completed")

            # Deliver response to source interface/session
            if response and self._respond_fn and hb.source_interface:
                try:
                    await self._respond_fn(hb.source_interface, hb.source_session, response)
                except Exception as e:
                    logger.warning(f"Heartbeat {hb.id} delivery failed: {e}")

        except Exception as e:
            logger.error(f"Heartbeat {hb.id} failed: {e}")
            await self._store.update_status(hb.id, HeartbeatStatus.failed)
