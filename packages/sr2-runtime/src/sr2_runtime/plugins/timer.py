"""Timer/heartbeat interface plugin."""

import asyncio
import logging
from datetime import UTC, datetime

from sr2_runtime.plugins.base import TriggerContext

logger = logging.getLogger(__name__)


class TimerPlugin:
    """Timer/heartbeat interface plugin.

    Fires a trigger at regular intervals. Replaces the old HeartbeatScheduler.

    Config (from YAML):
        plugin: timer
        interval_seconds: 300
        session:
          name: heartbeat_email
          lifecycle: ephemeral
        pipeline: interfaces/heartbeat_email.yaml
    """

    def __init__(self, interface_name: str, config: dict, agent_callback):
        self._name = interface_name
        self._config = config
        self._callback = agent_callback
        self._interval = config.get("interval_seconds", 300)
        self._session_config = config.get("session", {})
        self._enabled = config.get("enabled", True)
        self._running = False
        self._busy = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if not self._enabled:
            logger.info(f"Timer '{self._name}' is disabled")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name=f"timer_{self._name}")
        logger.info(f"Timer '{self._name}' started (every {self._interval}s)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Timer '{self._name}' stopped")

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """Timers don't proactively send. No-op."""
        pass

    async def _loop(self) -> None:
        """Main polling loop."""
        while self._running:
            if self._busy:
                logger.info(f"Timer '{self._name}' skipped — previous tick still running")
                await asyncio.sleep(self._interval)
                continue

            logger.info(f"Timer '{self._name}' ticked")
            self._busy = True
            try:
                if not self._running:
                    break

                session_name = self._session_config.get("name", self._name)
                lifecycle = self._session_config.get("lifecycle", "ephemeral")

                trigger = TriggerContext(
                    interface_name=self._name,
                    plugin_name="timer",
                    session_name=session_name,
                    session_lifecycle=lifecycle,
                    input_data="",
                    metadata={"tick_time": datetime.now(UTC).isoformat()},
                )

                logger.debug(f"Timer triggering call on session {session_name}")
                response = await self._callback(trigger)
                logger.debug(
                    f"Timer '{self._name}' tick processed: {response[:100] if response else 'empty'}"
                )

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Timer '{self._name}' error: {e}")
            finally:
                self._busy = False
                await asyncio.sleep(self._interval)
