"""Single-shot interface plugin — fires once with a message, returns response."""

import logging

from runtime.plugins.base import TriggerContext

logger = logging.getLogger(__name__)


class SingleShotPlugin:
    """Single-shot interface plugin.

    Designed for CLI/scripting use: receives a message, runs it through
    the agent pipeline once, returns the response text. No polling, no
    listening — the trigger is called explicitly from the CLI.

    Config (from YAML):
        plugin: single_shot
        session:
          name: task_runner
          lifecycle: ephemeral
        pipeline: interfaces/task_runner.yaml
    """

    def __init__(self, interface_name: str, config: dict, agent_callback):
        self._name = interface_name
        self._config = config
        self._callback = agent_callback
        self._session_config = config.get("session", {})

    async def start(self) -> None:
        """No-op — single-shot trigger is called explicitly."""
        pass

    async def stop(self) -> None:
        """No-op."""
        pass

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """Single-shot doesn't proactively send. No-op."""
        pass

    async def run(self, message: str) -> str:
        """Fire the trigger once with the given message and return the response."""
        session_name = self._session_config.get("name", self._name)
        lifecycle = self._session_config.get("lifecycle", "ephemeral")

        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="single_shot",
            session_name=session_name,
            session_lifecycle=lifecycle,
            input_data=message,
        )

        logger.info(f"Single-shot '{self._name}' firing")
        response = await self._callback(trigger)
        logger.info(
            f"Single-shot '{self._name}' complete: {len(response)} chars"
        )
        return response
