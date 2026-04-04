"""A2A protocol interface plugin."""

import logging

from sr2_runtime.plugins.base import TriggerContext

logger = logging.getLogger(__name__)


class A2APlugin:
    """A2A protocol interface plugin.

    Receives A2A messages and routes them through the pipeline.

    Config:
        plugin: a2a
        session:
          name: "a2a_{task_id}"
          lifecycle: ephemeral
        pipeline: interfaces/a2a_inbound.yaml
    """

    def __init__(self, interface_name: str, config: dict, agent_callback):
        self._name = interface_name
        self._config = config
        self._callback = agent_callback
        self._session_config = config.get("session", {})

    async def start(self) -> None:
        logger.info(f"A2A plugin '{self._name}' registered")

    async def stop(self) -> None:
        pass

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """A2A doesn't proactively send. Responses go through the A2A protocol."""
        pass

    async def handle_a2a_request(
        self, task_id: str, message: str, metadata: dict | None = None
    ) -> str:
        """Handle an inbound A2A request. Called by the A2A server adapter."""
        session_name = self._session_config.get("name", f"a2a_{task_id}")
        if "{task_id}" in session_name:
            session_name = session_name.replace("{task_id}", task_id)

        lifecycle = self._session_config.get("lifecycle", "ephemeral")

        trigger = TriggerContext(
            interface_name=self._name,
            plugin_name="a2a",
            session_name=session_name,
            session_lifecycle=lifecycle,
            input_data=message,
            metadata={"task_id": task_id, **(metadata or {})},
        )
        return await self._callback(trigger)
