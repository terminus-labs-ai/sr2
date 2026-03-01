"""HTTP API interface plugin."""

import logging

from runtime.plugins.base import TriggerContext

logger = logging.getLogger(__name__)


class HTTPPlugin:
    """HTTP API interface plugin.

    Provides FastAPI routes. Started by the Agent when creating the HTTP app.
    Unlike other plugins, this one doesn't run its own event loop — it
    registers routes on the shared FastAPI app.

    Config:
        plugin: http
        port: 8008
        session:
          name: "{request.session_id}"
          lifecycle: persistent
        pipeline: interfaces/user_message.yaml
    """

    def __init__(self, interface_name: str, config: dict, agent_callback):
        self._name = interface_name
        self._config = config
        self._callback = agent_callback
        self._session_config = config.get("session", {})
        self._port = config.get("port", 8008)

    async def start(self) -> None:
        """HTTP plugin doesn't self-start. Routes are registered in create_app()."""
        logger.info(f"HTTP plugin '{self._name}' registered (port {self._port})")

    async def stop(self) -> None:
        pass

    async def send(self, session_name: str, content: str, metadata: dict | None = None) -> None:
        """HTTP can't proactively send. No-op."""
        pass

    def get_routes(self):
        """Return FastAPI route handlers for the Agent to mount."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        async def chat(request: Request):
            body = await request.json()
            message = body.get("message", "")
            session_id = body.get("session_id", "http_default")

            session_name = self._session_config.get("name", session_id)
            if "{request.session_id}" in session_name:
                session_name = session_name.replace("{request.session_id}", session_id)

            lifecycle = self._session_config.get("lifecycle", "persistent")

            trigger = TriggerContext(
                interface_name=self._name,
                plugin_name="http",
                session_name=session_name,
                session_lifecycle=lifecycle,
                input_data=message,
                metadata={"session_id": session_id},
            )
            response = await self._callback(trigger)
            return JSONResponse({"response": response})

        return {"chat": chat}
