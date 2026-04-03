"""A2A integration wiring — FastAPI routes and client registry."""

import json

from sr2.a2a.client import A2AClientTool
from sr2.a2a.server import A2AServerAdapter


def create_a2a_routes(server_adapter: A2AServerAdapter):
    """Create a FastAPI sub-application with A2A routes.

    Routes:
    - GET /.well-known/agent.json -> Agent Card
    - POST /a2a/message -> Handle A2A message
    - GET /health -> Health check

    This can be mounted onto a larger FastAPI app:
        main_app.mount("/", create_a2a_routes(adapter))

    Requires: pip install sr2[runtime]
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError(
            "FastAPI is required for A2A routes. Install with: pip install sr2[runtime]"
        )
    app = FastAPI(title="SR2 A2A Agent")

    @app.get("/.well-known/agent.json")
    async def agent_card():
        """Return the Agent Card for A2A discovery."""
        return JSONResponse(content=server_adapter.get_agent_card())

    @app.post("/a2a/message")
    async def handle_message(request: Request):
        """Handle an inbound A2A message."""
        body = await request.body()
        response_json = await server_adapter.handle_raw_json(body.decode("utf-8"))
        return JSONResponse(content=json.loads(response_json))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "agent": server_adapter.get_agent_card().get("name", "unknown"),
        }

    return app


class A2AClientRegistry:
    """Registry of A2A client tools for calling remote agents."""

    def __init__(self):
        self._clients: dict[str, A2AClientTool] = {}

    def register(self, name: str, client: A2AClientTool) -> None:
        """Register an A2A client tool."""
        self._clients[name] = client

    def get(self, name: str) -> A2AClientTool:
        """Get a client by name. Raises KeyError if not found."""
        if name not in self._clients:
            raise KeyError(f"No A2A client registered: {name}")
        return self._clients[name]

    def get_all_tool_definitions(self) -> list[dict]:
        """Get tool definitions for all registered A2A clients.

        These should be added to the agent's tool set.
        """
        return [c.tool_definition for c in self._clients.values()]

    @property
    def registered_clients(self) -> list[str]:
        return list(self._clients.keys())
