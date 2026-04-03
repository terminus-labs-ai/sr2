"""Generic HTTP endpoints for any agent."""

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from sr2_runtime.agent import Agent

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _find_http_plugin(agent: Agent):
    """Find the HTTP plugin instance from agent._plugins.

    Returns the plugin if found, otherwise None.
    """
    for plugin in agent._plugins.values():
        from sr2_runtime.plugins.http import HTTPPlugin

        if isinstance(plugin, HTTPPlugin):
            return plugin
    return None


def _create_default_http_plugin(agent: Agent):
    """Create a default HTTPPlugin for backwards compatibility.

    Used when no explicit HTTP interface is configured in the agent YAML
    but the agent is started with --http.
    """
    from sr2_runtime.plugins.http import HTTPPlugin

    return HTTPPlugin(
        interface_name="http",
        config={"agent_name": agent._name.lower()},
        agent_callback=agent._handle_trigger,
    )


def create_http_app(agent: Agent) -> FastAPI:
    """Create a FastAPI app with standard agent endpoints.

    Endpoints:
      POST /chat                    - send a message, get a response
      GET  /health                  - health check
      GET  /metrics                 - Prometheus metrics
      GET  /                        - web chat UI
      GET  /v1/models               - OpenAI-compatible model list
      POST /v1/chat/completions     - OpenAI-compatible chat endpoint
      GET  /.well-known/agent.json  - A2A Agent Card
    """
    app = FastAPI(title=f"{agent._name} Agent")

    # --- Plugin-provided routes (chat, OpenAI-compat) ---
    http_plugin = _find_http_plugin(agent)
    if http_plugin is None:
        logger.info("No HTTP plugin configured; creating default for backwards compatibility")
        http_plugin = _create_default_http_plugin(agent)

    # Ensure the plugin knows the agent name for model IDs
    http_plugin._agent_name = agent._name.lower()

    routes = http_plugin.get_routes()

    app.post("/chat")(routes["chat"])
    app.post("/v1/chat/completions")(routes["openai_chat"])
    app.get("/v1/models")(routes["openai_models"])

    # --- Infrastructure endpoints (agent-level, not interface-specific) ---

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "agent": agent._name,
            "plugins": list(agent._plugins.keys()),
            "mcp_servers": list(agent._mcp_manager._sessions.keys()),
        }

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            content=agent._sr2.export_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/a2a/message")
    async def a2a_message(request: Request):
        body = await request.json()
        task_id = body.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        message = body.get("message", "")
        metadata = body.get("metadata")

        a2a_plugin = next(
            (p for p in agent._plugins.values() if hasattr(p, "handle_a2a_request")),
            None,
        )
        if a2a_plugin is None:
            return JSONResponse(
                {"task_id": task_id, "status": "failed", "result": "No A2A interface configured"},
                status_code=404,
            )

        try:
            result = await a2a_plugin.handle_a2a_request(task_id, message, metadata)
        except Exception as e:
            logger.error(f"A2A message handling failed for task {task_id}: {e}", exc_info=True)
            return JSONResponse(
                {"task_id": task_id, "status": "failed", "result": f"Internal error: {e}"},
                status_code=500,
            )
        return JSONResponse({"task_id": task_id, "status": "completed", "result": result})

    @app.get("/.well-known/agent.json")
    async def agent_card():
        card = agent._generate_agent_card()
        return JSONResponse(content=card)

    @app.get("/")
    async def chat_ui():
        html = (_STATIC_DIR / "chat.html").read_text()
        return HTMLResponse(content=html)

    return app
