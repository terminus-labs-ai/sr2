"""Generic HTTP endpoints for any agent."""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from runtime.agent import Agent

_STATIC_DIR = Path(__file__).parent / "static"


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

    @app.post("/chat")
    async def chat(request: Request):
        body = await request.json()
        response = await agent.handle_user_message(
            message=body.get("message", ""),
            session_id=body.get("session_id", "http_default"),
        )
        return {"response": response}

    # -- OpenAI-compatible endpoints (for Open WebUI, etc.) --

    @app.get("/v1/models")
    async def openai_models():
        model_id = f"sr2-{agent._name.lower()}"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "sr2",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        body = await request.json()
        messages = body.get("messages", [])

        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle content array format
                    user_message = " ".join(
                        p.get("text", "")
                        for p in content
                        if p.get("type") == "text"
                    )
                else:
                    user_message = content
                break

        # Use model field as a session namespace so different
        # "model" selections in Open WebUI get separate sessions
        session_id = f"openai_{body.get('model', 'default')}"

        response = await agent.handle_user_message(
            message=user_message,
            session_id=session_id,
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", f"sr2-{agent._name.lower()}"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # -- Standard endpoints --

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

    @app.get("/.well-known/agent.json")
    async def agent_card():
        card = agent._generate_agent_card()
        return JSONResponse(content=card)

    @app.get("/")
    async def chat_ui():
        html = (_STATIC_DIR / "chat.html").read_text()
        return HTMLResponse(content=html)

    return app
