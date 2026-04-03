"""Tests for the generic HTTP app."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sr2_runtime.http import create_http_app


def _mock_agent(name="TestAgent"):
    """Create a mock Agent with the attributes the HTTP app needs."""
    agent = MagicMock()
    agent._name = name
    agent._sr2 = MagicMock()
    agent._sr2.export_metrics = MagicMock(return_value="# HELP test\n")
    agent._plugins = {"telegram": MagicMock(), "email_check": MagicMock()}
    agent._mcp_manager = MagicMock()
    agent._mcp_manager._sessions = {"gmail": MagicMock()}
    agent._handle_trigger = AsyncMock(return_value="Hello!")
    agent._generate_agent_card = MagicMock(return_value={
        "name": name,
        "version": "0.1.0",
    })
    return agent


def test_create_http_app_returns_fastapi():
    """create_http_app() returns a FastAPI app."""
    agent = _mock_agent()
    app = create_http_app(agent)
    assert isinstance(app, FastAPI)


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health returns agent name and status."""
    agent = _mock_agent("EDI")
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["agent"] == "EDI"
    assert "email_check" in data["plugins"]
    assert "gmail" in data["mcp_servers"]


@pytest.mark.asyncio
async def test_chat_endpoint():
    """POST /chat returns response from agent via plugin pipeline."""
    agent = _mock_agent()
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/chat", json={"message": "Hi", "session_id": "s1"},
        )

    assert resp.status_code == 200
    assert resp.json()["response"] == "Hello!"

    # Verify it went through _handle_trigger, not handle_user_message
    agent._handle_trigger.assert_called_once()
    trigger = agent._handle_trigger.call_args[0][0]
    assert trigger.plugin_name == "http"
    assert trigger.input_data == "Hi"
    assert trigger.metadata["session_id"] == "s1"


@pytest.mark.asyncio
async def test_chat_endpoint_default_session():
    """POST /chat without session_id uses 'http_default'."""
    agent = _mock_agent()
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/chat", json={"message": "hello"})

    assert resp.status_code == 200
    trigger = agent._handle_trigger.call_args[0][0]
    assert trigger.session_name == "http_default"
    assert trigger.session_lifecycle == "persistent"


@pytest.mark.asyncio
async def test_openai_chat_endpoint():
    """POST /v1/chat/completions returns OpenAI-compatible response."""
    agent = _mock_agent()
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "sr2-testagent",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"
    assert data["choices"][0]["finish_reason"] == "stop"

    trigger = agent._handle_trigger.call_args[0][0]
    assert trigger.input_data == "Hi"


@pytest.mark.asyncio
async def test_openai_chat_content_array():
    """POST /v1/chat/completions handles content array format."""
    agent = _mock_agent()
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ],
                    }
                ],
            },
        )

    assert resp.status_code == 200
    trigger = agent._handle_trigger.call_args[0][0]
    assert trigger.input_data == "Hello World"


@pytest.mark.asyncio
async def test_openai_models_endpoint():
    """GET /v1/models returns model list."""
    agent = _mock_agent("MyAgent")
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "sr2-myagent"
    assert data["data"][0]["owned_by"] == "sr2"


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """GET /metrics returns Prometheus text format."""
    agent = _mock_agent()
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_uses_configured_http_plugin():
    """When an HTTPPlugin is in agent._plugins, it is used instead of default."""
    from sr2_runtime.plugins.http import HTTPPlugin

    agent = _mock_agent()
    plugin = HTTPPlugin(
        interface_name="api",
        config={"session": {"lifecycle": "ephemeral"}, "agent_name": "testagent"},
        agent_callback=AsyncMock(return_value="Plugin response!"),
    )
    agent._plugins["api"] = plugin
    app = create_http_app(agent)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/chat", json={"message": "test"})

    assert resp.status_code == 200
    assert resp.json()["response"] == "Plugin response!"
    # Verify it used the plugin's callback, not agent._handle_trigger
    plugin._callback.assert_called_once()
    trigger = plugin._callback.call_args[0][0]
    assert trigger.interface_name == "api"
    assert trigger.session_lifecycle == "ephemeral"
