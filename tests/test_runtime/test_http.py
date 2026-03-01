"""Tests for the generic HTTP app (HOTFIX-04)."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from runtime.http import create_http_app


def _mock_agent(name="TestAgent"):
    """Create a mock Agent with the attributes the HTTP app needs."""
    agent = MagicMock()
    agent._name = name
    agent._sr2 = MagicMock()
    agent._sr2.export_metrics = MagicMock(return_value="# HELP test\n")
    agent._plugins = {"telegram": MagicMock(), "email_check": MagicMock()}
    agent._mcp_manager = MagicMock()
    agent._mcp_manager._sessions = {"gmail": MagicMock()}
    agent.handle_user_message = AsyncMock(return_value="Hello!")
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
    """POST /chat returns response from agent."""
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
    agent.handle_user_message.assert_called_once_with(message="Hi", session_id="s1")


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
