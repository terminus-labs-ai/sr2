"""Tests for A2A integration wiring (FastAPI routes + client registry)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from sr2.a2a.app import A2AClientRegistry, create_a2a_routes
from sr2.a2a.card import AgentCardGenerator
from sr2.a2a.client import A2AClientTool, A2AToolConfig
from sr2.a2a.server import A2AServerAdapter
from sr2.pipeline.engine import CompiledContext, PipelineEngine
from sr2.pipeline.result import PipelineResult
from sr2.pipeline.router import InterfaceRouter


def _make_app():
    """Create a FastAPI app with a mocked server adapter."""
    result = PipelineResult()
    result._overall_status = "success"

    compiled = CompiledContext(
        content="Pipeline result",
        tokens=15,
        pipeline_result=result,
    )

    engine = AsyncMock(spec=PipelineEngine)
    engine.compile.return_value = compiled

    router = MagicMock(spec=InterfaceRouter)
    router.route.return_value = MagicMock()

    card_gen = AgentCardGenerator(agent_name="test-agent", description="Test agent")

    adapter = A2AServerAdapter(
        pipeline_engine=engine,
        interface_router=router,
        card_generator=card_gen,
    )

    return create_a2a_routes(adapter)


class TestA2ARoutes:
    @pytest.mark.asyncio
    async def test_get_agent_card(self):
        """GET /.well-known/agent.json returns agent card."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/.well-known/agent.json")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_post_message_valid(self):
        """POST /a2a/message with valid payload -> completed response."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/a2a/message",
                content=json.dumps({"task_id": "t1", "message": "Hello"}),
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"] == "Pipeline result"

    @pytest.mark.asyncio
    async def test_post_message_invalid_json(self):
        """POST /a2a/message with invalid JSON -> error response."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/a2a/message",
                content="not json {{{",
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert "Invalid JSON" in data["result"]

    @pytest.mark.asyncio
    async def test_health_check(self):
        """GET /health returns ok status."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["agent"] == "test-agent"


class TestA2AClientRegistry:
    def test_register_and_get(self):
        """A2AClientRegistry register/get/list works."""
        registry = A2AClientRegistry()
        config = A2AToolConfig(name="agent_a", target_url="http://a:8008")
        client = A2AClientTool(config=config)

        registry.register("agent_a", client)

        assert registry.get("agent_a") is client
        assert "agent_a" in registry.registered_clients

    def test_get_unknown_raises(self):
        """A2AClientRegistry.get() unknown name raises KeyError."""
        registry = A2AClientRegistry()

        with pytest.raises(KeyError, match="No A2A client registered"):
            registry.get("nonexistent")

    def test_get_all_tool_definitions(self):
        """get_all_tool_definitions() returns schemas for all clients."""
        registry = A2AClientRegistry()

        for name in ["agent_a", "agent_b"]:
            config = A2AToolConfig(name=name, target_url=f"http://{name}:8008")
            registry.register(name, A2AClientTool(config=config))

        defs = registry.get_all_tool_definitions()
        assert len(defs) == 2
        names = {d["name"] for d in defs}
        assert names == {"agent_a", "agent_b"}
