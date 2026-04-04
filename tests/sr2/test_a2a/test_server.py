"""Tests for A2A server adapter."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2.a2a.card import AgentCardGenerator
from sr2.a2a.server import A2ARequest, A2AServerAdapter
from sr2.pipeline.engine import CompiledContext, PipelineEngine
from sr2.pipeline.result import PipelineResult
from sr2.pipeline.router import InterfaceRouter


def _make_adapter(
    compiled_content: str = "Hello from pipeline",
    compiled_tokens: int = 10,
    pipeline_status: str = "success",
    compile_side_effect: Exception | None = None,
) -> A2AServerAdapter:
    """Create a server adapter with mocked dependencies."""
    result = PipelineResult()
    result._overall_status = pipeline_status

    compiled = CompiledContext(
        content=compiled_content,
        tokens=compiled_tokens,
        pipeline_result=result,
    )

    engine = AsyncMock(spec=PipelineEngine)
    if compile_side_effect:
        engine.compile.side_effect = compile_side_effect
    else:
        engine.compile.return_value = compiled

    router = MagicMock(spec=InterfaceRouter)
    router.route.return_value = MagicMock()

    card_gen = AgentCardGenerator(agent_name="test-agent", description="Test")

    return A2AServerAdapter(
        pipeline_engine=engine,
        interface_router=router,
        card_generator=card_gen,
    )


class TestA2AServerAdapter:

    @pytest.mark.asyncio
    async def test_handle_message_completed(self):
        """handle_message() with valid request -> completed response."""
        adapter = _make_adapter()
        request = A2ARequest(task_id="task-1", message="Hello")
        response = await adapter.handle_message(request)

        assert response.status == "completed"
        assert response.result == "Hello from pipeline"
        assert response.task_id == "task-1"

    @pytest.mark.asyncio
    async def test_handle_message_includes_tokens(self):
        """handle_message() includes pipeline tokens in metadata."""
        adapter = _make_adapter(compiled_tokens=42)
        request = A2ARequest(task_id="task-2", message="Hi")
        response = await adapter.handle_message(request)

        assert response.metadata is not None
        assert response.metadata["tokens"] == 42

    @pytest.mark.asyncio
    async def test_handle_message_pipeline_failure(self):
        """handle_message() with pipeline failure -> failed response."""
        adapter = _make_adapter(compile_side_effect=RuntimeError("Pipeline broke"))
        request = A2ARequest(task_id="task-3", message="Fail")
        response = await adapter.handle_message(request)

        assert response.status == "failed"
        assert "Pipeline broke" in response.result

    def test_get_agent_card(self):
        """get_agent_card() returns valid card."""
        adapter = _make_adapter()
        card = adapter.get_agent_card()

        assert card["name"] == "test-agent"
        assert "capabilities" in card

    @pytest.mark.asyncio
    async def test_handle_raw_json_valid(self):
        """handle_raw_json() parses JSON and calls handle_message."""
        adapter = _make_adapter()
        raw = json.dumps({"task_id": "task-4", "message": "Raw hello"})
        result_json = await adapter.handle_raw_json(raw)
        result = json.loads(result_json)

        assert result["status"] == "completed"
        assert result["task_id"] == "task-4"

    @pytest.mark.asyncio
    async def test_handle_raw_json_invalid(self):
        """handle_raw_json() with invalid JSON -> error response."""
        adapter = _make_adapter()
        result_json = await adapter.handle_raw_json("not json at all {{{")
        result = json.loads(result_json)

        assert result["status"] == "failed"
        assert "Invalid JSON" in result["result"]

    @pytest.mark.asyncio
    async def test_missing_message_field(self):
        """Missing message field -> handles gracefully (empty string)."""
        adapter = _make_adapter()
        raw = json.dumps({"task_id": "task-5"})
        result_json = await adapter.handle_raw_json(raw)
        result = json.loads(result_json)

        assert result["status"] == "completed"
        assert result["task_id"] == "task-5"
