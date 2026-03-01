"""Tests for A2A client tool."""

import pytest

from sr2.a2a.client import A2AClientTool, A2AToolConfig


def _make_config(**kwargs) -> A2AToolConfig:
    defaults = {
        "name": "remote_agent",
        "target_url": "http://remote:8008",
        "description": "Call remote agent",
    }
    defaults.update(kwargs)
    return A2AToolConfig(**defaults)


class TestA2AClientTool:
    def test_tool_definition_has_correct_schema(self):
        """tool_definition has correct schema."""
        tool = A2AClientTool(config=_make_config())
        defn = tool.tool_definition

        assert defn["name"] == "remote_agent"
        assert defn["description"] == "Call remote agent"
        assert "message" in defn["parameters"]["properties"]
        assert "message" in defn["parameters"]["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """execute() with successful response -> returns result string."""

        async def mock_http(url, payload, timeout):
            return {"status": "completed", "result": "Done!"}

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        result = await tool.execute("Do something")

        assert result == "Done!"

    @pytest.mark.asyncio
    async def test_execute_failed_response(self):
        """execute() with failed response -> returns error message with status."""

        async def mock_http(url, payload, timeout):
            return {"status": "failed", "result": "Bad request"}

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        result = await tool.execute("Do something")

        assert "failed" in result
        assert "Bad request" in result

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """execute() with timeout -> returns timeout message."""

        async def mock_http(url, payload, timeout):
            raise TimeoutError("Connection timed out")

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        result = await tool.execute("Do something")

        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        """execute() with exception -> returns error message (no crash)."""

        async def mock_http(url, payload, timeout):
            raise ConnectionError("Connection refused")

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        result = await tool.execute("Do something")

        assert "A2A call failed" in result
        assert "Connection refused" in result

    @pytest.mark.asyncio
    async def test_execute_without_http_callable(self):
        """execute() without http_callable -> returns error message."""
        tool = A2AClientTool(config=_make_config())
        result = await tool.execute("Do something")

        assert "No HTTP client configured" in result

    @pytest.mark.asyncio
    async def test_payload_includes_task_id_and_message(self):
        """Payload includes task_id and message."""
        captured = {}

        async def mock_http(url, payload, timeout):
            captured.update(payload)
            return {"status": "completed", "result": "ok"}

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        await tool.execute("Hello", task_id="my-task")

        assert captured["task_id"] == "my-task"
        assert captured["message"] == "Hello"

    @pytest.mark.asyncio
    async def test_payload_includes_skill_id(self):
        """Payload includes skill_id when configured."""
        captured = {}

        async def mock_http(url, payload, timeout):
            captured.update(payload)
            return {"status": "completed", "result": "ok"}

        config = _make_config(skill_id="summarize")
        tool = A2AClientTool(config=config, http_callable=mock_http)
        await tool.execute("Summarize this")

        assert captured["skill_id"] == "summarize"

    @pytest.mark.asyncio
    async def test_fetch_agent_card_success(self):
        """fetch_agent_card() returns card dict on success."""

        async def mock_http(url, payload, timeout):
            return {"name": "remote", "version": "1.0"}

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        card = await tool.fetch_agent_card()

        assert card is not None
        assert card["name"] == "remote"

    @pytest.mark.asyncio
    async def test_fetch_agent_card_failure(self):
        """fetch_agent_card() returns None on failure."""

        async def mock_http(url, payload, timeout):
            raise ConnectionError("Cannot connect")

        tool = A2AClientTool(config=_make_config(), http_callable=mock_http)
        card = await tool.fetch_agent_card()

        assert card is None
