"""Tests for MCP client integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runtime.mcp import MCPManager, MCPToolHandler
from runtime.config import MCPServerConfig
from runtime.tool_executor import ToolExecutor


def _make_mock_session(tools=None):
    """Create a mock ClientSession with initialize/list_tools."""
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=MagicMock(tools=tools or []))
    # Support async with session:
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def _make_mock_tool(name, description="", input_schema=None):
    """Create a mock MCP tool."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {"type": "object", "properties": {}}
    return tool


def _make_transport_cm():
    """Create a mock transport context manager that yields (read, write)."""
    cm = MagicMock()
    read_stream = MagicMock()
    write_stream = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(read_stream, write_stream))
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestMCPServerConfig:
    """MCPServerConfig with minimal fields creates valid config."""

    def test_minimal_config(self):
        config = MCPServerConfig(name="gmail", url="npx @anthropic/mcp-server-gmail")
        assert config.name == "gmail"
        assert config.url == "npx @anthropic/mcp-server-gmail"
        assert config.transport == "stdio"
        assert config.tools is None
        assert config.env is None
        assert config.args is None

    def test_full_config(self):
        config = MCPServerConfig(
            name="linear",
            url="npx @anthropic/mcp-server-linear",
            transport="stdio",
            tools=["list_issues", "get_issue"],
            env={"LINEAR_API_KEY": "test"},
            args=["--verbose"],
        )
        assert config.tools == ["list_issues", "get_issue"]
        assert config.env == {"LINEAR_API_KEY": "test"}
        assert config.args == ["--verbose"]


class TestMCPToolHandler:
    """MCPToolHandler.execute() calls session.call_tool() with correct args."""

    @pytest.mark.asyncio
    async def test_execute_text_content(self):
        mock_session = MagicMock()
        text_block = MagicMock(type="text", text="Search result: 3 emails found")
        mock_session.call_tool = AsyncMock(return_value=MagicMock(
            content=[text_block],
            isError=False,
        ))

        handler = MCPToolHandler(mock_session, "search_emails")
        result = await handler.execute(query="meeting", limit=10)

        mock_session.call_tool.assert_called_once_with(
            "search_emails", arguments={"query": "meeting", "limit": 10},
        )
        assert result == "Search result: 3 emails found"

    @pytest.mark.asyncio
    async def test_execute_multiple_blocks(self):
        mock_session = MagicMock()
        blocks = [
            MagicMock(type="text", text="Line 1"),
            MagicMock(type="text", text="Line 2"),
        ]
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=blocks, isError=False))

        handler = MCPToolHandler(mock_session, "read_email")
        result = await handler.execute(id="123")

        assert result == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_execute_image_block(self):
        mock_session = MagicMock()
        block = MagicMock(type="image", mimeType="image/png")
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[block], isError=False))

        handler = MCPToolHandler(mock_session, "screenshot")
        result = await handler.execute()

        assert "[image: image/png]" in result

    @pytest.mark.asyncio
    async def test_execute_empty_content(self):
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))

        handler = MCPToolHandler(mock_session, "empty_tool")
        result = await handler.execute()

        assert result == "Tool returned no content."

    @pytest.mark.asyncio
    async def test_execute_raises_on_isError(self):
        mock_session = MagicMock()
        text_block = MagicMock(type="text", text="Error: something went wrong")
        mock_session.call_tool = AsyncMock(return_value=MagicMock(
            content=[text_block],
            isError=True,
        ))

        handler = MCPToolHandler(mock_session, "failing_tool")
        with pytest.raises(RuntimeError, match="Error: something went wrong"):
            await handler.execute(foo="bar")


class TestMCPManager:
    """Tests for the MCPManager class."""

    def test_add_server_stores_config(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="test", url="echo test")
        mgr.add_server(config)

        assert len(mgr._configs) == 1
        assert mgr._configs[0].name == "test"

    def test_get_tool_schemas_empty(self):
        mgr = MCPManager()
        assert mgr.get_tool_schemas() == []

    def test_get_tool_schemas_returns_discovered(self):
        mgr = MCPManager()
        mgr._discovered_tools = {
            "search_emails": {
                "name": "search_emails",
                "description": "Search emails",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }

        schemas = mgr.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "search_emails"

    @pytest.mark.asyncio
    async def test_connect_server_discovers_and_registers_tools(self):
        """connect_all spawns a task, discovers tools, and registers them."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test_server", url="echo hello"))

        mock_tool = _make_mock_tool("my_tool", "A test tool", {"type": "object", "properties": {"x": {"type": "string"}}})
        mock_session = _make_mock_session(tools=[mock_tool])
        mock_cm = _make_transport_cm()
        executor = ToolExecutor()

        with (
            patch("runtime.mcp.client._MCP_AVAILABLE", True),
            patch("runtime.mcp.client.StdioServerParameters", create=True),
            patch("runtime.mcp.client.stdio_client", create=True, return_value=mock_cm),
            patch("runtime.mcp.client.ClientSession", create=True, return_value=mock_session),
        ):
            result = await mgr.connect_all(executor)

        assert "test_server" in result
        assert "my_tool" in result["test_server"]
        assert executor.has("my_tool")
        assert mgr.get_tool_schemas()[0]["name"] == "my_tool"

        # Clean up background task
        await mgr.disconnect_all()

    @pytest.mark.asyncio
    async def test_curated_tools_filters_discovered(self):
        """When tools list is specified, only those tools are registered."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(
            name="filtered", url="echo hello", tools=["allowed_tool"],
        ))

        allowed = _make_mock_tool("allowed_tool", "OK")
        blocked = _make_mock_tool("blocked_tool", "Nope")
        mock_session = _make_mock_session(tools=[allowed, blocked])
        mock_cm = _make_transport_cm()
        executor = ToolExecutor()

        with (
            patch("runtime.mcp.client._MCP_AVAILABLE", True),
            patch("runtime.mcp.client.StdioServerParameters", create=True),
            patch("runtime.mcp.client.stdio_client", create=True, return_value=mock_cm),
            patch("runtime.mcp.client.ClientSession", create=True, return_value=mock_session),
        ):
            result = await mgr.connect_all(executor)

        assert "allowed_tool" in result["filtered"]
        assert "blocked_tool" not in result["filtered"]
        assert executor.has("allowed_tool")
        assert not executor.has("blocked_tool")

        await mgr.disconnect_all()

    @pytest.mark.asyncio
    async def test_failed_server_doesnt_crash_connect_all(self):
        """Failed server connection logs error and continues."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="bad_server", url="nonexistent_command"))

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=ConnectionError("Server not found"))
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        executor = ToolExecutor()

        with (
            patch("runtime.mcp.client._MCP_AVAILABLE", True),
            patch("runtime.mcp.client.StdioServerParameters", create=True),
            patch("runtime.mcp.client.stdio_client", create=True, return_value=mock_cm),
        ):
            result = await mgr.connect_all(executor)

        assert result["bad_server"] == []

    @pytest.mark.asyncio
    async def test_disconnect_all_cleans_up(self):
        """disconnect_all() signals tasks and waits for them."""
        mgr = MCPManager()

        # Simulate a running server task
        stop_event = asyncio.Event()
        mgr._stop_events["test"] = stop_event
        mgr._sessions["test"] = MagicMock()

        async def fake_server():
            await stop_event.wait()

        task = asyncio.create_task(fake_server())
        mgr._server_tasks["test"] = task

        await mgr.disconnect_all()

        assert task.done()
        assert len(mgr._sessions) == 0
        assert len(mgr._server_tasks) == 0
        assert len(mgr._stop_events) == 0
