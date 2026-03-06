"""Tests for MCP client integration."""

import asyncio
import time
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
    session.list_resources = AsyncMock(side_effect=Exception("Not supported"))
    session.list_prompts = AsyncMock(side_effect=Exception("Not supported"))
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


def _make_transport_cm(session=None):
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
    """MCPToolHandler.execute() connects on demand and calls tool."""

    @pytest.mark.asyncio
    async def test_execute_text_content(self):
        mgr = MCPManager()
        text_block = MagicMock(type="text", text="Search result: 3 emails found")
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(
            content=[text_block],
            isError=False,
        ))
        # Pre-populate session so _get_session returns it
        mgr._sessions["test_server"] = mock_session
        mgr._last_activity["test_server"] = time.monotonic()

        handler = MCPToolHandler(mgr, "test_server", "search_emails")
        result = await handler.execute(query="meeting", limit=10)

        mock_session.call_tool.assert_called_once_with(
            "search_emails", arguments={"query": "meeting", "limit": 10},
        )
        assert result == "Search result: 3 emails found"

    @pytest.mark.asyncio
    async def test_execute_multiple_blocks(self):
        mgr = MCPManager()
        blocks = [
            MagicMock(type="text", text="Line 1"),
            MagicMock(type="text", text="Line 2"),
        ]
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=blocks, isError=False))
        mgr._sessions["srv"] = mock_session
        mgr._last_activity["srv"] = time.monotonic()

        handler = MCPToolHandler(mgr, "srv", "read_email")
        result = await handler.execute(id="123")

        assert result == "Line 1\nLine 2"

    @pytest.mark.asyncio
    async def test_execute_image_block(self):
        mgr = MCPManager()
        block = MagicMock(type="image", mimeType="image/png")
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[block], isError=False))
        mgr._sessions["srv"] = mock_session
        mgr._last_activity["srv"] = time.monotonic()

        handler = MCPToolHandler(mgr, "srv", "screenshot")
        result = await handler.execute()

        assert "[image: image/png]" in result

    @pytest.mark.asyncio
    async def test_execute_empty_content(self):
        mgr = MCPManager()
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[], isError=False))
        mgr._sessions["srv"] = mock_session
        mgr._last_activity["srv"] = time.monotonic()

        handler = MCPToolHandler(mgr, "srv", "empty_tool")
        result = await handler.execute()

        assert result == "Tool returned no content."

    @pytest.mark.asyncio
    async def test_execute_raises_on_isError(self):
        mgr = MCPManager()
        text_block = MagicMock(type="text", text="Error: something went wrong")
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=MagicMock(
            content=[text_block],
            isError=True,
        ))
        mgr._sessions["srv"] = mock_session
        mgr._last_activity["srv"] = time.monotonic()

        handler = MCPToolHandler(mgr, "srv", "failing_tool")
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
        assert mgr._configs_by_name["test"] is config

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
    async def test_discover_all_discovers_and_registers_tools(self):
        """discover_all connects, discovers schemas, registers handlers, disconnects."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test_server", url="echo hello"))

        mock_tool = _make_mock_tool(
            "my_tool", "A test tool",
            {"type": "object", "properties": {"x": {"type": "string"}}},
        )
        mock_session = _make_mock_session(tools=[mock_tool])
        mock_cm = _make_transport_cm()
        executor = ToolExecutor()

        with (
            patch("runtime.mcp.client._MCP_AVAILABLE", True),
            patch("runtime.mcp.client.StdioServerParameters", create=True),
            patch("runtime.mcp.client.stdio_client", create=True, return_value=mock_cm),
            patch("runtime.mcp.client.ClientSession", create=True, return_value=mock_session),
        ):
            result = await mgr.discover_all(executor)

        assert "test_server" in result
        assert "my_tool" in result["test_server"]
        assert executor.has("my_tool")
        assert mgr.get_tool_schemas()[0]["name"] == "my_tool"

        # After discovery, no live connections should remain
        assert len(mgr._sessions) == 0
        assert len(mgr._server_tasks) == 0

    @pytest.mark.asyncio
    async def test_connect_all_is_alias_for_discover_all(self):
        """connect_all() is a backward-compat alias."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="test_server", url="echo hello"))

        mock_tool = _make_mock_tool("my_tool", "A test tool")
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

        assert "my_tool" in result["test_server"]

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
            result = await mgr.discover_all(executor)

        assert "allowed_tool" in result["filtered"]
        assert "blocked_tool" not in result["filtered"]
        assert executor.has("allowed_tool")
        assert not executor.has("blocked_tool")

    @pytest.mark.asyncio
    async def test_failed_server_doesnt_crash_discover_all(self):
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
            result = await mgr.discover_all(executor)

        assert result["bad_server"] == []

    @pytest.mark.asyncio
    async def test_schemas_persist_after_discovery(self):
        """Tool schemas remain available after discovery disconnects."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="srv", url="echo hello"))

        mock_tool = _make_mock_tool("persistent_tool", "Persists")
        mock_session = _make_mock_session(tools=[mock_tool])
        mock_cm = _make_transport_cm()
        executor = ToolExecutor()

        with (
            patch("runtime.mcp.client._MCP_AVAILABLE", True),
            patch("runtime.mcp.client.StdioServerParameters", create=True),
            patch("runtime.mcp.client.stdio_client", create=True, return_value=mock_cm),
            patch("runtime.mcp.client.ClientSession", create=True, return_value=mock_session),
        ):
            await mgr.discover_all(executor)

        # No live sessions
        assert len(mgr._sessions) == 0

        # But schemas are still there
        schemas = mgr.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "persistent_tool"

        # And tool-server mapping exists
        assert mgr._tool_server_map["persistent_tool"] == "srv"

    @pytest.mark.asyncio
    async def test_tool_handler_registered_with_manager_ref(self):
        """Registered MCPToolHandler holds manager reference, not session."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="srv", url="echo hello"))

        mock_tool = _make_mock_tool("my_tool", "Test")
        mock_session = _make_mock_session(tools=[mock_tool])
        mock_cm = _make_transport_cm()
        executor = ToolExecutor()

        with (
            patch("runtime.mcp.client._MCP_AVAILABLE", True),
            patch("runtime.mcp.client.StdioServerParameters", create=True),
            patch("runtime.mcp.client.stdio_client", create=True, return_value=mock_cm),
            patch("runtime.mcp.client.ClientSession", create=True, return_value=mock_session),
        ):
            await mgr.discover_all(executor)

        handler = executor._handlers["my_tool"]
        assert isinstance(handler, MCPToolHandler)
        assert handler._manager is mgr
        assert handler._server_name == "srv"
        assert handler._tool_name == "my_tool"

    @pytest.mark.asyncio
    async def test_disconnect_all_cleans_up(self):
        """disconnect_all() cleans up all on-demand connections."""
        mgr = MCPManager()

        # Simulate a running server task
        stop_event = asyncio.Event()
        mgr._stop_events["test"] = stop_event
        mgr._sessions["test"] = MagicMock()
        mgr._last_activity["test"] = time.monotonic()

        async def fake_server():
            await stop_event.wait()

        task = asyncio.create_task(fake_server())
        mgr._server_tasks["test"] = task

        await mgr.disconnect_all()

        assert task.done()
        assert len(mgr._sessions) == 0
        assert len(mgr._server_tasks) == 0
        assert len(mgr._stop_events) == 0

    @pytest.mark.asyncio
    async def test_get_session_reuses_existing(self):
        """_get_session returns existing session without reconnecting."""
        mgr = MCPManager()
        mgr.add_server(MCPServerConfig(name="srv", url="echo hello"))

        mock_session = MagicMock()
        mgr._sessions["srv"] = mock_session
        mgr._last_activity["srv"] = time.monotonic()

        result = await mgr._get_session("srv")
        assert result is mock_session

    @pytest.mark.asyncio
    async def test_get_session_unknown_server_raises(self):
        """_get_session raises KeyError for unknown server."""
        mgr = MCPManager()

        with pytest.raises(KeyError, match="No MCP server config"):
            await mgr._get_session("nonexistent")

    @pytest.mark.asyncio
    async def test_idle_timeout_disconnects(self):
        """Connection is torn down after idle timeout."""
        mgr = MCPManager()
        mgr._idle_timeout = 0.1  # 100ms for test speed

        stop_event = asyncio.Event()
        mgr._stop_events["srv"] = stop_event
        mgr._sessions["srv"] = MagicMock()
        mgr._last_activity["srv"] = time.monotonic() - 10  # already very idle

        async def fake_server():
            await stop_event.wait()

        task = asyncio.create_task(fake_server())
        mgr._server_tasks["srv"] = task

        mgr._start_idle_watcher("srv")

        # Wait for idle watcher to fire (checks at timeout/2 = 50ms, then disconnects)
        for _ in range(20):
            await asyncio.sleep(0.05)
            if "srv" not in mgr._sessions:
                break

        assert "srv" not in mgr._sessions
        assert task.done()

    @pytest.mark.asyncio
    async def test_make_transport_stdio(self):
        """_make_transport creates stdio transport."""
        config = MCPServerConfig(name="test", url="npx server", args=["--flag"])

        with (
            patch("runtime.mcp.client.StdioServerParameters", create=True) as mock_params,
            patch("runtime.mcp.client.stdio_client", create=True) as mock_stdio,
        ):
            MCPManager._make_transport(config)
            mock_params.assert_called_once_with(
                command="npx",
                args=["server", "--flag"],
                env=None,
            )
            mock_stdio.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_transport_http(self):
        """_make_transport creates HTTP transport."""
        config = MCPServerConfig(
            name="test", url="http://localhost:8080",
            transport="http", headers={"Auth": "Bearer x"},
        )

        with patch("runtime.mcp.client.streamablehttp_client", create=True) as mock_http:
            MCPManager._make_transport(config)
            mock_http.assert_called_once_with(
                "http://localhost:8080", headers={"Auth": "Bearer x"},
            )

    def test_make_transport_unknown_raises(self):
        """_make_transport raises ValueError for unknown transport."""
        config = MCPServerConfig(name="test", url="test")
        # Force an invalid transport value
        object.__setattr__(config, "transport", "grpc")

        with pytest.raises(ValueError, match="Unknown MCP transport"):
            MCPManager._make_transport(config)
