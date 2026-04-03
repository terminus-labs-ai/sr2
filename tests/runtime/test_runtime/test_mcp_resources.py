"""Tests for MCP resource discovery, listing, and reading."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2_runtime.config import MCPServerConfig
from sr2_runtime.mcp.client import MCPListResourcesHandler, MCPManager, MCPResourceHandler


def _make_mock_resource(uri, name="", description="", mime_type=None):
    """Create a mock MCP resource."""
    r = MagicMock()
    r.uri = uri
    r.name = name
    r.description = description
    r.mimeType = mime_type
    return r


class TestMCPManagerResources:
    """Tests for resource discovery and access on MCPManager."""

    @pytest.mark.asyncio
    async def test_discover_resources_stores_info(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="fs", url="test")

        resources = [
            _make_mock_resource("file:///tmp/a.txt", "a.txt", "A file"),
            _make_mock_resource("file:///tmp/b.txt", "b.txt"),
        ]
        session = MagicMock()
        session.list_resources = AsyncMock(return_value=MagicMock(resources=resources))

        await mgr._discover_resources(config, session)

        assert len(mgr._discovered_resources["fs"]) == 2
        assert mgr._discovered_resources["fs"][0]["uri"] == "file:///tmp/a.txt"
        assert mgr._resource_server_map["file:///tmp/a.txt"] == "fs"

    @pytest.mark.asyncio
    async def test_discover_resources_handles_no_support(self):
        mgr = MCPManager()
        config = MCPServerConfig(name="no_res", url="test")
        session = MagicMock()
        session.list_resources = AsyncMock(side_effect=Exception("Not supported"))

        await mgr._discover_resources(config, session)

        assert mgr._discovered_resources.get("no_res") is None

    @pytest.mark.asyncio
    async def test_list_resources_all(self):
        mgr = MCPManager()
        mgr._discovered_resources = {
            "s1": [{"uri": "a", "name": "a"}],
            "s2": [{"uri": "b", "name": "b"}],
        }
        result = await mgr.list_resources()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_resources_filtered(self):
        mgr = MCPManager()
        mgr._discovered_resources = {
            "s1": [{"uri": "a", "name": "a"}],
            "s2": [{"uri": "b", "name": "b"}],
        }
        result = await mgr.list_resources(server_name="s1")
        assert len(result) == 1
        assert result[0]["uri"] == "a"

    @pytest.mark.asyncio
    async def test_read_resource_by_uri(self):
        mgr = MCPManager()
        mgr._resource_server_map["file:///test.txt"] = "fs"

        text_content = MagicMock()
        text_content.text = "Hello, world!"
        session = MagicMock()
        session.read_resource = AsyncMock(return_value=MagicMock(contents=[text_content]))
        # _get_session will find the session directly
        mgr._sessions["fs"] = session
        mgr._last_activity["fs"] = 0

        result = await mgr.read_resource("file:///test.txt")
        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_read_resource_explicit_server(self):
        mgr = MCPManager()

        text_content = MagicMock()
        text_content.text = "Content from explicit server"
        session = MagicMock()
        session.read_resource = AsyncMock(return_value=MagicMock(contents=[text_content]))
        mgr._sessions["my_server"] = session
        mgr._last_activity["my_server"] = 0

        result = await mgr.read_resource("some://uri", server_name="my_server")
        assert result == "Content from explicit server"

    @pytest.mark.asyncio
    async def test_read_resource_unknown_uri_raises(self):
        mgr = MCPManager()

        with pytest.raises(KeyError, match="No MCP server could read resource"):
            await mgr.read_resource("unknown://nothing")

    @pytest.mark.asyncio
    async def test_read_resource_disconnected_server_raises(self):
        mgr = MCPManager()
        mgr._resource_server_map["file:///x"] = "gone"
        # Server config exists but no session and no config to reconnect
        # _get_session will raise KeyError because config is missing
        with pytest.raises(KeyError, match="No MCP server config"):
            await mgr.read_resource("file:///x")

    def test_get_resource_tool_schemas(self):
        mgr = MCPManager()
        schemas = mgr.get_resource_tool_schemas()
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"mcp_list_resources", "mcp_read_resource"}


class TestMCPResourceHandlers:
    """Tests for MCPResourceHandler and MCPListResourcesHandler."""

    @pytest.mark.asyncio
    async def test_resource_handler_calls_read(self):
        mgr = MCPManager()
        mgr.read_resource = AsyncMock(return_value="file content")

        handler = MCPResourceHandler(mgr)
        result = await handler.execute(uri="file:///test.txt", server="fs")

        mgr.read_resource.assert_called_once_with("file:///test.txt", server_name="fs")
        assert result == "file content"

    @pytest.mark.asyncio
    async def test_list_resources_handler_returns_json(self):
        mgr = MCPManager()
        mgr.list_resources = AsyncMock(return_value=[{"uri": "a", "name": "a"}])

        handler = MCPListResourcesHandler(mgr)
        result = await handler.execute(server="s1")

        mgr.list_resources.assert_called_once_with(server_name="s1")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["uri"] == "a"
