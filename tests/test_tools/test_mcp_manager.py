"""Tests for MCP tool manager."""

import pytest

from sr2.tools.mcp_manager import MCPToolConfig, MCPToolManager
from sr2.tools.models import ToolDefinition


def _make_tools() -> dict[str, list[ToolDefinition]]:
    return {
        "filesystem": [
            ToolDefinition(name="read_file", description="Read a file from disk"),
            ToolDefinition(name="write_file", description="Write content to a file"),
            ToolDefinition(name="list_dir", description="List directory contents"),
        ],
        "database": [
            ToolDefinition(name="query_db", description="Execute SQL query"),
            ToolDefinition(name="insert_row", description="Insert a database row"),
        ],
    }


class TestMCPToolManager:

    def test_curated_only_specified(self):
        """Curated strategy returns only specified tools."""
        mgr = MCPToolManager(
            strategy="curated",
            mcp_configs=[MCPToolConfig("filesystem", curated_tools=["read_file"])],
            all_available_tools=_make_tools(),
        )
        tools = mgr.get_context_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_file"

    def test_curated_with_discovery(self):
        """curated_with_discovery returns curated + discover_mcp_tool."""
        mgr = MCPToolManager(
            strategy="curated_with_discovery",
            mcp_configs=[MCPToolConfig("filesystem", curated_tools=["read_file"])],
            all_available_tools=_make_tools(),
        )
        tools = mgr.get_context_tools()
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "discover_mcp_tool" in names

    def test_all_in_context(self):
        """all_in_context returns all tools from all servers."""
        mgr = MCPToolManager(
            strategy="all_in_context",
            all_available_tools=_make_tools(),
        )
        tools = mgr.get_context_tools()
        assert len(tools) == 5

    def test_curated_none_means_all_from_server(self):
        """Curated config with None curated_tools -> all tools from that server."""
        mgr = MCPToolManager(
            strategy="curated",
            mcp_configs=[MCPToolConfig("filesystem", curated_tools=None)],
            all_available_tools=_make_tools(),
        )
        tools = mgr.get_context_tools()
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_discover_by_name(self):
        """discover() finds tools by name keyword."""
        mgr = MCPToolManager(all_available_tools=_make_tools())
        results = await mgr.discover("query")
        assert len(results) == 1
        assert results[0].name == "query_db"

    @pytest.mark.asyncio
    async def test_discover_by_description(self):
        """discover() finds tools by description keyword."""
        mgr = MCPToolManager(all_available_tools=_make_tools())
        results = await mgr.discover("directory")
        assert len(results) == 1
        assert results[0].name == "list_dir"

    @pytest.mark.asyncio
    async def test_discover_no_match(self):
        """discover() returns empty for no match."""
        mgr = MCPToolManager(all_available_tools=_make_tools())
        results = await mgr.discover("zzzznonexistent")
        assert results == []

    def test_discovery_tool_schema(self):
        """Discovery tool has correct schema."""
        mgr = MCPToolManager(
            strategy="curated_with_discovery",
            mcp_configs=[],
            all_available_tools={},
        )
        tools = mgr.get_context_tools()
        discovery = [t for t in tools if t.name == "discover_mcp_tool"][0]
        assert discovery.type == "retrieval"
        assert len(discovery.parameters) == 1
        assert discovery.parameters[0].name == "query"
        assert discovery.parameters[0].required is True

    def test_missing_server_no_crash(self):
        """Missing server in all_available_tools -> no tools (no crash)."""
        mgr = MCPToolManager(
            strategy="curated",
            mcp_configs=[MCPToolConfig("nonexistent", curated_tools=["tool1"])],
            all_available_tools=_make_tools(),
        )
        tools = mgr.get_context_tools()
        assert len(tools) == 0
