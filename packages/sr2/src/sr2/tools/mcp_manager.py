"""MCP tool management — curated sets, discovery, and all-in-context strategies."""

import logging

from sr2.tools.models import ToolDefinition, ToolParameter

logger = logging.getLogger(__name__)


class MCPToolConfig:
    """Configuration for an MCP server's tools."""

    def __init__(self, server_name: str, curated_tools: list[str] | None = None):
        self.server_name = server_name
        self.curated_tools = curated_tools  # None = all tools from this server


class MCPToolManager:
    """Manages MCP tool definitions for an agent.

    Strategies:
    - curated: only specified tools from each MCP server
    - curated_with_discovery: curated + a discover_mcp_tool meta-tool
    - all_in_context: all tools from all servers
    """

    def __init__(
        self,
        strategy: str = "curated_with_discovery",
        mcp_configs: list[MCPToolConfig] | None = None,
        all_available_tools: dict[str, list[ToolDefinition]] | None = None,
    ):
        self._strategy = strategy
        self._configs = {c.server_name: c for c in (mcp_configs or [])}
        self._all_tools = all_available_tools or {}

    def get_context_tools(self) -> list[ToolDefinition]:
        """Get the tools that should be in the LLM's context."""
        if self._strategy == "all_in_context":
            return self._get_all_tools()

        curated = self._get_curated_tools()

        if self._strategy == "curated_with_discovery":
            curated.append(self._get_discovery_tool())

        logger.info(f"Curated list of tools: {curated}")
        return curated

    def _get_all_tools(self) -> list[ToolDefinition]:
        """Return all tools from all servers."""
        tools = []
        for server_tools in self._all_tools.values():
            tools.extend(server_tools)
        return tools

    def _get_curated_tools(self) -> list[ToolDefinition]:
        """Return only curated tools from configured servers."""
        tools = []
        for server_name, config in self._configs.items():
            server_tools = self._all_tools.get(server_name, [])
            if config.curated_tools is None:
                tools.extend(server_tools)
            else:
                curated_set = set(config.curated_tools)
                tools.extend(t for t in server_tools if t.name in curated_set)
        return tools

    def _get_discovery_tool(self) -> ToolDefinition:
        """Return the discovery meta-tool definition."""
        return ToolDefinition(
            name="discover_mcp_tool",
            type="retrieval",
            description=(
                "Search available MCP tools by capability description. "
                "Use when you need a tool that isn't in your current set."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Description of the capability you need",
                    required=True,
                ),
            ],
            category="read",
        )

    async def discover(self, query: str) -> list[ToolDefinition]:
        """Search all available MCP tools by description.

        Simple keyword matching for now.
        """
        query_lower = query.lower()
        results = []
        for server_tools in self._all_tools.values():
            for tool in server_tools:
                if query_lower in tool.name.lower() or query_lower in tool.description.lower():
                    results.append(tool)
        return results
