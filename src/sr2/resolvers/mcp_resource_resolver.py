"""Resolver for MCP resources.

Fetches content from MCP servers via an injected read function,
keeping the sr2 package independent of the runtime MCP client.
"""

from typing import Callable

from sr2.resolvers.registry import ResolvedContent, ResolverContext


class MCPResourceResolver:
    """Resolves content from MCP resources.

    Pipeline config items use ``source: mcp_resource`` with the key being
    the resource URI. Optional config fields: ``server`` (server name).
    """

    def __init__(self, read_fn: Callable):
        """
        Args:
            read_fn: async (uri: str, server_name: str | None) -> str
        """
        self._read_fn = read_fn

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        server = config.get("server")
        content = await self._read_fn(key, server_name=server)
        return ResolvedContent(
            key=key,
            content=content,
            tokens=len(content.split()),
            metadata={"source": "mcp_resource", "server": server},
        )
