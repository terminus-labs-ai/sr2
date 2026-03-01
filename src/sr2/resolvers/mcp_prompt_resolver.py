"""Resolver for MCP prompts.

Fetches filled prompt templates from MCP servers via an injected callable,
keeping the sr2 package independent of the runtime MCP client.
"""

from typing import Callable

from sr2.resolvers.registry import ResolvedContent, ResolverContext


class MCPPromptResolver:
    """Resolves content from MCP prompts.

    Pipeline config items use ``source: mcp_prompt`` with the key being
    the prompt name. Optional config fields: ``server``, ``arguments`` (dict).
    """

    def __init__(self, get_prompt_fn: Callable):
        """
        Args:
            get_prompt_fn: async (name: str, arguments: dict | None, server_name: str | None) -> str
        """
        self._get_prompt_fn = get_prompt_fn

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        server = config.get("server")
        arguments = config.get("arguments")
        content = await self._get_prompt_fn(key, arguments, server_name=server)
        return ResolvedContent(
            key=key,
            content=content,
            tokens=len(content.split()),
            metadata={"source": "mcp_prompt", "server": server},
        )
