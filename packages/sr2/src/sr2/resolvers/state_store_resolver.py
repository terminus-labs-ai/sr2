"""Resolver that reads from agent_config state store values."""

import json

from sr2.resolvers.registry import ResolverContext, ResolvedContent, estimate_tokens


class StateStoreResolver:
    """Reads state from context.agent_config[key]."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        if key not in context.agent_config:
            raise KeyError(f"Key '{key}' not found in agent_config state store")
        value = context.agent_config[key]
        if not isinstance(value, str):
            value = json.dumps(value)
        return ResolvedContent(key=key, content=value, tokens=estimate_tokens(value))
