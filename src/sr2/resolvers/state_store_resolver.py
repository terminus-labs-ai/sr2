"""Resolver that reads from agent_config state store values."""

from sr2.resolvers.registry import ResolverContext, ResolvedContent


class StateStoreResolver:
    """Reads state from context.agent_config[key]."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        value = context.agent_config.get(key, "{}")
        if not isinstance(value, str):
            import json

            value = json.dumps(value)
        return ResolvedContent(key=key, content=value, tokens=len(value.split()))
