from sr2.resolvers.registry import ResolverContext, ResolvedContent, estimate_tokens


class ConfigResolver:
    """Reads from context.agent_config[key]."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        if key not in context.agent_config:
            raise KeyError(f"Key '{key}' not found in agent_config")
        value = str(context.agent_config[key])
        return ResolvedContent(key=key, content=value, tokens=estimate_tokens(value))
