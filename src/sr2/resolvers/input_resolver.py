from sr2.resolvers.registry import ResolverContext, ResolvedContent


class InputResolver:
    """Reads from context.trigger_input."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        if isinstance(context.trigger_input, dict):
            if key not in context.trigger_input:
                raise KeyError(f"Key '{key}' not found in trigger_input")
            value = str(context.trigger_input[key])
        else:
            value = str(context.trigger_input)
        return ResolvedContent(key=key, content=value, tokens=len(value.split()))
