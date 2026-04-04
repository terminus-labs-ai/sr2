from sr2.resolvers.registry import ResolverContext, ResolvedContent, estimate_tokens


class StaticTemplateResolver:
    """Returns a fixed string from config['template']."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        if "template" not in config:
            raise KeyError("StaticTemplateResolver requires 'template' field in config")
        value = config["template"]
        return ResolvedContent(key=key, content=value, tokens=estimate_tokens(value))
