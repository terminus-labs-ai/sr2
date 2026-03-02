from sr2.resolvers.registry import ResolverContext, ResolvedContent


class SessionResolver:
    """Reads from context.agent_config['session_history']."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        history = context.agent_config.get("session_history", [])
        window = config.get("window")
        if window is not None:
            history = history[-window:]
        formatted = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in history
        )
        return ResolvedContent(key=key, content=formatted, tokens=len(formatted.split()))
