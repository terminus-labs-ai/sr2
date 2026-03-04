from datetime import datetime, timezone
from sr2.resolvers.registry import ResolverContext, ResolvedContent, estimate_tokens


class RuntimeResolver:
    """Produces runtime values."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        if key == "current_timestamp":
            value = datetime.now(timezone.utc).isoformat()
        else:
            raise KeyError(f"Unknown runtime key: {key}")
        return ResolvedContent(key=key, content=value, tokens=estimate_tokens(value))
