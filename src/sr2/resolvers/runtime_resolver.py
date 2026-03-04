from datetime import datetime, timezone
from sr2.resolvers.registry import ResolverContext, ResolvedContent, estimate_tokens


class RuntimeResolver:
    """Produces runtime values."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        if key == "current_timestamp":
            now = datetime.now(timezone.utc)
            value = f"Current date and time (UTC): {now.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            raise KeyError(f"Unknown runtime key: {key}")
        return ResolvedContent(key=key, content=value, tokens=estimate_tokens(value))
