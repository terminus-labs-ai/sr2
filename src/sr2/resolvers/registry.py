from dataclasses import dataclass
from typing import Protocol, Any


@dataclass
class ResolverContext:
    """Context passed to every resolver."""

    agent_config: dict
    trigger_input: Any
    session_id: str | None = None
    interface_type: str = ""


@dataclass
class ResolvedContent:
    """Output of a content resolver."""

    key: str
    content: str
    tokens: int
    metadata: dict | None = None


class ContentResolver(Protocol):
    """Protocol that all resolvers must implement."""

    async def resolve(
        self,
        key: str,
        config: dict,
        context: ResolverContext,
    ) -> ResolvedContent: ...


class ContentResolverRegistry:
    """Registry mapping source names to resolver instances."""

    def __init__(self) -> None:
        self._resolvers: dict[str, ContentResolver] = {}

    def register(self, source_name: str, resolver: ContentResolver) -> None:
        """Register a resolver for a source name. Overwrites existing."""
        self._resolvers[source_name] = resolver

    def get(self, source_name: str) -> ContentResolver:
        """Get resolver by source name. Raises KeyError if not found."""
        if source_name not in self._resolvers:
            raise KeyError(f"No resolver registered for source: {source_name}")
        return self._resolvers[source_name]

    def has(self, source_name: str) -> bool:
        return source_name in self._resolvers

    @property
    def registered_sources(self) -> list[str]:
        return list(self._resolvers.keys())
