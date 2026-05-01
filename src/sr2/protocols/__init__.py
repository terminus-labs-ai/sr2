"""SR2 v2 Protocol Definitions.

Runtime-checkable Protocol classes that define extension point contracts
for the SR2 context engineering library.

Each Protocol has a single responsibility (SRP) and is closed for modification
but open for extension (OCP). Shared value types are defined once (DRY) and
reused across protocols.

Usage:
    from sr2.protocols import ContentProvider, ContentReducer, MemoryStore, MetricExporter
    isinstance(provider, ContentProvider)  # works thanks to runtime_checkable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Value objects (shared data types)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderContext:
    """Context passed to a ContentProvider when resolving content.

    Attributes:
        session_id: Unique identifier for the current session.
        layer_name: Name of the layer requesting content.
        config: Arbitrary configuration dict for this provider invocation.
        topic_tags: Weighted topic relevance tags for content filtering.
    """

    session_id: str
    layer_name: str
    config: dict[str, Any] = field(default_factory=dict)
    topic_tags: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedContent:
    """Content returned by a ContentProvider after resolution.

    Attributes:
        content: The resolved text content.
        tokens: Estimated token count of the content.
        metadata: Arbitrary metadata attached during resolution.
    """

    content: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducedContent:
    """Content returned by a ContentReducer after reduction.

    Attributes:
        content: The reduced/compressed text.
        original_tokens: Token count before reduction.
        reduced_tokens: Token count after reduction.
        metadata: Arbitrary metadata describing the reduction.
    """

    content: str
    original_tokens: int
    reduced_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSnapshot:
    """A point-in-time capture of metrics for export.

    Attributes:
        timestamp: When this snapshot was taken.
        turn_id: Identifier for the conversation turn.
        metrics: Key-value pairs of metric measurements.
    """

    turn_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ContentProvider(Protocol):
    """Fetches content for a layer.

    Single responsibility: locate and retrieve raw or enriched content
    (files, API responses, embeddings, etc.) given a resolution context.

    Implementations are plugged into layers at runtime. The ``name`` property
    identifies the provider in logs and debug output.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this provider."""
        ...

    async def resolve(self, ctx: ProviderContext) -> ResolvedContent:
        """Resolve content for the given context.

        Args:
            ctx: The resolution context describing what content is needed.

        Returns:
            The resolved content with token count and metadata.
        """
        ...


@runtime_checkable
class ContentReducer(Protocol):
    """Transforms and compresses content within a layer.

    Single responsibility: reduce the token footprint of content while
    preserving semantic relevance. Implementations may use summarization,
    truncation, keyword extraction, or any other compression strategy.

    The ``name`` property identifies the reducer in logs and debug output.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this reducer."""
        ...

    async def reduce(self, content: str, budget: int) -> ReducedContent:
        """Reduce content to fit within a token budget.

        Args:
            content: The raw content to reduce.
            budget: Maximum target token count after reduction.

        Returns:
            The reduced content with before/after token counts.
        """
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence backend for agent memories.

    Single responsibility: CRUD and lifecycle operations on memory
    records. Implementations may back to SQLite, PostgreSQL, vector stores,
    in-memory dicts, or any other storage medium.

    The protocol is intentionally agnostic about the memory schema; it
    passes memory objects as opaque payloads and uses string identifiers
    for addressing.
    """

    async def save(self, memory: Any) -> str:
        """Persist a new memory record or update an existing one.

        Args:
            memory: The memory object to store. Schema is implementation-defined.

        Returns:
            The unique identifier assigned to the stored memory.
        """
        ...

    async def search(
        self, query: str, scope: str | None = None, limit: int = 20
    ) -> list[Any]:
        """Search memories by query text with optional scoping.

        Args:
            query: The search query string.
            scope: Optional scope filter (e.g., "session", "global").
            limit: Maximum number of results to return.

        Returns:
            A list of matching memory objects.
        """
        ...

    async def get(self, memory_id: str) -> Any | None:
        """Retrieve a single memory by its identifier.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The memory object, or None if not found.
        """
        ...

    async def delete(self, memory_id: str) -> bool:
        """Permanently remove a memory.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if the memory was deleted, False if not found.
        """
        ...

    async def list(
        self,
        status: str | None = None,
        scope: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Any]:
        """List memories with optional filters and pagination.

        Args:
            status: Optional status filter (e.g., "active", "archived").
            scope: Optional scope filter.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            A list of memory objects matching the criteria.
        """
        ...

    async def merge(self, ids: list[str]) -> str:
        """Merge multiple memories into a single consolidated memory.

        Args:
            ids: List of memory identifiers to merge.

        Returns:
            The identifier of the new merged memory.
        """
        ...

    async def archive(self, ids: list[str]) -> int:
        """Move memories to archived status without deleting them.

        Args:
            ids: List of memory identifiers to archive.

        Returns:
            The number of memories successfully archived.
        """
        ...


@runtime_checkable
class MetricExporter(Protocol):
    """Pushes or pulls metrics to an external destination.

    Single responsibility: ship metric snapshots to observability backends
    (Prometheus, Datadog, console, file, etc.). The ``close`` method
    allows implementations to flush buffers or release connections.
    """

    async def emit(self, snapshot: MetricSnapshot) -> None:
        """Export a single metric snapshot.

        Args:
            snapshot: The metric data to export.
        """
        ...

    def close(self) -> None:
        """Flush and release any held resources."""
        ...


# ---------------------------------------------------------------------------
# Re-exports for convenience
# ---------------------------------------------------------------------------

__all__ = [
    # Value objects
    "ProviderContext",
    "ResolvedContent",
    "ReducedContent",
    "MetricSnapshot",
    # Protocols
    "ContentProvider",
    "ContentReducer",
    "MemoryStore",
    "MetricExporter",
]
