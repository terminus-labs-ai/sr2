"""Extended protocols for memory store backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2.memory.schema import Memory


@runtime_checkable
class LifecycleStore(Protocol):
    """Store that requires lifecycle management (e.g., table creation)."""

    async def create_tables(self) -> None: ...


@runtime_checkable
class EmbeddingStore(Protocol):
    """Store that supports embedding management operations."""

    async def update_embedding(self, memory_id: str, embedding: list[float]) -> None: ...

    async def list_without_embeddings(self, limit: int = 100) -> list[Memory]: ...
