"""Memory subsystem contract.

MemoryStore is the protocol that implementations must satisfy. It covers
persistence, retrieval, and lifecycle for Memory entries. Extraction is a
separate concern — the store doesn't know how memories are created, only
how they're stored and found.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schema import ExtractionResult, Memory, MemoryScope, MemorySearchResult


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence and retrieval contract for memory entries."""

    def save(self, memory: Memory) -> Memory: ...
    def search(self, query: str, scope: MemoryScope | None = None, limit: int = 10) -> list[MemorySearchResult]: ...
    def delete(self, memory_id: str) -> bool: ...
    def get_all(self, scope: MemoryScope | None = None) -> list[Memory]: ...


@runtime_checkable
class TaggedMemoryStore(MemoryStore, Protocol):
    """Extension of MemoryStore for backends that support tag-based retrieval."""

    def get_by_tag(self, tag: str, scope: MemoryScope | None = None, limit: int = 10) -> list[MemorySearchResult]: ...


@runtime_checkable
class MemoryExtractor(Protocol):
    """Extract durable facts from a conversation turn."""

    def extract(self, turn_text: str, turn_id: str | None = None) -> ExtractionResult: ...
