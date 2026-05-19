"""Provenance tracking for SR2 pipeline entries.

Provides Entry, EntryOrigin, ProvenanceStore protocol, and InMemoryProvenanceStore.
Phase 1: in-memory only. SQLiteProvenanceStore is Phase 1 (separate file).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from sr2.models import ContentBlock, Message


@dataclass(frozen=True)
class EntryOrigin:
    kind: Literal["resolver", "transformer"]
    name: str


@dataclass(frozen=True)
class Entry:
    id: str                          # ULID string (26 chars, sortable)
    content: ContentBlock | Message  # the payload
    sources: tuple[str, ...]         # () for genesis entries
    origin: EntryOrigin
    layer: str
    session_id: str
    created_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.origin.kind == "transformer" and not self.sources:
            raise ValueError(
                f"Transformer-origin entry {self.id} has no sources. "
                f"Transformers must populate `sources` with the IDs they derived from."
            )
        if self.origin.kind == "resolver" and self.sources:
            raise ValueError(
                f"Resolver-origin entry {self.id} has sources. "
                f"Resolvers produce genesis entries (sources must be empty)."
            )


@runtime_checkable
class ProvenanceStore(Protocol):
    async def write(self, entry: Entry) -> None: ...
    async def write_batch(self, entries: list[Entry]) -> None: ...
    async def get(self, entry_id: str) -> Entry | None: ...
    async def get_lineage(self, entry_id: str, depth: int = -1) -> list[Entry]: ...
    async def get_session(
        self,
        session_id: str,
        layer: str | None = None,
        since: datetime | None = None,
    ) -> list[Entry]: ...


class InMemoryProvenanceStore:
    """In-memory implementation of ProvenanceStore. No persistence. For tests and defaults."""

    def __init__(self) -> None:
        self._store: dict[str, Entry] = {}

    async def write(self, entry: Entry) -> None:
        self._store[entry.id] = entry

    async def write_batch(self, entries: list[Entry]) -> None:
        for entry in entries:
            self._store[entry.id] = entry

    async def get(self, entry_id: str) -> Entry | None:
        return self._store.get(entry_id)

    async def get_lineage(self, entry_id: str, depth: int = -1) -> list[Entry]:
        """BFS traversal following sources links.

        depth=-1: unlimited traversal.
        depth=N: traverse at most N hops from the start entry.
        Returns [] if entry_id is unknown.
        """
        start = self._store.get(entry_id)
        if start is None:
            return []

        visited: dict[str, Entry] = {}
        queue: deque[tuple[str, int]] = deque()
        queue.append((entry_id, 0))

        while queue:
            current_id, current_depth = queue.popleft()
            if current_id in visited:
                continue
            entry = self._store.get(current_id)
            if entry is None:
                continue
            visited[current_id] = entry
            if depth == -1 or current_depth < depth:
                for source_id in entry.sources:
                    if source_id not in visited:
                        queue.append((source_id, current_depth + 1))

        return list(visited.values())

    async def get_session(
        self,
        session_id: str,
        layer: str | None = None,
        since: datetime | None = None,
    ) -> list[Entry]:
        results = []
        for entry in self._store.values():
            if entry.session_id != session_id:
                continue
            if layer is not None and entry.layer != layer:
                continue
            if since is not None and entry.created_at < since:
                continue
            results.append(entry)
        return results
