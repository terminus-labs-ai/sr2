"""Provenance tracking for SR2 pipeline entries.

Provides Entry, EntryOrigin, the causal-metadata vocabulary
(ContextReason, ContextAuthority, ContextLifecycle, ContextScope),
ProvenanceStore protocol, and InMemoryProvenanceStore.
Phase 1: in-memory only. SQLiteProvenanceStore is Phase 1 (separate file).

Stores enforce the spec's append-only rules (FR4, FR5): a write for a
stored id never changes the stored entry, and a write whose ``sources``
reference ids absent from the store is rejected — dangling source ids are
invalid lineage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from sr2.models import ContentBlock, Message


class ContextReason(str, Enum):
    USER_INPUT = "user_input"
    ASSISTANT_OUTPUT = "assistant_output"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    SYSTEM_CONFIG = "system_config"
    LIVE_SOURCE = "live_source"
    ACTIVE_PLAN = "active_plan"
    RETRIEVAL = "retrieval"
    TRANSFORMATION = "transformation"


class ContextAuthority(str, Enum):
    USER = "user"
    LIVE_SOURCE = "live_source"
    VERIFIED = "verified"
    AGENT_TENTATIVE = "agent_tentative"
    DERIVED = "derived"


class ContextLifecycle(str, Enum):
    PINNED = "pinned"
    TURN = "turn"
    RECENT = "recent"
    FRAME = "frame"
    PROJECT = "project"
    RETRIEVED = "retrieved"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class ContextScope:
    owner_id: str
    agent_id: str | None = None
    projects: tuple[str, ...] = ()
    frame_id: str | None = None
    topics: tuple[str, ...] = ()
    sensitivity: tuple[str, ...] = ()


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
    reason: ContextReason = ContextReason.SYSTEM_CONFIG
    authority: ContextAuthority = ContextAuthority.DERIVED
    lifecycle: ContextLifecycle = ContextLifecycle.TURN
    scope: ContextScope = field(
        default_factory=lambda: ContextScope(owner_id="default")
    )
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
    """In-memory implementation of ProvenanceStore. No persistence. For tests and defaults.

    Append-only semantics (FR4, FR5): a second write for a stored id
    leaves the stored entry unchanged (first write wins), and a
    ``write()`` whose ``sources`` reference ids absent from the store
    raises ValueError. ``write_batch()`` is the pipeline's batch-flush
    path — it persists the batch in dependency order but does not
    re-validate sources, because the layer builds the whole turn's write
    set before anything is persisted.
    """

    def __init__(self) -> None:
        self._store: dict[str, Entry] = {}

    def _checked(self, entry: Entry) -> bool:
        """FR5 single-write validation. Returns True when sources resolve."""
        for source_id in entry.sources:
            if source_id not in self._store:
                raise ValueError(
                    f"Entry {entry.id} references source {source_id!r} "
                    f"which is not present in the store."
                )
        return True

    async def write(self, entry: Entry) -> None:
        self._checked(entry)
        # FR4: append-only — the first write for an id wins.
        if entry.id not in self._store:
            self._store[entry.id] = entry

    async def write_batch(self, entries: list[Entry]) -> None:
        # Batch-flush path: dangling sources are tolerated because a
        # caller may legitimately persist a graph whose external source
        # ids are not part of this partition.
        for entry in entries:
            if entry.id not in self._store:
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


# ---------------------------------------------------------------------------
# Turn-archive read surface (sr2-93 — AC2)
# ---------------------------------------------------------------------------

TURN_ARCHIVE_LAYER = "turn_archive"


async def read_turn_archive(
    store: ProvenanceStore, session_id: str, turn_seq: int
) -> list[Entry]:
    """Return one turn's archived event entries in SR2-visible order.

    Filters the session's entries on ``layer == "turn_archive"`` and
    ``meta["turn_seq"] == turn_seq`` and orders them by the durable
    per-turn ``meta["seq"]`` (the orchestrator's append order) — never by
    ``created_at`` or id, which carry no ordering information (identical
    timestamps and ULID-shaped ids are legitimate for one turn). Entries
    without a numeric seq sort last, stably.

    Works against any ProvenanceStore through ``get_session`` alone, so
    durable reopen (AC2) is the store's concern and this helper stays a
    pure filter/sort.
    """
    entries = await store.get_session(session_id, layer=TURN_ARCHIVE_LAYER)
    wanted = [
        entry
        for entry in entries
        if entry.meta.get("turn_seq") == turn_seq
        and not isinstance(entry.meta.get("turn_seq"), bool)
    ]

    def _seq_key(entry: Entry) -> tuple[int, int]:
        seq = entry.meta.get("seq")
        if isinstance(seq, int) and not isinstance(seq, bool):
            return (0, seq)
        return (1, 0)

    return sorted(wanted, key=_seq_key)
