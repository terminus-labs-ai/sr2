"""Pipeline dependency container."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from sr2.protocols.llm import LLMCallable

if TYPE_CHECKING:
    from sr2.memory.protocol import MemoryExtractor, MemoryStore


@dataclasses.dataclass(frozen=True)
class Dependencies:
    """Immutable container for runtime dependencies injected into pipeline components."""

    llm: dict[str, LLMCallable] | None = None
    memory_store: "MemoryStore | None" = None
    memory_extractor: "MemoryExtractor | None" = None
    session_id: str = ""
    active_frame_provider: Callable[[str], str | None] | None = None
    """Origin-aware active-frame provider.

    When present, the orchestrator calls ``provider(origin)`` to resolve the
    active frame for the current turn's *origin*.  The provider returns the
    work-frame id if one is open on that origin, or the ambient frame id
    bound to the origin.  When ``None`` (default), no stamping occurs and
    core behaviour is unchanged (regression-safe).

    The *origin* parameter is typically a transport-identifier string
    (e.g. ``"tui"``, ``"discord:channel_id"``).
    """
