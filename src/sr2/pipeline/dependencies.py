"""Pipeline dependency container."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

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
    extras: Mapping[str, Any] = dataclasses.field(default_factory=dict)
