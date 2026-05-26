"""Memory data types — value objects for the Memory subsystem.

All models are Pydantic BaseModels (frozen where identity matters) matching
the existing sr2 pattern: simple data containers, no behavior.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    """Visibility boundary for a memory entry."""

    PRIVATE = "private"    # Agent-only (internal state, scratchpads)
    PROJECT = "project"    # Shared within a project/session
    SHARED = "shared"      # Cross-project (user preferences, corrections)


class Memory(BaseModel):
    """A single memory entry — the durable fact extracted from a session turn."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    scope: MemoryScope = MemoryScope.PRIVATE
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    frequency: int = 0  # How many times this fact was re-extracted (reinforcement signal)
    last_accessed: datetime | None = None


class MemorySearchResult(BaseModel):
    """Projected view returned by search — drops mutable internals."""

    id: str
    content: str
    score: float
    scope: MemoryScope
    tags: list[str]


class ExtractionResult(BaseModel):
    """Batch of memories extracted from a single conversation turn."""

    memories: list[Memory] = Field(default_factory=list)
    source_turn_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
