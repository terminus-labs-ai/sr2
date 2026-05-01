"""Core data models for SR2 v2.

These are the value objects and domain types used across the entire system.
Defined once (DRY), reused everywhere. No logic — pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory record."""
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"
    MERGED = "merged"


class MemoryType(str, Enum):
    """Semantic categorization of a memory."""
    IDENTITY = "identity"
    KNOWLEDGE = "knowledge"
    PREFERENCE = "preference"
    TASK = "task"
    EPHEMERAL = "ephemeral"


class MemoryScope(str, Enum):
    """Visibility boundary for a memory."""
    PRIVATE = "private"
    PROJECT = "project"
    TEAM = "team"
    SHARED = "shared"


class CachePolicy(str, Enum):
    """Cache invalidation strategy for a layer."""
    STATIC = "static"        # Never changes
    EPHEMERAL = "ephemeral"  # Stable between runs, invalidates on content change
    NONE = "none"            # No caching


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation from an LLM response."""
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class TurnResult:
    """Result of a single LLM turn, fed back to SR2 post_process().

    This is SR2's inbound schema — provider-agnostic. The harness translates
    Anthropic/OpenAI/etc responses into this format before calling SR2.
    """
    role: str                          # Always "assistant"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class Memory:
    """A single memory record.

    Schema matches the v2 redesign plan:
    - id: unique identifier
    - content: the memory text
    - key: semantic key for dedup/conflict detection
    - type: categorization (identity, knowledge, preference, task, ephemeral)
    - stability: 0.0-1.0 how likely to change
    - created_by: agent ID that created this
    - created_at: timestamp
    - source_session: which conversation it came from
    - last_accessed: staleness signal
    - access_count: usefulness signal
    - scope: visibility boundary (private, project, team, shared)
    - status: lifecycle state (active, stale, archived, merged)
    - embedding: vector for semantic search
    """
    content: str
    key: str
    type: MemoryType = MemoryType.KNOWLEDGE
    stability: float = 0.5
    created_by: str = ""
    source_session: str = ""
    scope: MemoryScope = MemoryScope.PRIVATE
    status: MemoryStatus = MemoryStatus.ACTIVE
    embedding: list[float] | None = None

    # Generated fields
    id: str = ""
    access_count: int = 0
