"""Heartbeat data model — scheduled agent callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class HeartbeatStatus(str, Enum):
    """Lifecycle states for a heartbeat."""

    pending = "pending"
    firing = "firing"
    completed = "completed"
    cancelled = "cancelled"
    failed = "failed"


@dataclass
class Heartbeat:
    """A scheduled future callback for an agent."""

    id: str
    agent_name: str
    source_session: str
    prompt: str
    fire_at: datetime
    source_interface: str = ""
    context_turns: list[dict] = field(default_factory=list)
    status: HeartbeatStatus = HeartbeatStatus.pending
    key: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    recurring: bool = False
    interval_seconds: int | None = None
