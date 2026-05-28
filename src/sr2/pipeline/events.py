"""Event data models for the SR2 pipeline.

Defines the Event, EventPhase, and EventSubscription types used throughout
the pipeline. The EventBus (delivery mechanism) lives in event_bus.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class EventPhase(Enum):
    """Lifecycle phases an event can be in."""

    STARTING = "starting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Event:
    """A single pipeline event."""

    name: str
    phase: EventPhase
    source_layer: str
    data: Any = None
    iteration_seq: int = 1


@dataclass
class EventSubscription:
    """A subscription matches events by name only.

    The ``phase`` field is retained for legacy compatibility but is not
    used in matching. Phase filtering is the subscriber's responsibility.
    """

    event_name: str
    phase: Optional[EventPhase] = None

    def matches(self, event: Event) -> bool:
        return event.name == self.event_name
