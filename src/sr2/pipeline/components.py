"""Abstract base class for SR2 pipeline components.

Provides state machine, execution counting, self-subscription, and turn_start
reset logic for resolvers and transformers.

FRs covered:
  FR14: turn_start resets execution_count to 0 and state to idle
  FR18: Components self-subscribe to bus during initialization
  FR24: Three component states: idle, pending, exhausted (and is_done semantics)
  FR25: State transitions — idle→pending→idle/exhausted, any→idle on turn_start
  FR26: tracks_pending=False — ignores STARTING, fires on COMPLETED, never blocks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase


class ComponentState(Enum):
    """States a pipeline component can be in."""

    IDLE = "idle"
    PENDING = "pending"
    EXHAUSTED = "exhausted"


class BaseComponent(ABC):
    """Abstract base class for pipeline components (resolvers and transformers).

    Lifecycle
    ---------
    - IDLE: waiting for the next trigger
    - PENDING: a STARTING event was received; waiting for COMPLETED
    - EXHAUSTED: max_executions reached; ignores all further events until
      turn_start resets it

    Self-subscription
    -----------------
    The component subscribes itself to the bus on construction. Each name in
    ``subscriptions`` receives ``_on_event``. ``turn_start`` always receives
    ``_on_turn_start``, exactly once.

    tracks_pending
    --------------
    When False, STARTING events are ignored and COMPLETED events fire
    ``_execute`` directly from IDLE (the component never enters PENDING).
    """

    def __init__(
        self,
        name: str,
        subscriptions: list[str],
        bus: EventBus,
        layer: Any,
        max_executions: int = 1,
        tracks_pending: bool = True,
    ) -> None:
        self.name = name
        self.layer = layer
        self._max_executions = max_executions
        self._tracks_pending = tracks_pending

        self._state: ComponentState = ComponentState.IDLE
        self._execution_count: int = 0

        # Self-subscribe
        for event_name in subscriptions:
            bus.subscribe(event_name, self._on_event)
        bus.subscribe("turn_start", self._on_turn_start)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ComponentState:
        return self._state

    @property
    def execution_count(self) -> int:
        return self._execution_count

    @property
    def is_done(self) -> bool:
        """True when the component is not blocking (IDLE or EXHAUSTED)."""
        return self._state in (ComponentState.IDLE, ComponentState.EXHAUSTED)

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    async def _on_event(self, event: Event) -> None:
        """Main event callback — handles STARTING, COMPLETED, FAILED phases."""
        if self._state is ComponentState.EXHAUSTED:
            return

        if event.phase is EventPhase.STARTING:
            if self._tracks_pending:
                self._state = ComponentState.PENDING
            # If not tracks_pending, ignore STARTING entirely

        elif event.phase is EventPhase.COMPLETED:
            # Execute regardless of whether we went through PENDING (handles
            # both tracks_pending=True and tracks_pending=False flows, plus
            # the edge case of COMPLETED without preceding STARTING).
            await self._execute(event)
            self._execution_count += 1
            if self._execution_count >= self._max_executions:
                self._state = ComponentState.EXHAUSTED
            else:
                self._state = ComponentState.IDLE

        # FAILED: no state change, no execute

    async def _on_turn_start(self, event: Event) -> None:
        """Reset state and execution count at the start of each turn."""
        self._state = ComponentState.IDLE
        self._execution_count = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, event: Event) -> None:
        """Subclasses implement component-specific logic here."""
        ...
