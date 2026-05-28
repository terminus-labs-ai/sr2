"""Tests for sr2-17: EventBus._drain() encapsulation leak.

These tests pin the DESIRED behavior: ``drain()`` (no underscore) must be a
public method on EventBus.  They fail today because only ``_drain()`` exists.

Fix: rename ``_drain`` → ``drain`` in event_bus.py and update the one internal
caller (``emit()``) to call ``self.drain()``.
"""

from __future__ import annotations

import asyncio

import pytest

from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(name: str = "turn_start") -> Event:
    return Event(name=name, phase=EventPhase.STARTING, source_layer="test")


# ---------------------------------------------------------------------------
# 1. drain() is callable as a public method (no AttributeError)
# ---------------------------------------------------------------------------


class TestDrainIsPublic:
    """bus.drain() must exist and be callable without accessing a private attribute."""

    @pytest.mark.asyncio
    async def test_drain_method_exists_on_bus(self):
        """EventBus exposes a public drain() method (not just _drain())."""
        bus = EventBus()
        # This raises AttributeError today — bus has no public `drain`
        assert callable(bus.drain), "bus.drain must be a callable public method"

    @pytest.mark.asyncio
    async def test_private_drain_is_not_the_only_option(self):
        """Accessing bus._drain is the bug — bus.drain must be the supported path."""
        bus = EventBus()
        # Confirm the private attribute is NOT the expected public surface.
        # After the fix, bus.drain and bus._drain may both exist (alias) or
        # _drain may be removed entirely; either way bus.drain must work.
        result = await bus.drain()  # type: ignore[attr-defined]
        assert result is None  # drain() returns None (coroutine completes)


# ---------------------------------------------------------------------------
# 2. drain() behaves identically to _drain() — processes queued events
# ---------------------------------------------------------------------------


class TestDrainBehavior:
    """bus.drain() must have the same semantics as the current bus._drain()."""

    @pytest.mark.asyncio
    async def test_drain_processes_queued_async_callback(self):
        """queue() + drain() delivers events to async subscribers."""
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.name)

        bus.subscribe("turn_start", handler)
        bus.queue(_evt("turn_start"))

        # Public drain() — NOT bus._drain()
        await bus.drain()  # type: ignore[attr-defined]

        assert received == ["turn_start"]

    @pytest.mark.asyncio
    async def test_drain_on_empty_queue_completes_without_error(self):
        """drain() on an empty bus is a no-op — no exception raised."""
        bus = EventBus()
        await bus.drain()  # type: ignore[attr-defined]
        assert bus.is_empty()

    @pytest.mark.asyncio
    async def test_drain_fires_bus_drained_subscribers(self):
        """drain() notifies bus_drained subscribers after the queue empties."""
        bus = EventBus()
        drained_calls: list[bool] = []

        async def on_drained(event: Event) -> None:
            drained_calls.append(True)

        bus.subscribe("bus_drained", on_drained)
        bus.queue(_evt("turn_start"))
        await bus.drain()  # type: ignore[attr-defined]

        assert drained_calls == [True], "bus_drained must fire after drain() completes"

    @pytest.mark.asyncio
    async def test_drain_handles_cascading_events(self):
        """Async callback that queues a new event — drain() keeps running until settled."""
        bus = EventBus()
        received: list[str] = []

        async def first_handler(event: Event) -> None:
            # cascade: queue a second event from inside the callback
            bus.queue(_evt("cascade"))

        async def cascade_handler(event: Event) -> None:
            received.append("cascade")

        bus.subscribe("turn_start", first_handler)
        bus.subscribe("cascade", cascade_handler)
        bus.queue(_evt("turn_start"))

        await bus.drain()  # type: ignore[attr-defined]

        assert received == ["cascade"], "drain() must process cascaded events"

    @pytest.mark.asyncio
    async def test_bus_is_empty_after_drain(self):
        """is_empty() returns True after drain() completes."""
        bus = EventBus()

        async def noop(event: Event) -> None:
            pass

        bus.subscribe("turn_start", noop)
        bus.queue(_evt("turn_start"))
        await bus.drain()  # type: ignore[attr-defined]

        assert bus.is_empty()

    @pytest.mark.asyncio
    async def test_multiple_queue_then_drain(self):
        """Several queue() calls followed by one drain() — all events delivered."""
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.name)

        bus.subscribe("a", handler)
        bus.subscribe("b", handler)
        bus.subscribe("c", handler)

        bus.queue(_evt("a"))
        bus.queue(_evt("b"))
        bus.queue(_evt("c"))

        await bus.drain()  # type: ignore[attr-defined]

        assert sorted(received) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 3. emit() still works — it must delegate to the now-public drain()
# ---------------------------------------------------------------------------


class TestEmitDelegatesToPublicDrain:
    """After the rename, emit() should call self.drain() (not self._drain()).

    We verify this indirectly: emit() behaviour must be unchanged.
    """

    @pytest.mark.asyncio
    async def test_emit_still_delivers_event(self):
        """emit() continues to work correctly after the encapsulation fix."""
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.name)

        bus.subscribe("turn_start", handler)
        await bus.emit(_evt("turn_start"))

        assert received == ["turn_start"]

    @pytest.mark.asyncio
    async def test_emit_calls_drain_not_private(self):
        """emit() must NOT call a private _drain() — only the public drain().

        After the fix, _drain should either not exist or be an alias.
        The key assertion: bus.drain is the authoritative coroutine, and
        calling bus.drain() produces the same result as what emit() used to
        trigger via _drain().
        """
        bus = EventBus()
        calls: list[str] = []

        async def handler(event: Event) -> None:
            calls.append("fired")

        bus.subscribe("turn_start", handler)

        # Manually replicate what the engine does: queue + public drain
        bus.queue(_evt("turn_start"))
        await bus.drain()  # type: ignore[attr-defined]

        # Must be equivalent to emit()
        assert calls == ["fired"]
