"""Self-contained pub/sub event bus for the SR2 pipeline.

The bus handles subscription matching, event queuing, and asynchronous
draining. The Engine calls ``emit()`` which queues the event then drains
until all cascading events settle.

Key invariants
--------------
1. ``subscribe(sub, callback)`` is sync — callbacks registered immediately.
   ``sub`` can be a plain string or an ``EventSubscription``.
2. ``queue()`` is sync — fire-and-forget enqueue. Sync subscribers are
   notified immediately; async subscribers are notified during drain.
3. ``emit()`` is async — queues the event then runs the internal drain loop
   until the queue is empty AND all running tasks are done.
4. Async callbacks fire concurrently via ``asyncio.create_task()``.
5. ``bus_drained`` subscriptions are kept in a separate list and notified
   after drain completes, not via the main drain loop.
6. ``is_empty()`` reflects whether the internal queue has pending events.
7. ``reset()`` clears per-turn state but keeps subscriptions intact.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Awaitable, Callable, List, Set, Tuple, Union

from sr2.pipeline.events import Event, EventPhase, EventSubscription

logger = logging.getLogger(__name__)

Callback = Callable[[Event], Any]

_BUS_DRAINED = "bus_drained"


class EventBus:
    """Self-contained async event bus with concurrent callback dispatch.

    Subscribers register via ``subscribe(sub, callback)``. Callbacks
    can fire cascading events using ``bus.queue()`` (sync). The engine drives
    the pipeline by calling ``bus.emit(event)`` (async) which queues the event
    and drains until all events and tasks settle.

    Both sync and async callbacks are supported. Sync callbacks are dispatched
    immediately when events are queued. Async callbacks are dispatched during
    the drain loop via ``asyncio.create_task()``.

    ``bus_drained`` is a special event name — subscribers are stored in a
    separate list and notified after drain completes, not via the drain loop.

    The bus does NOT know about layers, handlers, or results — it is purely
    a delivery mechanism.
    """

    def __init__(self, max_drain_iterations: int = 100) -> None:
        # (event_name, callback, is_async) triples for normal subscriptions
        self._subs: List[Tuple[str, Callback, bool]] = []
        # Separate list for bus_drained subscribers
        self._drained_subs: List[Callback] = []
        self._queue: deque[Event] = deque()
        self._running_tasks: Set[asyncio.Task] = set()  # type: ignore[type-arg]
        self._max_drain_iterations = max_drain_iterations
        # Accumulated error strings for surfacing via PipelineMetrics.bus_errors
        self._errors: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        sub: Union[str, EventSubscription],
        callback: Callback,
    ) -> None:
        """Register a callback for an event name.

        ``sub`` can be a plain event name string or an ``EventSubscription``
        (the ``event_name`` field is extracted). Callbacks can be sync or
        async — sync callbacks are dispatched eagerly from ``queue()``,
        async callbacks during ``_drain()``.

        Special name ``"bus_drained"`` stores in a separate notification list.
        """
        event_name = sub.event_name if isinstance(sub, EventSubscription) else sub
        is_async = asyncio.iscoroutinefunction(callback)

        if event_name == _BUS_DRAINED:
            self._drained_subs.append(callback)
        else:
            self._subs.append((event_name, callback, is_async))

    def reset(self) -> None:
        """Clear per-turn state (queue, running tasks, errors) but keep subscriptions."""
        self._queue.clear()
        for task in self._running_tasks:
            task.cancel()
        self._running_tasks.clear()
        self._errors.clear()

    def get_errors(self) -> List[str]:
        """Return a snapshot of errors collected since the last reset()."""
        return list(self._errors)

    def queue(self, event: Event) -> None:
        """Fire-and-forget enqueue. Sync — safe to call from callbacks.

        Sync subscribers are notified immediately. Async subscribers are
        notified later during ``_drain()``.
        """
        self._queue.append(event)
        for event_name, callback, is_async in self._subs:
            if event_name == event.name and not is_async:
                try:
                    callback(event)
                except Exception as exc:
                    logger.error(
                        "EventBus sync callback raised an exception: %s",
                        exc,
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    self._errors.append(
                        f"sync callback error on event '{event.name}': {exc}"
                    )

    async def emit(self, event: Event) -> None:
        """Queue an event and drain until the queue is empty and tasks done."""
        self.queue(event)
        await self.drain()

    def is_empty(self) -> bool:
        """Return ``True`` if no events are pending in the queue."""
        return len(self._queue) == 0

    # ------------------------------------------------------------------
    # Drain loop (public)
    # ------------------------------------------------------------------

    async def drain(self) -> None:
        """Drain the queue until empty and all running tasks complete.

        Public method. The engine calls this after ``queue()``-ing lifecycle
        events to flush the bus between layer-processing cycles.

        Only async callbacks are dispatched here — sync callbacks were already
        called by ``queue()``.
        """
        iterations = 0

        while self._queue or self._running_tasks:
            while self._queue:
                event = self._queue.popleft()
                for event_name, callback, is_async in self._subs:
                    if event_name == event.name and is_async:
                        task = asyncio.create_task(callback(event))
                        self._running_tasks.add(task)

            if self._running_tasks:
                done, _ = await asyncio.wait(
                    self._running_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    self._running_tasks.discard(task)
                    exc = task.exception()
                    if exc is not None:
                        logger.error(
                            "EventBus callback raised an exception: %s",
                            exc,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
                        self._errors.append(
                            f"async callback error: {exc}"
                        )

            iterations += 1
            if iterations >= self._max_drain_iterations:
                logger.error(
                    "EventBus drain exceeded max_drain_iterations=%d — "
                    "possible infinite cascade. Aborting drain.",
                    self._max_drain_iterations,
                )
                self._errors.append(
                    f"drain aborted: exceeded max_drain_iterations={self._max_drain_iterations}"
                )
                return

        await self._notify_drained()

    async def _notify_drained(self) -> None:
        """Notify all bus_drained subscribers.

        After calling them, check if any events were queued. If so, discard
        them with a warning — bus_drained handlers must not trigger new drain
        cycles.
        """
        if not self._drained_subs:
            return

        drained_event = Event(
            name=_BUS_DRAINED,
            phase=EventPhase.COMPLETED,
            source_layer="event_bus",
        )

        # Call each drained subscriber directly (not via create_task, not via
        # the drain loop) so we can detect post-notification queue growth.
        for callback in self._drained_subs:
            try:
                await callback(drained_event)
            except Exception as exc:
                logger.error(
                    "EventBus bus_drained callback raised an exception: %s",
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                self._errors.append(
                    f"bus_drained callback error: {exc}"
                )

        # Discard any events queued during notification
        if self._queue:
            count = len(self._queue)
            logger.warning(
                "EventBus: %d event(s) queued during bus_drained notification — "
                "discarding. bus_drained handlers must not queue new events.",
                count,
            )
            self._queue.clear()
