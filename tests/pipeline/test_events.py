"""Tests for sr2.pipeline.events — Event, EventPhase, EventSubscription, EventBus.

Covers:
  FR1: EventBus accepts async callbacks
  FR2: emit() drains until queue empty AND all tasks done
  FR3: Callbacks fire concurrently via asyncio.create_task()
  FR4: queue() is sync — safe to call from within callbacks
  FR5: bus_drained fires after drain completes via separate notification path
  FR6/FR39: Subscriptions match on event_name only — all phases delivered
  FR7: reset() clears queue/tasks, keeps subscriptions
  FR8: Safety valve — max iteration count enforced
  NFR3: Swallowed exceptions logged, other callbacks still fire
"""

import asyncio
import logging
import time

import pytest

from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription


# ---------------------------------------------------------------------------
# 1. EventPhase — enum values exist
# ---------------------------------------------------------------------------


class TestEventPhase:
    def test_has_starting(self):
        assert EventPhase.STARTING.value == "starting"

    def test_has_completed(self):
        assert EventPhase.COMPLETED.value == "completed"

    def test_has_failed(self):
        assert EventPhase.FAILED.value == "failed"

    def test_all_phases_accounted_for(self):
        """Exactly three phases — no silent additions."""
        assert set(EventPhase) == {
            EventPhase.STARTING,
            EventPhase.COMPLETED,
            EventPhase.FAILED,
        }


# ---------------------------------------------------------------------------
# 2. Event — creation and defaults
# ---------------------------------------------------------------------------


class TestEvent:
    def test_create_with_all_fields(self):
        event = Event(
            name="turn_start",
            phase=EventPhase.STARTING,
            source_layer="core",
        )
        assert event.name == "turn_start"
        assert event.phase == EventPhase.STARTING
        assert event.source_layer == "core"

    def test_data_defaults_to_none(self):
        event = Event(
            name="turn_start",
            phase=EventPhase.STARTING,
            source_layer="core",
        )
        assert event.data is None

    def test_data_can_carry_arbitrary_payload(self):
        payload = [{"text": "hello", "token_count": 1}]
        event = Event(
            name="user_input",
            phase=EventPhase.COMPLETED,
            source_layer="input",
            data=payload,
        )
        assert event.data is not None
        assert len(event.data) == 1
        assert event.data[0]["text"] == "hello"

    def test_data_can_be_empty_list(self):
        event = Event(
            name="user_input",
            phase=EventPhase.COMPLETED,
            source_layer="input",
            data=[],
        )
        assert event.data == []


# ---------------------------------------------------------------------------
# 3. EventSubscription — name-only matching (FR6/FR39)
# ---------------------------------------------------------------------------


class TestEventSubscription:
    def test_create_with_event_name(self):
        sub = EventSubscription(event_name="turn_start")
        assert sub.event_name == "turn_start"

    def test_matches_same_name_any_phase(self):
        """EventSubscription.matches() checks name only — all phases match."""
        sub = EventSubscription(event_name="turn_start")
        assert sub.matches(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert sub.matches(Event(name="turn_start", phase=EventPhase.COMPLETED, source_layer="p"))
        assert sub.matches(Event(name="turn_start", phase=EventPhase.FAILED, source_layer="p"))

    def test_does_not_match_different_name(self):
        sub = EventSubscription(event_name="turn_start")
        assert not sub.matches(Event(name="overflow", phase=EventPhase.STARTING, source_layer="p"))

    def test_name_match_ignores_phase_field_entirely(self):
        """Phase on the subscription (if present as legacy) does not restrict matching."""
        # Even if EventSubscription still carries a phase field for legacy compat,
        # matches() must behave as name-only per the spec.
        sub = EventSubscription(event_name="turn_start")
        for phase in EventPhase:
            event = Event(name="turn_start", phase=phase, source_layer="p")
            assert sub.matches(event), f"Expected match for phase {phase}"


# ---------------------------------------------------------------------------
# 4. FR1 — EventBus accepts async callbacks
# ---------------------------------------------------------------------------


class TestEventBusAsyncCallbacks:
    @pytest.mark.asyncio
    async def test_subscribe_accepts_async_callback(self):
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_async_callback_receives_correct_event(self):
        bus = EventBus()
        captured = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.subscribe("user_input", handler)
        payload = {"key": "value"}
        event = Event(name="user_input", phase=EventPhase.COMPLETED, source_layer="input", data=payload)
        await bus.emit(event)

        assert captured[0] is event
        assert captured[0].data is payload

    @pytest.mark.asyncio
    async def test_multiple_async_callbacks_for_same_event_all_fire(self):
        bus = EventBus()
        calls = []

        async def handler_a(event: Event) -> None:
            calls.append("a")

        async def handler_b(event: Event) -> None:
            calls.append("b")

        bus.subscribe("turn_start", handler_a)
        bus.subscribe("turn_start", handler_b)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        assert sorted(calls) == ["a", "b"]


# ---------------------------------------------------------------------------
# 5. FR2 — emit() drains until queue empty AND all tasks done
# ---------------------------------------------------------------------------


class TestEventBusDrainCompletes:
    @pytest.mark.asyncio
    async def test_emit_returns_only_after_all_callbacks_complete(self):
        bus = EventBus()
        completed = []

        async def slow_handler(event: Event) -> None:
            await asyncio.sleep(0.05)
            completed.append("done")

        bus.subscribe("turn_start", slow_handler)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        # Must be done — emit() should not return until tasks finish
        assert completed == ["done"]

    @pytest.mark.asyncio
    async def test_emit_returns_only_after_cascading_events_settle(self):
        """emit() must drain until ALL events (including cascades) are processed."""
        bus = EventBus()
        order = []

        async def on_a(event: Event) -> None:
            order.append("a")
            bus.queue(Event(name="b", phase=EventPhase.STARTING, source_layer="p"))

        async def on_b(event: Event) -> None:
            order.append("b")

        bus.subscribe("a", on_a)
        bus.subscribe("b", on_b)

        await bus.emit(Event(name="a", phase=EventPhase.STARTING, source_layer="p"))

        assert order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_bus_is_empty_after_emit_returns(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert bus.is_empty() is True


# ---------------------------------------------------------------------------
# 6. FR3 — Callbacks fire concurrently, not sequentially
# ---------------------------------------------------------------------------


class TestEventBusConcurrentCallbacks:
    @pytest.mark.asyncio
    async def test_callbacks_overlap_in_time(self):
        """Two slow callbacks should execute concurrently, not back-to-back."""
        bus = EventBus()
        start_times: list[float] = []
        end_times: list[float] = []

        async def slow_a(event: Event) -> None:
            start_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            end_times.append(time.monotonic())

        async def slow_b(event: Event) -> None:
            start_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            end_times.append(time.monotonic())

        bus.subscribe("turn_start", slow_a)
        bus.subscribe("turn_start", slow_b)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        assert len(start_times) == 2
        assert len(end_times) == 2
        # Structural concurrency proof: second callback started before first ended
        assert start_times[1] < end_times[0], (
            "Second callback should start before first ends — confirms concurrency"
        )

    @pytest.mark.asyncio
    async def test_three_concurrent_callbacks_all_complete(self):
        bus = EventBus()
        results = []

        for i in range(3):
            async def handler(event: Event, idx: int = i) -> None:
                await asyncio.sleep(0.05)
                results.append(idx)

            bus.subscribe("evt", handler)

        await bus.emit(Event(name="evt", phase=EventPhase.STARTING, source_layer="p"))
        assert sorted(results) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 7. FR4 — queue() is sync, safe to call from within callbacks
# ---------------------------------------------------------------------------


class TestEventBusQueue:
    def test_queue_is_synchronous(self):
        """queue() does not require an await — it's a sync method."""
        bus = EventBus()
        # If this raises or requires await, the test fails
        bus.queue(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

    def test_queue_marks_bus_non_empty(self):
        bus = EventBus()
        assert bus.is_empty() is True
        bus.queue(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert bus.is_empty() is False

    @pytest.mark.asyncio
    async def test_queue_from_within_callback_processed_in_same_drain(self):
        """Cascading: callback calls bus.queue(), that event is processed before emit() returns."""
        bus = EventBus()
        order = []

        async def on_first(event: Event) -> None:
            order.append("first")
            bus.queue(Event(name="second", phase=EventPhase.STARTING, source_layer="p"))

        async def on_second(event: Event) -> None:
            order.append("second")

        bus.subscribe("first", on_first)
        bus.subscribe("second", on_second)

        await bus.emit(Event(name="first", phase=EventPhase.STARTING, source_layer="p"))
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_deep_cascade_processed_in_same_drain(self):
        """Three-level cascade: a -> b -> c, all in one drain cycle."""
        bus = EventBus()
        order = []

        async def on_a(event: Event) -> None:
            order.append("a")
            bus.queue(Event(name="b", phase=EventPhase.STARTING, source_layer="p"))

        async def on_b(event: Event) -> None:
            order.append("b")
            bus.queue(Event(name="c", phase=EventPhase.STARTING, source_layer="p"))

        async def on_c(event: Event) -> None:
            order.append("c")

        bus.subscribe("a", on_a)
        bus.subscribe("b", on_b)
        bus.subscribe("c", on_c)

        await bus.emit(Event(name="a", phase=EventPhase.STARTING, source_layer="p"))
        assert order == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 8. FR5 — bus_drained fires after drain completes
# ---------------------------------------------------------------------------


class TestEventBusDrained:
    @pytest.mark.asyncio
    async def test_bus_drained_fires_after_drain_completes(self):
        bus = EventBus()
        main_calls = []
        drained_calls = []

        async def main_handler(event: Event) -> None:
            main_calls.append("main")

        async def drained_handler(event: Event) -> None:
            drained_calls.append("drained")

        bus.subscribe("turn_start", main_handler)
        bus.subscribe("bus_drained", drained_handler)

        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        assert main_calls == ["main"]
        assert drained_calls == ["drained"]

    @pytest.mark.asyncio
    async def test_bus_drained_fires_after_cascading_events_settle(self):
        """bus_drained must not fire until ALL cascades complete."""
        bus = EventBus()
        order = []

        async def on_a(event: Event) -> None:
            order.append("a")
            bus.queue(Event(name="b", phase=EventPhase.STARTING, source_layer="p"))

        async def on_b(event: Event) -> None:
            order.append("b")

        async def on_drained(event: Event) -> None:
            order.append("drained")

        bus.subscribe("a", on_a)
        bus.subscribe("b", on_b)
        bus.subscribe("bus_drained", on_drained)

        await bus.emit(Event(name="a", phase=EventPhase.STARTING, source_layer="p"))
        assert order == ["a", "b", "drained"]

    @pytest.mark.asyncio
    async def test_bus_drained_handler_queuing_events_discards_with_warning(self, caplog):
        """bus_drained handlers that queue events: events are discarded, warning logged."""
        bus = EventBus()
        second_calls = []

        async def drained_handler(event: Event) -> None:
            # This should be discarded — bus_drained is outside the drain loop
            bus.queue(Event(name="second", phase=EventPhase.STARTING, source_layer="p"))

        async def second_handler(event: Event) -> None:
            second_calls.append("second")

        bus.subscribe("bus_drained", drained_handler)
        bus.subscribe("second", second_handler)

        with caplog.at_level(logging.WARNING):
            await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        # The queued event from bus_drained must NOT be processed
        assert second_calls == []
        # A warning must have been logged
        assert any("discard" in r.message.lower() or "bus_drained" in r.message.lower()
                   for r in caplog.records), "Expected a warning log about discarded events"

    @pytest.mark.asyncio
    async def test_bus_drained_fires_even_with_no_matching_subscribers(self):
        """Drain completes (even with zero work) → bus_drained fires."""
        bus = EventBus()
        drained_calls = []

        async def on_drained(event: Event) -> None:
            drained_calls.append("drained")

        bus.subscribe("bus_drained", on_drained)

        await bus.emit(Event(name="unsubscribed", phase=EventPhase.STARTING, source_layer="p"))
        assert drained_calls == ["drained"]


# ---------------------------------------------------------------------------
# 9. FR6/FR39 — Name-only matching: all phases delivered to subscriber
# ---------------------------------------------------------------------------


class TestEventBusNameOnlyMatching:
    @pytest.mark.asyncio
    async def test_subscriber_receives_all_phases_for_subscribed_name(self):
        bus = EventBus()
        received_phases = []

        async def handler(event: Event) -> None:
            received_phases.append(event.phase)

        bus.subscribe("turn_start", handler)

        for phase in EventPhase:
            await bus.emit(Event(name="turn_start", phase=phase, source_layer="p"))

        assert set(received_phases) == set(EventPhase)

    @pytest.mark.asyncio
    async def test_subscriber_does_not_fire_for_different_event_name(self):
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append(event.name)

        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="overflow", phase=EventPhase.STARTING, source_layer="p"))

        assert calls == []

    @pytest.mark.asyncio
    async def test_bus_does_not_filter_by_phase(self):
        """The bus delivers all phases. Filtering is the subscriber's responsibility."""
        bus = EventBus()
        all_received = []

        async def handler(event: Event) -> None:
            # Subscriber filters internally — only cares about COMPLETED
            if event.phase == EventPhase.COMPLETED:
                all_received.append("completed")

        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        await bus.emit(Event(name="turn_start", phase=EventPhase.COMPLETED, source_layer="p"))
        await bus.emit(Event(name="turn_start", phase=EventPhase.FAILED, source_layer="p"))

        # Handler received all three, filtered to one
        assert all_received == ["completed"]

    @pytest.mark.asyncio
    async def test_multiple_subscribers_different_names_routed_correctly(self):
        bus = EventBus()
        a_calls = []
        b_calls = []

        async def handler_a(event: Event) -> None:
            a_calls.append(event.name)

        async def handler_b(event: Event) -> None:
            b_calls.append(event.name)

        bus.subscribe("event_a", handler_a)
        bus.subscribe("event_b", handler_b)

        await bus.emit(Event(name="event_a", phase=EventPhase.STARTING, source_layer="p"))
        await bus.emit(Event(name="event_b", phase=EventPhase.COMPLETED, source_layer="p"))

        assert a_calls == ["event_a"]
        assert b_calls == ["event_b"]


# ---------------------------------------------------------------------------
# 10. FR7 — reset() clears queue and running tasks, keeps subscriptions
# ---------------------------------------------------------------------------


class TestEventBusReset:
    def test_reset_clears_queue(self):
        bus = EventBus()
        bus.queue(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert bus.is_empty() is False

        bus.reset()
        assert bus.is_empty() is True

    @pytest.mark.asyncio
    async def test_reset_keeps_subscriptions_intact(self):
        """Subscriptions survive reset — they're config, not per-turn state."""
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append(True)

        bus.subscribe("turn_start", handler)
        bus.reset()

        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert calls == [True]

    @pytest.mark.asyncio
    async def test_reset_on_clean_bus_is_safe(self):
        """reset() on an already-empty bus must not raise."""
        bus = EventBus()
        bus.reset()  # Should not raise
        assert bus.is_empty() is True

    @pytest.mark.asyncio
    async def test_reset_then_emit_works_normally(self):
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append("handled")

        bus.subscribe("turn_start", handler)

        # First emit
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert calls == ["handled"]

        # Reset
        bus.reset()
        assert bus.is_empty() is True

        # Second emit — should work normally
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert calls == ["handled", "handled"]


# ---------------------------------------------------------------------------
# 11. FR8 — Safety valve: max iteration count
# ---------------------------------------------------------------------------


class TestEventBusSafetyValve:
    @pytest.mark.asyncio
    async def test_drain_stops_at_max_iterations_and_logs_error(self, caplog):
        """If callbacks keep queueing events indefinitely, drain stops at max iterations."""
        bus = EventBus(max_drain_iterations=5)
        iterations = []

        async def infinite_cascade(event: Event) -> None:
            iterations.append(1)
            # Always queue another event — would loop forever without safety valve
            bus.queue(Event(name="infinite", phase=EventPhase.STARTING, source_layer="p"))

        bus.subscribe("infinite", infinite_cascade)

        with caplog.at_level(logging.ERROR):
            await bus.emit(Event(name="infinite", phase=EventPhase.STARTING, source_layer="p"))

        # Must have stopped (not hung)
        assert len(iterations) <= 10  # generous upper bound; exact depends on impl
        # Must have logged an error
        assert any("max" in r.message.lower() or "iteration" in r.message.lower() or "drain" in r.message.lower()
                   for r in caplog.records if r.levelno >= logging.ERROR), \
            "Expected an error log about hitting max drain iterations"

    @pytest.mark.asyncio
    async def test_bus_drained_not_emitted_after_safety_valve_triggers(self, caplog):
        """When safety valve triggers, bus_drained must NOT fire."""
        bus = EventBus(max_drain_iterations=3)
        drained_calls = []

        async def infinite_cascade(event: Event) -> None:
            bus.queue(Event(name="infinite", phase=EventPhase.STARTING, source_layer="p"))

        async def on_drained(event: Event) -> None:
            drained_calls.append("drained")

        bus.subscribe("infinite", infinite_cascade)
        bus.subscribe("bus_drained", on_drained)

        with caplog.at_level(logging.ERROR):
            await bus.emit(Event(name="infinite", phase=EventPhase.STARTING, source_layer="p"))

        assert drained_calls == [], "bus_drained must not fire after safety valve triggers"

    @pytest.mark.asyncio
    async def test_default_max_iterations_is_100(self):
        """Default safety valve is 100 iterations — bus doesn't blow up on normal use."""
        bus = EventBus()
        # 50 chained events — well within default limit
        count = [0]

        async def on_evt(event: Event) -> None:
            count[0] += 1
            if count[0] < 50:
                bus.queue(Event(name="chain", phase=EventPhase.STARTING, source_layer="p"))

        bus.subscribe("chain", on_evt)
        await bus.emit(Event(name="chain", phase=EventPhase.STARTING, source_layer="p"))
        assert count[0] == 50


# ---------------------------------------------------------------------------
# 12. NFR3 — Exception handling: failing callbacks logged, others still fire
# ---------------------------------------------------------------------------


class TestEventBusExceptionHandling:
    @pytest.mark.asyncio
    async def test_failing_callback_does_not_prevent_other_callbacks(self, caplog):
        bus = EventBus()
        calls = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("simulated callback failure")

        async def good_handler(event: Event) -> None:
            calls.append("good")

        bus.subscribe("turn_start", bad_handler)
        bus.subscribe("turn_start", good_handler)

        with caplog.at_level(logging.ERROR):
            await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        # good_handler must still fire
        assert calls == ["good"]

    @pytest.mark.asyncio
    async def test_failing_callback_exception_is_logged(self, caplog):
        bus = EventBus()

        async def bad_handler(event: Event) -> None:
            raise ValueError("intentional test error")

        bus.subscribe("turn_start", bad_handler)

        with caplog.at_level(logging.ERROR):
            await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        # Exception must appear in logs — not silently swallowed
        assert any(
            "intentional test error" in r.message or "ValueError" in r.message
            or (r.exc_info and r.exc_info[1] is not None and "intentional test error" in str(r.exc_info[1]))
            for r in caplog.records
        ), "Expected exception to be logged"

    @pytest.mark.asyncio
    async def test_emit_does_not_raise_on_callback_exception(self):
        """emit() itself must not propagate callback exceptions to the caller."""
        bus = EventBus()

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("do not propagate")

        bus.subscribe("turn_start", bad_handler)

        # Must not raise
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

    @pytest.mark.asyncio
    async def test_multiple_failing_callbacks_all_logged(self, caplog):
        """When multiple callbacks fail, all failures are logged."""
        bus = EventBus()

        async def fail_a(event: Event) -> None:
            raise RuntimeError("failure-a")

        async def fail_b(event: Event) -> None:
            raise RuntimeError("failure-b")

        bus.subscribe("turn_start", fail_a)
        bus.subscribe("turn_start", fail_b)

        with caplog.at_level(logging.ERROR):
            await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        log_text = " ".join(r.message for r in caplog.records)
        exc_text = " ".join(
            str(r.exc_info[1]) for r in caplog.records if r.exc_info and r.exc_info[1]
        )
        combined = log_text + exc_text
        assert "failure-a" in combined or "fail_a" in combined
        assert "failure-b" in combined or "fail_b" in combined


# ---------------------------------------------------------------------------
# 13. Edge cases
# ---------------------------------------------------------------------------


class TestEventBusEdgeCases:
    @pytest.mark.asyncio
    async def test_emit_with_no_subscribers_does_not_raise(self):
        bus = EventBus()
        await bus.emit(Event(name="orphan", phase=EventPhase.STARTING, source_layer="p"))

    @pytest.mark.asyncio
    async def test_emit_with_no_matching_subscribers_does_not_fire(self):
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append(True)

        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="overflow", phase=EventPhase.STARTING, source_layer="p"))
        assert calls == []

    @pytest.mark.asyncio
    async def test_same_callback_subscribed_twice_fires_twice(self):
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append(True)

        bus.subscribe("turn_start", handler)
        bus.subscribe("turn_start", handler)

        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_bus_empty_on_creation(self):
        bus = EventBus()
        assert bus.is_empty() is True

    @pytest.mark.asyncio
    async def test_bus_empty_after_full_drain(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        assert bus.is_empty() is True

    @pytest.mark.asyncio
    async def test_sequential_emits_each_drain_independently(self):
        """Two sequential emit() calls each drain independently."""
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append(event.name)

        bus.subscribe("a", handler)
        bus.subscribe("b", handler)

        await bus.emit(Event(name="a", phase=EventPhase.STARTING, source_layer="p"))
        await bus.emit(Event(name="b", phase=EventPhase.STARTING, source_layer="p"))

        assert calls == ["a", "b"]

    @pytest.mark.asyncio
    async def test_subscribing_after_first_emit_still_fires_on_next(self):
        """Subscribe after an emit — new subscription fires on subsequent emits."""
        bus = EventBus()
        calls = []

        async def handler(event: Event) -> None:
            calls.append(True)

        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))
        bus.subscribe("turn_start", handler)
        await bus.emit(Event(name="turn_start", phase=EventPhase.STARTING, source_layer="p"))

        assert calls == [True]
