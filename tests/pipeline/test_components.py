"""Tests for sr2.pipeline.components — BaseComponent and ComponentState.

Covers:
  FR14: turn_start resets execution_count to 0 and state to idle
  FR18: Components self-subscribe to bus during initialization
  FR24: Three component states: idle, pending, exhausted (and their is_done semantics)
  FR25: State transitions — idle→pending→idle/exhausted, any→idle on turn_start
  FR26: tracks_pending=False — ignores STARTING, fires on COMPLETED, never blocks layer
"""

import pytest

from sr2.pipeline.components import BaseComponent, ComponentState
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase


# ---------------------------------------------------------------------------
# Stub — concrete subclass for testing (records executed events)
# ---------------------------------------------------------------------------


class StubComponent(BaseComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_events: list[Event] = []

    async def _execute(self, event: Event) -> None:
        self.executed_events.append(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(name: str, phase: EventPhase, source: str = "test") -> Event:
    return Event(name=name, phase=phase, source_layer=source)


def make_component(
    name: str = "stub",
    subscriptions: list[str] | None = None,
    bus: EventBus | None = None,
    layer=None,
    max_executions: int = 1,
    tracks_pending: bool = True,
) -> tuple[StubComponent, EventBus]:
    if bus is None:
        bus = EventBus()
    comp = StubComponent(
        name=name,
        subscriptions=subscriptions or ["test_event"],
        bus=bus,
        layer=layer,
        max_executions=max_executions,
        tracks_pending=tracks_pending,
    )
    return comp, bus


# ---------------------------------------------------------------------------
# 1. ComponentState enum
# ---------------------------------------------------------------------------


class TestComponentState:
    def test_has_idle(self):
        assert ComponentState.IDLE is not None

    def test_has_pending(self):
        assert ComponentState.PENDING is not None

    def test_has_exhausted(self):
        assert ComponentState.EXHAUSTED is not None

    def test_exactly_three_states(self):
        """Exactly three states — no silent additions."""
        assert set(ComponentState) == {
            ComponentState.IDLE,
            ComponentState.PENDING,
            ComponentState.EXHAUSTED,
        }


# ---------------------------------------------------------------------------
# 2. Construction and initial state
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_creates_with_required_params(self):
        bus = EventBus()
        comp = StubComponent(
            name="my_comp",
            subscriptions=["some_event"],
            bus=bus,
            layer=None,
        )
        assert comp.name == "my_comp"

    def test_initial_state_is_idle(self):
        comp, _ = make_component()
        assert comp.state == ComponentState.IDLE

    def test_initial_execution_count_is_zero(self):
        comp, _ = make_component()
        assert comp.execution_count == 0

    def test_is_done_true_when_idle(self):
        comp, _ = make_component()
        assert comp.is_done is True

    def test_accepts_custom_max_executions(self):
        comp, _ = make_component(max_executions=5)
        assert comp.execution_count == 0
        assert comp.state == ComponentState.IDLE

    def test_accepts_layer_none(self):
        """Layer=None is valid — state machine does not require it."""
        comp, _ = make_component(layer=None)
        assert comp is not None

    def test_accepts_tracks_pending_false(self):
        comp, _ = make_component(tracks_pending=False)
        assert comp.state == ComponentState.IDLE


# ---------------------------------------------------------------------------
# 3. Self-subscription (FR18)
# ---------------------------------------------------------------------------


class TestSelfSubscription:
    def test_subscribes_to_configured_event(self):
        bus = EventBus()
        comp = StubComponent(
            name="comp",
            subscriptions=["my_event"],
            bus=bus,
            layer=None,
        )
        # The subscription should be registered — check by examining bus internals
        registered_names = [name for name, _, _ in bus._subs]
        assert "my_event" in registered_names

    def test_subscribes_to_turn_start_automatically(self):
        bus = EventBus()
        comp = StubComponent(
            name="comp",
            subscriptions=["my_event"],
            bus=bus,
            layer=None,
        )
        registered_names = [name for name, _, _ in bus._subs]
        assert "turn_start" in registered_names

    def test_subscribes_to_multiple_events(self):
        bus = EventBus()
        comp = StubComponent(
            name="comp",
            subscriptions=["event_a", "event_b", "event_c"],
            bus=bus,
            layer=None,
        )
        registered_names = [name for name, _, _ in bus._subs]
        assert "event_a" in registered_names
        assert "event_b" in registered_names
        assert "event_c" in registered_names

    def test_turn_start_registered_once_regardless_of_subscriptions_count(self):
        """turn_start should be registered exactly once."""
        bus = EventBus()
        comp = StubComponent(
            name="comp",
            subscriptions=["a", "b", "c"],
            bus=bus,
            layer=None,
        )
        count = sum(1 for name, _, _ in bus._subs if name == "turn_start")
        assert count == 1


# ---------------------------------------------------------------------------
# 4. State transitions (FR25)
# ---------------------------------------------------------------------------


class TestStateTransitions:
    @pytest.mark.asyncio
    async def test_starting_event_moves_to_pending(self):
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.state == ComponentState.PENDING

    @pytest.mark.asyncio
    async def test_pending_state_is_not_done(self):
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.is_done is False

    @pytest.mark.asyncio
    async def test_completed_while_pending_calls_execute(self):
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        completed = make_event("test_event", EventPhase.COMPLETED)
        await bus.emit(completed)
        assert len(comp.executed_events) == 1
        assert comp.executed_events[0] is completed

    @pytest.mark.asyncio
    async def test_completed_at_max_moves_to_exhausted(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED

    @pytest.mark.asyncio
    async def test_completed_below_max_returns_to_idle(self):
        comp, bus = make_component(max_executions=3)
        # First execution
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.IDLE
        assert comp.execution_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_ignores_further_events(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED

        # Another full cycle — should be ignored
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert len(comp.executed_events) == 1  # Still only one execution
        assert comp.state == ComponentState.EXHAUSTED

    @pytest.mark.asyncio
    async def test_exhausted_is_done(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.is_done is True


# ---------------------------------------------------------------------------
# 5. turn_start reset (FR14)
# ---------------------------------------------------------------------------


class TestTurnStartReset:
    @pytest.mark.asyncio
    async def test_turn_start_resets_state_from_idle(self):
        comp, bus = make_component()
        # Already idle — turn_start should keep it idle
        await bus.emit(make_event("turn_start", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE

    @pytest.mark.asyncio
    async def test_turn_start_resets_state_from_pending(self):
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.state == ComponentState.PENDING
        await bus.emit(make_event("turn_start", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE

    @pytest.mark.asyncio
    async def test_turn_start_resets_state_from_exhausted(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED
        await bus.emit(make_event("turn_start", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE

    @pytest.mark.asyncio
    async def test_turn_start_resets_execution_count(self):
        comp, bus = make_component(max_executions=3)
        # Execute once
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.execution_count == 1
        await bus.emit(make_event("turn_start", EventPhase.STARTING))
        assert comp.execution_count == 0

    @pytest.mark.asyncio
    async def test_after_turn_start_component_can_be_triggered_again(self):
        comp, bus = make_component(max_executions=1)
        # Run to exhaustion
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED
        assert len(comp.executed_events) == 1

        # Reset via turn_start
        await bus.emit(make_event("turn_start", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE

        # Now can execute again
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert len(comp.executed_events) == 2
        assert comp.state == ComponentState.EXHAUSTED


# ---------------------------------------------------------------------------
# 6. tracks_pending=False (FR26)
# ---------------------------------------------------------------------------


class TestTracksPendingFalse:
    @pytest.mark.asyncio
    async def test_starting_event_ignored_state_stays_idle(self):
        comp, bus = make_component(tracks_pending=False)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE

    @pytest.mark.asyncio
    async def test_completed_event_fires_execute_from_idle(self):
        comp, bus = make_component(tracks_pending=False)
        completed = make_event("test_event", EventPhase.COMPLETED)
        await bus.emit(completed)
        assert len(comp.executed_events) == 1
        assert comp.executed_events[0] is completed

    @pytest.mark.asyncio
    async def test_completed_moves_to_exhausted_at_max(self):
        comp, bus = make_component(tracks_pending=False, max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED

    @pytest.mark.asyncio
    async def test_completed_stays_idle_below_max(self):
        comp, bus = make_component(tracks_pending=False, max_executions=3)
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.IDLE
        assert comp.execution_count == 1

    @pytest.mark.asyncio
    async def test_is_done_true_after_starting_event(self):
        """tracks_pending=False never enters PENDING — layer never blocked."""
        comp, bus = make_component(tracks_pending=False)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.is_done is True

    @pytest.mark.asyncio
    async def test_is_done_true_before_any_event(self):
        comp, bus = make_component(tracks_pending=False)
        assert comp.is_done is True

    @pytest.mark.asyncio
    async def test_without_starting_completed_still_works(self):
        """COMPLETED without preceding STARTING fires execute normally."""
        comp, bus = make_component(tracks_pending=False)
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert len(comp.executed_events) == 1


# ---------------------------------------------------------------------------
# 7. Execution counting
# ---------------------------------------------------------------------------


class TestExecutionCounting:
    @pytest.mark.asyncio
    async def test_execution_count_increments_after_each_execute(self):
        comp, bus = make_component(max_executions=3)
        for i in range(1, 4):
            await bus.emit(make_event("test_event", EventPhase.STARTING))
            await bus.emit(make_event("test_event", EventPhase.COMPLETED))
            if i < 3:
                assert comp.execution_count == i
            else:
                assert comp.execution_count == 3  # At exhaustion

    @pytest.mark.asyncio
    async def test_max_executions_1_exhausted_after_one(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.execution_count == 1
        assert comp.state == ComponentState.EXHAUSTED

    @pytest.mark.asyncio
    async def test_max_executions_3_exhausted_after_three(self):
        comp, bus = make_component(max_executions=3)
        for _ in range(3):
            await bus.emit(make_event("test_event", EventPhase.STARTING))
            await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.execution_count == 3
        assert comp.state == ComponentState.EXHAUSTED

    @pytest.mark.asyncio
    async def test_exhausted_completed_does_not_increment_count(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.execution_count == 1

        # Extra cycle should not increment
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.execution_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_completed_does_not_call_execute(self):
        comp, bus = make_component(max_executions=1)
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert len(comp.executed_events) == 1

        # Extra cycle
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert len(comp.executed_events) == 1  # Unchanged


# ---------------------------------------------------------------------------
# 8. Event filtering
# ---------------------------------------------------------------------------


class TestEventFiltering:
    @pytest.mark.asyncio
    async def test_different_event_name_no_state_change(self):
        comp, bus = make_component(subscriptions=["test_event"])
        await bus.emit(make_event("other_event", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE

    @pytest.mark.asyncio
    async def test_different_event_name_no_execute(self):
        comp, bus = make_component(subscriptions=["test_event"])
        await bus.emit(make_event("other_event", EventPhase.COMPLETED))
        assert len(comp.executed_events) == 0

    @pytest.mark.asyncio
    async def test_failed_phase_no_state_change(self):
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.FAILED))
        assert comp.state == ComponentState.IDLE

    @pytest.mark.asyncio
    async def test_failed_phase_no_execute(self):
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.FAILED))
        assert len(comp.executed_events) == 0

    @pytest.mark.asyncio
    async def test_failed_phase_while_pending_no_change(self):
        """FAILED does not resolve pending — component stays pending."""
        comp, bus = make_component()
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.state == ComponentState.PENDING
        await bus.emit(make_event("test_event", EventPhase.FAILED))
        assert comp.state == ComponentState.PENDING

    @pytest.mark.asyncio
    async def test_correct_event_name_triggers_component(self):
        comp, bus = make_component(subscriptions=["the_right_event"])
        await bus.emit(make_event("the_right_event", EventPhase.STARTING))
        assert comp.state == ComponentState.PENDING


# ---------------------------------------------------------------------------
# 9. Integration with bus — full round-trips
# ---------------------------------------------------------------------------


class TestIntegrationWithBus:
    @pytest.mark.asyncio
    async def test_full_round_trip_starting_then_completed(self):
        comp, bus = make_component(max_executions=1)

        assert comp.state == ComponentState.IDLE
        assert comp.is_done is True

        await bus.emit(make_event("test_event", EventPhase.STARTING))
        assert comp.state == ComponentState.PENDING
        assert comp.is_done is False

        completed = make_event("test_event", EventPhase.COMPLETED)
        await bus.emit(completed)
        assert len(comp.executed_events) == 1
        assert comp.executed_events[0] is completed
        assert comp.state == ComponentState.EXHAUSTED
        assert comp.is_done is True

    @pytest.mark.asyncio
    async def test_turn_start_via_bus_resets_component(self):
        comp, bus = make_component(max_executions=1)

        # Run to exhaustion
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED

        # Reset via bus turn_start event
        await bus.emit(make_event("turn_start", EventPhase.STARTING))
        assert comp.state == ComponentState.IDLE
        assert comp.execution_count == 0
        assert comp.is_done is True

    @pytest.mark.asyncio
    async def test_multiple_components_on_same_bus_independent(self):
        bus = EventBus()
        comp_a = StubComponent(
            name="comp_a",
            subscriptions=["event_a"],
            bus=bus,
            layer=None,
            max_executions=1,
        )
        comp_b = StubComponent(
            name="comp_b",
            subscriptions=["event_b"],
            bus=bus,
            layer=None,
            max_executions=1,
        )

        # Trigger only comp_a
        await bus.emit(make_event("event_a", EventPhase.STARTING))
        await bus.emit(make_event("event_a", EventPhase.COMPLETED))

        assert comp_a.state == ComponentState.EXHAUSTED
        assert len(comp_a.executed_events) == 1
        assert comp_b.state == ComponentState.IDLE
        assert len(comp_b.executed_events) == 0

    @pytest.mark.asyncio
    async def test_turn_start_resets_all_components_on_shared_bus(self):
        bus = EventBus()
        comp_a = StubComponent(
            name="comp_a",
            subscriptions=["event_a"],
            bus=bus,
            layer=None,
            max_executions=1,
        )
        comp_b = StubComponent(
            name="comp_b",
            subscriptions=["event_b"],
            bus=bus,
            layer=None,
            max_executions=1,
        )

        # Run both to exhaustion
        await bus.emit(make_event("event_a", EventPhase.STARTING))
        await bus.emit(make_event("event_a", EventPhase.COMPLETED))
        await bus.emit(make_event("event_b", EventPhase.STARTING))
        await bus.emit(make_event("event_b", EventPhase.COMPLETED))

        assert comp_a.state == ComponentState.EXHAUSTED
        assert comp_b.state == ComponentState.EXHAUSTED

        # Turn start resets both
        await bus.emit(make_event("turn_start", EventPhase.STARTING))

        assert comp_a.state == ComponentState.IDLE
        assert comp_a.execution_count == 0
        assert comp_b.state == ComponentState.IDLE
        assert comp_b.execution_count == 0

    @pytest.mark.asyncio
    async def test_multi_execution_full_cycle(self):
        """max_executions=2 — two full starting/completed cycles before exhausted."""
        comp, bus = make_component(max_executions=2)

        # First cycle
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.IDLE
        assert comp.execution_count == 1

        # Second cycle
        await bus.emit(make_event("test_event", EventPhase.STARTING))
        await bus.emit(make_event("test_event", EventPhase.COMPLETED))
        assert comp.state == ComponentState.EXHAUSTED
        assert comp.execution_count == 2
        assert len(comp.executed_events) == 2
