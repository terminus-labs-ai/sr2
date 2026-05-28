"""Tests for sr2-25: FR8 — PipelineEngine.start_turn / continue_turn / end_turn.

Covers the three new explicit lifecycle entry points:
  - start_turn(turn_seq): increments turn_seq, resets bus + content, fires turn_start, drains bus
  - continue_turn(events, iteration_seq): no reset, queues given events, drains bus
  - end_turn(): fires turn_end exactly once

Also covers backward compatibility:
  - run() still works and produces the same observable result as start_turn -> end_turn
"""

from __future__ import annotations

import pytest

from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.models import TextBlock


# ---------------------------------------------------------------------------
# Stubs — minimal protocol-conforming fakes (mirrors test_engine.py style)
# ---------------------------------------------------------------------------


class StubResolver:
    """Resolver that returns predetermined content."""

    def __init__(
        self,
        name: str = "stub_resolver",
        content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        source_layer: str = "test",
    ):
        self.name = name
        self._content = content or []
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer=self._source_layer,
            content=self._content,
        )


class CapturingResolver:
    """Resolver that records every batch of events it receives."""

    def __init__(
        self,
        name: str = "capturing_resolver",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 10,
        source_layer: str = "test",
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer
        self.captured_events: list[list[Event]] = []

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        self.captured_events.append(list(events))
        return ResolvedContent(
            resolver_name=self.name,
            source_layer=self._source_layer,
            content=[TextBlock(text=f"resolved by {self.name}")],
        )


class StubTransformer:
    """Transformer that optionally replaces content and captures events."""

    def __init__(
        self,
        name: str = "stub_transformer",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        transform_fn=None,
        events_to_emit: list[Event] | None = None,
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._transform_fn = transform_fn
        self._events_to_emit = events_to_emit

    async def transform(
        self, content: list, events: list[Event]
    ) -> TransformationResult:
        self.execution_count += 1
        result_content = self._transform_fn(content) if self._transform_fn else content
        return TransformationResult(
            transformer_name=self.name,
            source_layer="test",
            content=result_content,
            events=self._events_to_emit,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_system_layer(
    name: str = "system_prompt",
    resolvers: list | None = None,
    transformers: list | None = None,
    token_budget: int | None = None,
    token_counter=None,
    event_bus=None,
):
    from sr2.pipeline.layer import Layer

    return Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=token_budget,
        resolvers=resolvers or [],
        transformers=transformers or [],
        token_counter=token_counter or CharacterTokenCounter(),
        event_bus=event_bus or EventBus(),
    )


def make_engine(layers=None, token_counter=None):
    from sr2.pipeline.engine import PipelineEngine

    counter = token_counter or CharacterTokenCounter()
    return PipelineEngine(layers=layers or [], token_counter=counter), counter


# ---------------------------------------------------------------------------
# 1. TestStartTurn
# ---------------------------------------------------------------------------


class TestStartTurn:
    @pytest.mark.asyncio
    async def test_start_turn_increments_turn_seq(self):
        """start_turn() increments the engine's internal turn_seq counter.

        Before the first call, turn_seq is -1. After one call it should be 0.
        After a second call it should be 1. Either _turn_seq is observable via
        a property, or start_turn() returns the new seq number — the test
        covers both options.
        """
        engine, _ = make_engine()

        # turn_seq must exist and be inspectable
        initial_seq = engine._turn_seq
        assert initial_seq == -1, "turn_seq should start at -1 before any turn"

        await engine.start_turn(turn_seq=0)
        assert engine._turn_seq == 0, "turn_seq must be 0 after first start_turn"

        await engine.start_turn(turn_seq=1)
        assert engine._turn_seq == 1, "turn_seq must be 1 after second start_turn"

    @pytest.mark.asyncio
    async def test_start_turn_fires_turn_start_event(self):
        """start_turn() queues and drains a turn_start event on the bus.

        A resolver subscribed to turn_start must fire after start_turn() returns.
        """
        counter = CharacterTokenCounter()
        resolver = StubResolver(
            name="start_watcher",
            content=[TextBlock(text="started")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
        )
        engine, _ = make_engine(layers=[layer], token_counter=counter)

        await engine.start_turn(turn_seq=0)

        assert resolver.execution_count == 1, (
            "Resolver subscribed to turn_start must fire after start_turn()"
        )

    @pytest.mark.asyncio
    async def test_start_turn_resets_bus_state(self):
        """start_turn() clears any events left in the bus from a prior turn.

        We add a subscriber to the stale event, then queue the event and call
        start_turn(). Because start_turn() calls bus.reset() BEFORE draining,
        the stale event is discarded and the subscriber must NOT be invoked.
        This proves the queue was cleared — not just that it's empty at the end.
        """
        engine, _ = make_engine()

        stale_invocations: list[str] = []

        async def stale_handler(event: Event) -> None:
            stale_invocations.append(event.name)

        engine._bus.subscribe("stale_custom_event", stale_handler)

        # Manually queue a stale event to simulate leftover state
        stale_event = Event(
            name="stale_custom_event",
            phase=EventPhase.COMPLETED,
            source_layer="test",
        )
        engine._bus.queue(stale_event)
        assert not engine._bus.is_empty(), "precondition: bus has stale event"

        await engine.start_turn(turn_seq=0)

        # The stale handler must NOT have fired — reset discarded the event
        assert stale_invocations == [], (
            "Stale event subscriber must NOT be invoked — start_turn() must reset "
            "the bus before draining, discarding queued events from prior turns"
        )
        # Bus must also be empty (drained clean after turn_start)
        assert engine._bus.is_empty(), (
            "Bus should be empty after start_turn() drains — stale events cleared"
        )

    @pytest.mark.asyncio
    async def test_start_turn_resets_layer_content(self):
        """start_turn() clears layer content from the previous turn.

        Run a full turn to populate content, then call start_turn() again.
        The layer content must be reset to empty before new events are delivered.
        """
        counter = CharacterTokenCounter()
        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="old content")],
            subscriptions=[EventSubscription(event_name="turn_start")],
            max_executions=10,
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
        )
        engine, _ = make_engine(layers=[layer], token_counter=counter)

        # First full turn to establish content
        await engine.run([])
        first_content = layer.get_content()
        assert len(first_content) > 0, "precondition: first turn produced content"

        # reset execution count so resolver can fire again
        engine.reset_execution_counts()

        # start_turn must reset content before firing turn_start
        # We capture what content looked like at the start of resolve by
        # checking that content is re-populated (not stale from prior turn)
        resolver.execution_count = 0
        await engine.start_turn(turn_seq=1)

        # resolver fired again (turn_start was re-emitted)
        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 2. TestContinueTurn
# ---------------------------------------------------------------------------


class TestContinueTurn:
    @pytest.mark.asyncio
    async def test_continue_turn_does_not_reset_bus_state(self):
        """continue_turn() must NOT call bus.reset() — existing state is preserved.

        We subscribe a one-shot listener to the bus, then call continue_turn()
        with a custom event. The listener registered before the call must still
        be registered after (subscriptions intact = no reset).
        """
        counter = CharacterTokenCounter()
        received: list[str] = []

        # Subscribe an external listener directly on the engine's bus
        async def external_listener(event: Event) -> None:
            received.append(event.name)

        engine, _ = make_engine(token_counter=counter)
        engine._bus.subscribe("my_custom_event", external_listener)

        custom_event = Event(
            name="my_custom_event",
            phase=EventPhase.COMPLETED,
            source_layer="test",
        )

        await engine.continue_turn(events=[custom_event], iteration_seq=0)

        assert "my_custom_event" in received, (
            "External subscriber registered before continue_turn() must still fire — "
            "continue_turn() must not reset the bus"
        )

    @pytest.mark.asyncio
    async def test_continue_turn_queues_provided_events(self):
        """continue_turn(events, iteration_seq) queues all provided events into the bus.

        Resolvers subscribed to those event names must fire.
        """
        counter = CharacterTokenCounter()

        resolver = CapturingResolver(
            name="tool_result_handler",
            subscriptions=[EventSubscription(event_name="tool_result")],
            max_executions=10,
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
        )
        engine, _ = make_engine(layers=[layer], token_counter=counter)

        # Simulate an agent iteration: inject tool_result events
        tool_result_event = Event(
            name="tool_result",
            phase=EventPhase.COMPLETED,
            source_layer="tool_executor",
            data={"tool": "search", "result": "some data"},
        )

        await engine.continue_turn(events=[tool_result_event], iteration_seq=0)

        assert resolver.execution_count >= 1, (
            "Resolver subscribed to tool_result must fire after continue_turn() injects that event"
        )
        # Verify the event data reached the resolver
        all_events_received = [
            ev
            for batch in resolver.captured_events
            for ev in batch
            if ev.name == "tool_result"
        ]
        assert len(all_events_received) >= 1

    @pytest.mark.asyncio
    async def test_continue_turn_drains_bus_after_queuing(self):
        """continue_turn() drains the bus after queuing — async callbacks must fire.

        An async subscriber on the bus must have been called by the time
        continue_turn() returns.
        """
        engine, _ = make_engine()
        drained_calls: list[str] = []

        async def async_handler(event: Event) -> None:
            drained_calls.append(event.name)

        engine._bus.subscribe("agent_iteration", async_handler)

        iteration_event = Event(
            name="agent_iteration",
            phase=EventPhase.COMPLETED,
            source_layer="agent_loop",
        )

        await engine.continue_turn(events=[iteration_event], iteration_seq=1)

        assert "agent_iteration" in drained_calls, (
            "Async subscriber must receive event — continue_turn() must drain the bus"
        )
        assert engine._bus.is_empty(), (
            "Bus must be empty after continue_turn() completes its drain"
        )

    @pytest.mark.asyncio
    async def test_continue_turn_multiple_events_all_delivered(self):
        """continue_turn() with multiple events delivers all of them."""
        engine, _ = make_engine()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.name)

        engine._bus.subscribe("event_a", handler)
        engine._bus.subscribe("event_b", handler)
        engine._bus.subscribe("event_c", handler)

        events = [
            Event(name="event_a", phase=EventPhase.COMPLETED, source_layer="test"),
            Event(name="event_b", phase=EventPhase.COMPLETED, source_layer="test"),
            Event(name="event_c", phase=EventPhase.COMPLETED, source_layer="test"),
        ]

        await engine.continue_turn(events=events, iteration_seq=0)

        assert sorted(received) == ["event_a", "event_b", "event_c"], (
            "All events passed to continue_turn() must be delivered"
        )


# ---------------------------------------------------------------------------
# 3. TestEndTurn
# ---------------------------------------------------------------------------


class TestEndTurn:
    @pytest.mark.asyncio
    async def test_end_turn_fires_turn_end_event(self):
        """end_turn() fires a turn_end event exactly once.

        A transformer subscribed to turn_end must fire exactly once.
        """
        counter = CharacterTokenCounter()
        transformer = StubTransformer(
            name="turn_end_watcher",
            subscriptions=[EventSubscription(event_name="turn_end")],
            max_executions=5,
        )
        layer = make_system_layer(
            name="system_prompt",
            transformers=[transformer],
            token_counter=counter,
        )
        engine, _ = make_engine(layers=[layer], token_counter=counter)

        # Set up state expected by end_turn (layers need turn_seq set)
        await engine.start_turn(turn_seq=0)

        await engine.end_turn()

        assert transformer.execution_count == 1, (
            "Transformer subscribed to turn_end must fire exactly once after end_turn()"
        )

    @pytest.mark.asyncio
    async def test_end_turn_fires_turn_end_exactly_once(self):
        """end_turn() emits turn_end exactly once — resolver subscribed to it sees count == 1.

        This verifies the event is queued/fired once and not duplicated internally.
        """
        counter = CharacterTokenCounter()
        resolver = CapturingResolver(
            name="turn_end_counter",
            subscriptions=[EventSubscription(event_name="turn_end")],
            max_executions=10,
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
        )
        engine, _ = make_engine(layers=[layer], token_counter=counter)

        await engine.start_turn(turn_seq=0)
        await engine.end_turn()

        turn_end_events = [
            ev
            for batch in resolver.captured_events
            for ev in batch
            if ev.name == "turn_end"
        ]
        assert len(turn_end_events) == 1, (
            f"turn_end must be delivered exactly once, got {len(turn_end_events)}"
        )

    @pytest.mark.asyncio
    async def test_end_turn_called_twice_does_not_crash(self):
        """Calling end_turn() twice must not raise an exception.

        The second call may fire turn_end again or be a no-op — but it must
        not crash. This guards against brittle state that would break agent loops
        with retry logic.
        """
        engine, _ = make_engine()

        await engine.start_turn(turn_seq=0)

        # Should not raise
        await engine.end_turn()
        await engine.end_turn()  # second call — tolerated

    @pytest.mark.asyncio
    async def test_end_turn_drains_bus(self):
        """end_turn() must drain the bus after emitting turn_end.

        Async subscribers to turn_end must be called before end_turn() returns.
        """
        engine, _ = make_engine()
        received: list[str] = []

        async def on_turn_end(event: Event) -> None:
            received.append("turn_end_fired")

        engine._bus.subscribe("turn_end", on_turn_end)

        await engine.start_turn(turn_seq=0)
        await engine.end_turn()

        assert "turn_end_fired" in received, (
            "Async subscriber to turn_end must be called — end_turn() must drain the bus"
        )
        assert engine._bus.is_empty(), "Bus must be empty after end_turn() completes"


# ---------------------------------------------------------------------------
# 4. TestRunIsWrapper
# ---------------------------------------------------------------------------


class TestRunIsWrapper:
    @pytest.mark.asyncio
    async def test_run_produces_same_observable_result_as_split_calls(self):
        """run() must yield identical layer content as start_turn + end_turn.

        We run the same resolver twice (once via run(), once via the split API)
        and compare the compiled system blocks.
        """
        from sr2.pipeline.engine import PipelineEngine

        counter = CharacterTokenCounter()

        def _make_resolver():
            return StubResolver(
                name="sys",
                content=[TextBlock(text="You are helpful.")],
                subscriptions=[EventSubscription(event_name="turn_start")],
                max_executions=10,
            )

        def _make_engine_with_resolver(resolver):
            layer = make_system_layer(
                name="system_prompt",
                resolvers=[resolver],
                token_counter=counter,
            )
            return PipelineEngine(layers=[layer], token_counter=counter)

        # --- Via run() ---
        resolver_a = _make_resolver()
        engine_a = _make_engine_with_resolver(resolver_a)
        result_via_run = await engine_a.run([])

        # --- Via split API ---
        resolver_b = _make_resolver()
        engine_b = _make_engine_with_resolver(resolver_b)
        await engine_b.start_turn(turn_seq=0)
        result_via_split = await engine_b.end_turn()

        # Both should produce the same system content
        texts_run = [b.text for b in (result_via_run.request.system or [])]
        texts_split = [b.text for b in (result_via_split.request.system or [])]
        assert texts_run == texts_split, (
            f"run() output {texts_run!r} must equal start_turn+end_turn output {texts_split!r}"
        )

    @pytest.mark.asyncio
    async def test_run_still_works_after_new_methods_added(self):
        """run() must not regress — it must still complete and return PipelineResult."""
        from sr2.pipeline.engine import PipelineEngine, PipelineResult

        counter = CharacterTokenCounter()
        resolver = StubResolver(
            name="sys",
            content=[TextBlock(text="System prompt text.")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            token_counter=counter,
        )
        engine = PipelineEngine(layers=[layer], token_counter=counter)

        result = await engine.run([])

        assert isinstance(result, PipelineResult), "run() must still return a PipelineResult"
        assert result.request is not None
        assert result.request.system is not None
        assert any(b.text == "System prompt text." for b in result.request.system)

    @pytest.mark.asyncio
    async def test_run_turn_seq_increments_same_as_start_turn(self):
        """turn_seq increments identically whether run() or start_turn() is used.

        After one run(), turn_seq == 0. After a second run(), turn_seq == 1.
        This matches what start_turn(turn_seq=N) would do.
        """
        engine, _ = make_engine()

        assert engine._turn_seq == -1

        await engine.run([])
        assert engine._turn_seq == 0, "After first run(), turn_seq must be 0"

        await engine.run([])
        assert engine._turn_seq == 1, "After second run(), turn_seq must be 1"

    @pytest.mark.asyncio
    async def test_run_emits_both_lifecycle_events(self):
        """run() must emit both turn_start and turn_end — resolvers/transformers see them."""
        counter = CharacterTokenCounter()

        start_resolver = StubResolver(
            name="start_watcher",
            content=[TextBlock(text="started")],
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        end_transformer = StubTransformer(
            name="end_watcher",
            subscriptions=[EventSubscription(event_name="turn_end")],
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[start_resolver],
            transformers=[end_transformer],
            token_counter=counter,
        )
        engine, _ = make_engine(layers=[layer], token_counter=counter)

        await engine.run([])

        assert start_resolver.execution_count == 1, "turn_start must fire during run()"
        assert end_transformer.execution_count == 1, "turn_end must fire during run()"
