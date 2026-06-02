"""Tests for sr2-59: Layer.wire() public method and event_bus optional default.

Covers behavioral requirements for the fix to PipelineEngine._wire_layers():
  BR1: After PipelineEngine init, all layers share the engine's event bus.
  BR2: After PipelineEngine init, all layers share the engine's provenance store.
  BR3: After PipelineEngine init, all layers use the engine's tracer.
  BR4: Layer can be constructed without event_bus (optional, defaults to None).
  BR5: Layer exposes a public wire(bus, provenance_store, tracer) method.
  BR6: After layer.wire(new_bus, ...), events fired on new_bus reach the layer's handlers.

All assertions are via observable behavior — no internal attribute inspection
except where behavior is only verifiable through state (e.g. tracer identity).
"""

from __future__ import annotations

import pytest

from conftest import run_engine

from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.provenance import (
    Entry,
    EntryOrigin,
    InMemoryProvenanceStore,
)
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer
from sr2.models import TextBlock
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers — shared factories
# ---------------------------------------------------------------------------


def _make_layer(name: str = "test_layer", event_bus=None):
    """Build a minimal Layer. event_bus is omitted when None (tests BR4)."""
    from sr2.pipeline.layer import Layer

    kwargs = dict(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
    )
    if event_bus is not None:
        kwargs["event_bus"] = event_bus
    return Layer(**kwargs)


def _make_engine(layers=None, tracer=None, provenance_store=None, bus=None):
    from sr2.pipeline.engine import PipelineEngine

    kwargs = dict(
        layers=layers if layers is not None else [],
        token_counter=CharacterTokenCounter(),
    )
    if tracer is not None:
        kwargs["tracer"] = tracer
    if provenance_store is not None:
        kwargs["provenance_store"] = provenance_store
    if bus is not None:
        kwargs["bus"] = bus
    return PipelineEngine(**kwargs)


def _make_entry(text: str = "test") -> Entry:
    return Entry(
        id=f"PROV{text[:22]:0<22}",
        content=TextBlock(text=text),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="test_resolver"),
        layer="test_layer",
        session_id="session-test",
        created_at=datetime.now(tz=timezone.utc),
    )


class CapturingResolver:
    """Resolver that records every event batch it receives."""

    def __init__(
        self,
        name: str = "capturing_resolver",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 10,
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self.captured_events: list[list[Event]] = []

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        self.captured_events.append(list(events))
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            content=[TextBlock(text=f"resolved by {self.name}")],
        )


class EntryResolver:
    """Resolver that returns a fixed Entry via the entries= path."""

    def __init__(
        self,
        name: str = "entry_resolver",
        entry: Entry | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._entry = entry or _make_entry()
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self.max_executions = max_executions
        self.execution_count = 0

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            entries=[self._entry],
        )


# ---------------------------------------------------------------------------
# BR4 — Layer event_bus is optional (default None, no TypeError)
# ---------------------------------------------------------------------------


class TestLayerEventBusOptional:
    def test_layer_constructs_without_event_bus(self):
        """BR4: Layer can be constructed without passing event_bus — no TypeError raised."""
        from sr2.pipeline.layer import Layer

        # Should not raise
        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        assert layer is not None

    def test_layer_without_event_bus_can_be_wired_by_engine(self):
        """BR4: Layer constructed without event_bus can subsequently be wired by the engine.

        Verifies that the layer's initial bus state does not prevent wiring —
        the engine must be able to call layer.wire() after constructing the layer.
        This is purely behavioral: does the wire() call succeed without error?
        """
        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        bus = EventBus()
        store = InMemoryProvenanceStore()
        # Should not raise — the engine must be able to wire a layer that was
        # constructed without an event_bus argument.
        layer.wire(bus, store, None)


# ---------------------------------------------------------------------------
# BR5 — Layer.wire() public method exists and is callable
# ---------------------------------------------------------------------------


class TestLayerWireMethodExists:
    def test_layer_has_wire_method(self):
        """BR5: Layer exposes a public wire() method."""
        layer = _make_layer(event_bus=EventBus())
        assert hasattr(layer, "wire"), "Layer must expose a wire() method"
        assert callable(layer.wire), "Layer.wire must be callable"

    def test_wire_accepts_bus_store_tracer(self):
        """BR5: wire(bus, provenance_store, tracer) signature accepted without error."""
        layer = _make_layer(event_bus=EventBus())
        bus = EventBus()
        store = InMemoryProvenanceStore()
        tracer = CollectingTracer()

        # Should not raise
        layer.wire(bus, store, tracer)

    def test_wire_accepts_none_tracer(self):
        """BR5: wire(bus, store, tracer=None) is accepted — tracer is optional at wiring time."""
        layer = _make_layer(event_bus=EventBus())
        bus = EventBus()
        store = InMemoryProvenanceStore()

        # Should not raise with tracer=None
        layer.wire(bus, store, None)


# ---------------------------------------------------------------------------
# BR6 — wire() connects new bus so events reach the layer's handlers
# ---------------------------------------------------------------------------


class TestLayerWireConnectsBus:
    @pytest.mark.asyncio
    async def test_events_on_wired_bus_reach_layer_handlers(self):
        """BR6: After layer.wire(new_bus, ...), events fired on new_bus reach the layer's resolver."""
        captured_events: list[Event] = []

        resolver = CapturingResolver(
            subscriptions=[EventSubscription(event_name="custom_event")],
            max_executions=5,
        )

        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="wired_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )

        new_bus = EventBus()
        store = InMemoryProvenanceStore()
        layer.wire(new_bus, store, None)

        # Subscribe the layer's handler to the bus (as the engine does)
        for sub in layer.subscriptions:
            new_bus.subscribe(sub, layer.handle_event)

        # Fire an event on the new bus — layer must collect it
        await new_bus.emit(
            Event(name="custom_event", phase=EventPhase.COMPLETED, source_layer="test")
        )

        # The layer's pending events should include the custom_event
        assert any(e.name == "custom_event" for e in layer._pending_events), (
            "After wire(new_bus, ...) and subscription, events on new_bus must reach the layer"
        )

    @pytest.mark.asyncio
    async def test_events_on_old_bus_do_not_reach_rewired_layer(self):
        """BR6: After wire(new_bus, ...), events on the OLD bus no longer reach this layer.

        We first subscribe the layer's handle_event to old_bus (simulating what the
        engine does during initial wiring). We then wire() to new_bus, which must
        unsubscribe from old_bus. Events emitted on old_bus must NOT add to
        layer._pending_events after the rewire.
        """
        old_bus = EventBus()
        new_bus = EventBus()
        store = InMemoryProvenanceStore()

        sub = EventSubscription(event_name="old_event")

        from sr2.pipeline.layer import Layer

        resolver = CapturingResolver(
            subscriptions=[sub],
            max_executions=5,
        )

        layer = Layer(
            name="rewired_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )

        # Simulate engine initial wiring to old_bus
        layer.wire(old_bus, store, None)
        old_bus.subscribe(sub, layer.handle_event)

        # Now rewire to new_bus — must drop old_bus subscriptions
        layer.wire(new_bus, store, None)

        # Emit on old_bus — must NOT reach the layer after rewire
        await old_bus.emit(
            Event(name="old_event", phase=EventPhase.COMPLETED, source_layer="test")
        )

        assert not any(e.name == "old_event" for e in layer._pending_events), (
            "After wire(new_bus, ...) replaces old_bus, events on old_bus must NOT "
            "reach the layer — wire() must unsubscribe from the previous bus"
        )


# ---------------------------------------------------------------------------
# BR1 — PipelineEngine: all layers share the engine's event bus
# ---------------------------------------------------------------------------


class TestEngineWiredLayersShareBus:
    @pytest.mark.asyncio
    async def test_wired_layers_share_engine_bus(self):
        """BR1: After PipelineEngine init, an event fired on the engine bus reaches all subscribed layers.

        Observable test: layer resolvers subscribed to turn_start fire during start_turn(),
        which is driven by the engine's bus. If both resolvers fire, both layers share the bus.
        """
        counter = CharacterTokenCounter()
        resolver_a = CapturingResolver(
            name="resolver_a",
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        resolver_b = CapturingResolver(
            name="resolver_b",
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        from sr2.pipeline.layer import Layer

        layer_a = Layer(
            name="layer_a",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver_a],
            transformers=[],
            token_counter=counter,
        )
        layer_b = Layer(
            name="layer_b",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver_b],
            transformers=[],
            token_counter=counter,
        )

        engine = _make_engine(layers=[layer_a, layer_b])
        await engine.start_turn(turn_seq=0)

        assert resolver_a.execution_count >= 1, (
            "Resolver in layer_a must fire — it should be subscribed to the shared engine bus"
        )
        assert resolver_b.execution_count >= 1, (
            "Resolver in layer_b must fire — it should be subscribed to the shared engine bus"
        )

    @pytest.mark.asyncio
    async def test_event_fired_by_one_layer_reaches_another_via_shared_bus(self):
        """BR1: An event queued by one layer's resolver is received by another layer's resolver.

        This proves both layers are on the same bus: layer_a queues a custom event via
        its result, and layer_b (subscribed to that event) must fire in response.
        """
        counter = CharacterTokenCounter()

        # layer_a fires turn_start → queues "relay_event"
        class RelayResolver:
            name = "relay_resolver"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def resolve(self, events):
                self.execution_count += 1
                return ResolvedContent(
                    resolver_name=self.name,
                    source_layer="layer_a",
                    content=[TextBlock(text="relay")],
                    events=[
                        Event(
                            name="relay_event",
                            phase=EventPhase.COMPLETED,
                            source_layer="layer_a",
                        )
                    ],
                )

        relay = RelayResolver()

        receiver = CapturingResolver(
            name="receiver",
            subscriptions=[EventSubscription(event_name="relay_event")],
        )

        from sr2.pipeline.layer import Layer

        layer_a = Layer(
            name="layer_a",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[relay],
            transformers=[],
            token_counter=counter,
        )
        layer_b = Layer(
            name="layer_b",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[receiver],
            transformers=[],
            token_counter=counter,
        )

        engine = _make_engine(layers=[layer_a, layer_b])
        await engine.start_turn(turn_seq=0)

        assert receiver.execution_count >= 1, (
            "Receiver in layer_b must fire in response to relay_event queued by layer_a — "
            "both layers must share the same event bus"
        )


# ---------------------------------------------------------------------------
# BR2 — PipelineEngine: all layers share the engine's provenance store
# ---------------------------------------------------------------------------


class TestEngineWiredLayersShareProvenanceStore:
    @pytest.mark.asyncio
    async def test_wired_layers_use_engine_provenance_store(self):
        """BR2: After engine init, writes by any layer resolver appear in the engine's store.

        Observable test: run the engine with an entry-returning resolver. The entry
        must be retrievable from the engine's provenance store after the turn.
        """
        counter = CharacterTokenCounter()
        store = InMemoryProvenanceStore()
        entry = _make_entry("hello")

        resolver = EntryResolver(entry=entry)

        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="layer_a",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=counter,
        )

        engine = _make_engine(layers=[layer], provenance_store=store)
        await run_engine(engine, [])

        retrieved = await store.get(entry.id)
        assert retrieved is entry, (
            "Entry written by layer_a's resolver must be retrievable from the engine's "
            "provenance_store — the layer must share the engine's store"
        )

    @pytest.mark.asyncio
    async def test_two_layers_write_to_same_provenance_store(self):
        """BR2: Entries written by different layers both appear in the engine's shared store."""
        counter = CharacterTokenCounter()
        store = InMemoryProvenanceStore()
        entry_a = _make_entry("from_layer_a")
        entry_b = _make_entry("from_layer_b")

        from sr2.pipeline.layer import Layer

        layer_a = Layer(
            name="layer_a",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[EntryResolver(name="resolver_a", entry=entry_a)],
            transformers=[],
            token_counter=counter,
        )
        layer_b = Layer(
            name="layer_b",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[EntryResolver(name="resolver_b", entry=entry_b)],
            transformers=[],
            token_counter=counter,
        )

        engine = _make_engine(layers=[layer_a, layer_b], provenance_store=store)
        await run_engine(engine, [])

        retrieved_a = await store.get(entry_a.id)
        retrieved_b = await store.get(entry_b.id)

        assert retrieved_a is entry_a, (
            "Entry from layer_a must be in the shared provenance store"
        )
        assert retrieved_b is entry_b, (
            "Entry from layer_b must be in the shared provenance store"
        )


# ---------------------------------------------------------------------------
# BR3 — PipelineEngine: all layers use the engine's tracer
# ---------------------------------------------------------------------------


class TestEngineWiredLayersUseTracer:
    @pytest.mark.asyncio
    async def test_wired_layers_use_engine_tracer(self):
        """BR3: After engine init with a tracer, running a turn produces FiringRecords in that tracer.

        Proves the engine's tracer is the one layers use when processing events.
        """
        counter = CharacterTokenCounter()
        tracer = CollectingTracer()

        resolver = CapturingResolver(
            name="traced_resolver",
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        from sr2.pipeline.layer import Layer

        layer = Layer(
            name="traced_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=counter,
        )

        engine = _make_engine(layers=[layer], tracer=tracer)
        await engine.start_turn(turn_seq=0)

        records = tracer.get_trace()
        assert records, (
            "CollectingTracer must have received FiringRecords after start_turn() — "
            "the engine's tracer must be the one used by wired layers"
        )

    @pytest.mark.asyncio
    async def test_tracer_records_come_from_all_layers(self):
        """BR3: FiringRecords in the engine's tracer include entries from multiple layers."""
        counter = CharacterTokenCounter()
        tracer = CollectingTracer()

        from sr2.pipeline.layer import Layer

        layer_a = Layer(
            name="layer_alpha",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[
                CapturingResolver(
                    name="resolver_alpha",
                    subscriptions=[EventSubscription(event_name="turn_start")],
                )
            ],
            transformers=[],
            token_counter=counter,
        )
        layer_b = Layer(
            name="layer_beta",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[
                CapturingResolver(
                    name="resolver_beta",
                    subscriptions=[EventSubscription(event_name="turn_start")],
                )
            ],
            transformers=[],
            token_counter=counter,
        )

        engine = _make_engine(layers=[layer_a, layer_b], tracer=tracer)
        await engine.start_turn(turn_seq=0)

        layer_names_in_trace = {r.layer for r in tracer.get_trace()}
        assert "layer_alpha" in layer_names_in_trace, (
            "FiringRecords from layer_alpha must appear in the tracer"
        )
        assert "layer_beta" in layer_names_in_trace, (
            "FiringRecords from layer_beta must appear in the tracer"
        )

    def test_engine_tracer_is_threaded_to_layers_on_init(self):
        """BR3: The tracer passed to the engine is the same object used by each layer.

        This directly verifies the wiring contract — the engine must set each layer's
        tracer to the engine's own tracer instance, not a copy.
        """
        tracer = CollectingTracer()
        layers = [
            _make_layer("layer_x"),
            _make_layer("layer_y"),
        ]

        engine = _make_engine(layers=layers, tracer=tracer)

        for layer in engine._layers:
            assert layer._tracer is tracer, (
                f"Layer '{layer.name}' must use the engine's tracer instance after wiring"
            )
