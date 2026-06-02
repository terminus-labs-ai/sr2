"""Tests for transformer execution infrastructure in Layer.process_pending().

Acceptance Criteria covered:
  AC1: Transformer with max_executions=1 fires once, skipped on subsequent same-event occurrences.
  AC2: Transformer with max_executions=2 fires twice, skipped on third occurrence.
  AC3: Transformer returning result.entries — entries appear in the provenance store after turn.
  AC4: Transformer returning result.content=[block] — layer.get_content() returns [block].
  AC5: Transformer returning result.content=None — layer content unchanged.
  AC6: Transformer returning result.events=[e] — event e is queued on the bus.
  AC7: LayerMetrics.transformer_executions for a transformer that fired twice shows {name: 2}.

Critical constraint:
  PassiveTransformer does NOT self-increment execution_count. The Layer-side
  increment (the fix being tested) is the only incrementor.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from conftest import run_engine

from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import (
    CompilationTarget,
    ResolvedContent,
    TransformationResult,
)
from sr2.pipeline.provenance import Entry, EntryOrigin, InMemoryProvenanceStore
from sr2.pipeline.token_counting import CharacterTokenCounter

# ---------------------------------------------------------------------------
# ID generation helper
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_id() -> str:
    global _COUNTER
    _COUNTER += 1
    return f"XFRM{_COUNTER:022d}"


# ---------------------------------------------------------------------------
# PassiveTransformer — does NOT self-increment execution_count
# ---------------------------------------------------------------------------


class PassiveTransformer:
    """Transformer stub that records calls without touching execution_count.

    The Layer is expected to increment execution_count after calling transform().
    This stub intentionally omits that increment so that tests verify Layer does it.
    """

    def __init__(
        self,
        name: str = "passive_transformer",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        result_content: list | None = None,
        result_events: list[Event] | None = None,
        result_entries: list[Entry] | None = None,
    ):
        self.name = name
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start", phase=EventPhase.COMPLETED)
        ]
        self.max_executions = max_executions
        self.execution_count = 0  # Layer will increment this
        self._result_content = result_content  # None means "don't replace content"
        self._result_events = result_events
        self._result_entries = result_entries or []
        self.call_count = 0  # separate tracker to verify actual call count

    async def transform(
        self, content: list, events: list[Event]
    ) -> TransformationResult:
        # Intentionally does NOT increment execution_count — Layer owns that.
        self.call_count += 1
        return TransformationResult(
            transformer_name=self.name,
            source_layer="test",
            content=self._result_content,
            events=self._result_events,
            entries=self._result_entries,
        )


# ---------------------------------------------------------------------------
# Layer factory helper
# ---------------------------------------------------------------------------


def make_layer(
    name: str = "test_layer",
    transformers: list | None = None,
    provenance_store=None,
    token_budget: int | None = None,
    bus: EventBus | None = None,
):
    from sr2.pipeline.layer import Layer

    _bus = bus or EventBus()
    kwargs = dict(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=token_budget,
        resolvers=[],
        transformers=transformers or [],
        token_counter=CharacterTokenCounter(),
        event_bus=_bus,
    )
    if provenance_store is not None:
        kwargs["provenance_store"] = provenance_store
    return Layer(**kwargs), _bus


def make_transformer_entry(
    *,
    source_ids: tuple[str, ...],
    content: TextBlock | None = None,
    transformer_name: str = "passive_transformer",
    layer: str = "test_layer",
    session_id: str = "session-test",
) -> Entry:
    """Build a valid transformer-origin Entry (must have non-empty sources)."""
    return Entry(
        id=_next_id(),
        content=content if content is not None else TextBlock(text="transformed"),
        sources=source_ids,
        origin=EntryOrigin(kind="transformer", name=transformer_name),
        layer=layer,
        session_id=session_id,
        created_at=datetime.now(tz=timezone.utc),
        meta={},
    )


def turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.COMPLETED, source_layer="engine")


# ---------------------------------------------------------------------------
# AC1 + AC2: max_executions guard
# ---------------------------------------------------------------------------


class TestTransformerMaxExecutionsGuard:
    """AC1 and AC2: execution_count is incremented by Layer; max_executions enforced."""

    @pytest.mark.asyncio
    async def test_ac1_transformer_fires_once_with_max_executions_1(self):
        """AC1: max_executions=1 transformer fires on first event, skipped on second."""
        transformer = PassiveTransformer(
            max_executions=1,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        layer, bus = make_layer(transformers=[transformer])

        # First event: should fire
        layer.handle_event(turn_start_event())
        await layer.process_pending()

        assert transformer.call_count == 1
        assert transformer.execution_count == 1

        # Second event: execution_count >= max_executions → must be skipped
        layer.handle_event(turn_start_event())
        await layer.process_pending()

        assert transformer.call_count == 1  # no additional call
        assert transformer.execution_count == 1  # not incremented again

    @pytest.mark.asyncio
    async def test_ac2_transformer_fires_twice_with_max_executions_2(self):
        """AC2: max_executions=2 transformer fires twice, is skipped on third occurrence."""
        transformer = PassiveTransformer(
            max_executions=2,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        layer, bus = make_layer(transformers=[transformer])

        # First occurrence
        layer.handle_event(turn_start_event())
        await layer.process_pending()
        assert transformer.call_count == 1
        assert transformer.execution_count == 1

        # Second occurrence: still under max
        layer.handle_event(turn_start_event())
        await layer.process_pending()
        assert transformer.call_count == 2
        assert transformer.execution_count == 2

        # Third occurrence: execution_count >= max_executions → skip
        layer.handle_event(turn_start_event())
        await layer.process_pending()
        assert transformer.call_count == 2  # no additional call
        assert transformer.execution_count == 2

    def test_execution_count_starts_at_zero(self):
        """execution_count is 0 before any events are processed."""
        transformer = PassiveTransformer(max_executions=3)
        layer, bus = make_layer(transformers=[transformer])

        assert transformer.execution_count == 0

    @pytest.mark.asyncio
    async def test_non_matching_event_does_not_increment_execution_count(self):
        """A transformer whose subscription doesn't match the event is not called."""
        transformer = PassiveTransformer(
            max_executions=5,
            subscriptions=[EventSubscription(event_name="turn_end")],
        )
        layer, bus = make_layer(transformers=[transformer])

        # Emit a turn_start — transformer subscribes to turn_end, should not fire
        layer.handle_event(turn_start_event())
        await layer.process_pending()

        assert transformer.call_count == 0
        assert transformer.execution_count == 0


# ---------------------------------------------------------------------------
# AC3: result.entries appear in provenance store after the turn
# ---------------------------------------------------------------------------


class TestTransformerEntriesInProvenanceStore:
    """AC3: result.entries returned by a transformer are flushed to the store."""

    @pytest.mark.asyncio
    async def test_ac3_single_entry_written_to_store(self):
        """AC3: One entry returned in result.entries appears in the store after process_pending."""
        store = InMemoryProvenanceStore()
        src_id = _next_id()
        entry = make_transformer_entry(source_ids=(src_id,))

        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_entries=[entry],
        )
        layer, bus = make_layer(transformers=[transformer], provenance_store=store)

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        retrieved = await store.get(entry.id)
        assert retrieved is entry

    @pytest.mark.asyncio
    async def test_ac3_multiple_entries_all_written_to_store(self):
        """AC3: Multiple entries returned by a transformer are all in the store."""
        store = InMemoryProvenanceStore()
        src_id = _next_id()
        entry1 = make_transformer_entry(source_ids=(src_id,), content=TextBlock(text="first"))
        entry2 = make_transformer_entry(source_ids=(src_id,), content=TextBlock(text="second"))

        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_entries=[entry1, entry2],
        )
        layer, bus = make_layer(transformers=[transformer], provenance_store=store)

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        assert await store.get(entry1.id) is entry1
        assert await store.get(entry2.id) is entry2

    @pytest.mark.asyncio
    async def test_ac3_transformer_with_empty_entries_does_not_grow_store(self):
        """Transformer returning empty entries does not add to entries already in store.

        Differential: on unfixed code result.entries is never flushed (existing entries
        from a resolver would be the only writes). On fixed code, the transformer's empty
        list must not corrupt or extend the store. Tests that the flush path correctly
        handles an empty list rather than writing garbage.
        """
        store = InMemoryProvenanceStore()
        src_id = _next_id()
        # Write a resolver entry via the entries= path so the store is non-empty
        resolver_entry = make_transformer_entry(
            source_ids=(src_id,),
            content=TextBlock(text="resolver output"),
            transformer_name="resolver",
        )
        await store.write_batch([resolver_entry])

        # Transformer fires but returns no entries
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_entries=[],
        )
        layer, bus = make_layer(transformers=[transformer], provenance_store=store)

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        # Store should still contain only the one pre-existing resolver entry
        all_entries = await store.get_session(resolver_entry.session_id)
        assert len(all_entries) == 1
        assert all_entries[0].id == resolver_entry.id

    @pytest.mark.asyncio
    async def test_ac3_entries_not_in_store_before_process_pending(self):
        """Entries are buffered in _pending_writes but not in the store until process_pending."""
        store = InMemoryProvenanceStore()
        src_id = _next_id()
        entry = make_transformer_entry(source_ids=(src_id,))

        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_entries=[entry],
        )
        layer, bus = make_layer(transformers=[transformer], provenance_store=store)

        # Before processing: store is empty
        assert await store.get(entry.id) is None

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        # After processing: entry is in the store
        assert await store.get(entry.id) is entry


# ---------------------------------------------------------------------------
# AC4: result.content replaces layer content
# ---------------------------------------------------------------------------


class TestTransformerContentReplacement:
    """AC4 and AC5: result.content behavior — replacement vs. no-op."""

    @pytest.mark.asyncio
    async def test_ac4_transformer_content_replaces_layer_content(self):
        """AC4: result.content=[block] → layer.get_content() returns [block] after processing."""
        replacement_block = TextBlock(text="transformed content")
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_content=[replacement_block],
        )
        layer, bus = make_layer(transformers=[transformer])

        # Pre-load some content into the layer
        layer.add_content(
            ResolvedContent(
                resolver_name="initial",
                source_layer="test",
                content=[TextBlock(text="original content")],
            )
        )

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        content = layer.get_content()
        assert len(content) == 1
        assert content[0] is replacement_block

    @pytest.mark.asyncio
    async def test_ac4_transformer_content_with_multiple_blocks(self):
        """AC4: result.content with multiple blocks replaces all layer content."""
        block1 = TextBlock(text="block one")
        block2 = TextBlock(text="block two")
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_content=[block1, block2],
        )
        layer, bus = make_layer(transformers=[transformer])

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        content = layer.get_content()
        assert len(content) == 2
        assert content[0] is block1
        assert content[1] is block2

    @pytest.mark.asyncio
    async def test_ac5_transformer_content_none_leaves_content_unchanged(self):
        """AC5: result.content=None → layer content unchanged after processing."""
        original_block = TextBlock(text="original")
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_content=None,  # explicit None — must not modify content
        )
        layer, bus = make_layer(transformers=[transformer])

        layer.add_content(
            ResolvedContent(
                resolver_name="initial",
                source_layer="test",
                content=[original_block],
            )
        )

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        content = layer.get_content()
        assert len(content) == 1
        assert content[0] is original_block

    @pytest.mark.asyncio
    async def test_ac5_content_none_does_not_clear_existing_content(self):
        """AC5: result.content=None on a layer with multiple blocks leaves all intact."""
        blocks = [TextBlock(text=f"block {i}") for i in range(3)]
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_content=None,
        )
        layer, bus = make_layer(transformers=[transformer])

        for block in blocks:
            layer.add_content(
                ResolvedContent(
                    resolver_name="r",
                    source_layer="test",
                    content=[block],
                )
            )

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        content = layer.get_content()
        assert len(content) == 3


# ---------------------------------------------------------------------------
# AC6: result.events are queued on the bus
# ---------------------------------------------------------------------------


class TestTransformerEventEmission:
    """AC6: Events returned in result.events are queued on the event bus."""

    @pytest.mark.asyncio
    async def test_ac6_transformer_emitted_event_is_queued(self):
        """AC6: An event in result.events is present on the bus after process_pending."""
        emitted_event = Event(
            name="context_updated",
            phase=EventPhase.COMPLETED,
            source_layer="passive_transformer",
        )
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_events=[emitted_event],
        )
        bus = EventBus()
        layer, _ = make_layer(transformers=[transformer], bus=bus)

        received_events: list[Event] = []
        bus.subscribe(
            EventSubscription(event_name="context_updated"),
            lambda e: received_events.append(e),
        )

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        # The event should have been queued (and sync subscribers notified)
        assert len(received_events) == 1
        assert received_events[0].name == "context_updated"

    @pytest.mark.asyncio
    async def test_ac6_multiple_emitted_events_all_queued(self):
        """AC6: Multiple events in result.events are all queued."""
        event_a = Event(
            name="signal_a", phase=EventPhase.COMPLETED, source_layer="passive_transformer"
        )
        event_b = Event(
            name="signal_b", phase=EventPhase.COMPLETED, source_layer="passive_transformer"
        )
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_events=[event_a, event_b],
        )
        bus = EventBus()
        layer, _ = make_layer(transformers=[transformer], bus=bus)

        signal_a_events: list[Event] = []
        signal_b_events: list[Event] = []
        bus.subscribe(EventSubscription(event_name="signal_a"), lambda e: signal_a_events.append(e))
        bus.subscribe(EventSubscription(event_name="signal_b"), lambda e: signal_b_events.append(e))

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        assert len(signal_a_events) == 1
        assert len(signal_b_events) == 1

    @pytest.mark.asyncio
    async def test_ac6_no_events_returned_does_not_queue_anything(self):
        """AC6: result.events=None leaves the bus empty after processing."""
        transformer = PassiveTransformer(
            subscriptions=[EventSubscription(event_name="turn_start")],
            result_events=None,
        )
        bus = EventBus()
        layer, _ = make_layer(transformers=[transformer], bus=bus)

        layer.handle_event(turn_start_event())
        await layer.process_pending()

        # Bus queue should be drained/empty (process_pending checks is_empty at end)
        assert bus.is_empty()


# ---------------------------------------------------------------------------
# AC7: LayerMetrics.transformer_executions reflects execution_count
# ---------------------------------------------------------------------------


class TestTransformerMetrics:
    """AC7: LayerMetrics.transformer_executions reflects the transformer's execution_count."""

    @pytest.mark.asyncio
    async def test_ac7_metrics_show_two_executions(self):
        """AC7: Transformer fired twice → transformer_executions[name] == 2 in metrics."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.layer import Layer

        transformer = PassiveTransformer(
            name="my_transformer",
            max_executions=2,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        bus = EventBus()
        layer = Layer(
            name="system_prompt",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[transformer],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
        )

        # Run twice: each run emits turn_start once, so transformer fires once per run
        await run_engine(engine, [])
        await run_engine(engine, [])

        # After two runs, execution_count should be 2
        assert transformer.execution_count == 2

        # Third run: max_executions reached — transformer skipped
        await run_engine(engine, [])
        assert transformer.execution_count == 2

    @pytest.mark.asyncio
    async def test_ac7_engine_metrics_transformer_executions_dict(self):
        """AC7: run_engine() result.metrics.layers includes correct transformer_executions dict."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.layer import Layer

        transformer = PassiveTransformer(
            name="my_transformer",
            max_executions=3,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )

        bus = EventBus()
        layer = Layer(
            name="system_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[transformer],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
        )

        # First run
        result = await run_engine(engine, [])
        layer_metrics = result.metrics.layers["system_layer"]
        assert layer_metrics.transformer_executions == {"my_transformer": 1}

        # Second run
        result = await run_engine(engine, [])
        layer_metrics = result.metrics.layers["system_layer"]
        assert layer_metrics.transformer_executions == {"my_transformer": 2}

    @pytest.mark.asyncio
    async def test_ac7_transformer_not_fired_absent_from_metrics(self):
        """Transformer that never fires has execution_count=0 and is absent from metrics dict."""
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.layer import Layer

        transformer = PassiveTransformer(
            name="never_fires",
            max_executions=5,
            subscriptions=[EventSubscription(event_name="turn_end")],
        )

        # turn_end is emitted in the second run_loop by the engine, but the engine
        # wires handlers to the layer's handle_event. Let's use a non-existent event
        # to ensure it never fires.
        transformer.subscriptions = [EventSubscription(event_name="nonexistent_event")]

        bus = EventBus()
        layer = Layer(
            name="system_layer",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[transformer],
            token_counter=CharacterTokenCounter(),
            event_bus=bus,
        )

        engine = PipelineEngine(
            layers=[layer],
            token_counter=CharacterTokenCounter(),
        )

        result = await run_engine(engine, [])
        layer_metrics = result.metrics.layers["system_layer"]

        # execution_count is 0 — should not appear in transformer_executions
        assert "never_fires" not in layer_metrics.transformer_executions

    @pytest.mark.asyncio
    async def test_ac7_direct_execution_count_check(self):
        """AC7 (direct): After two process_pending calls, execution_count is 2."""
        transformer = PassiveTransformer(
            name="counter_check",
            max_executions=5,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        layer, bus = make_layer(transformers=[transformer])

        layer.handle_event(turn_start_event())
        await layer.process_pending()
        assert transformer.execution_count == 1

        layer.handle_event(turn_start_event())
        await layer.process_pending()
        assert transformer.execution_count == 2
