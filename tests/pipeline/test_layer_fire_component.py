"""Tests for sr2-15 — _fire_component() helper extracted from Layer.process_pending().

These tests target the *desired* post-refactor API: Layer._fire_component().
They will FAIL against the current code because that method does not exist yet.
Once the refactor is done they document the behavioral contract of the helper.

Behavioral contract of _fire_component(comp, kind, events):

  1. Calls the correct component method (resolve / transform / provide) based on kind.
  2. Snapshots content_before and content_after when a tracer is attached.
  3. Emits a FiringRecord with status='ok' on success.
  4. Emits a FiringRecord with status='failed' and error text on exception.
  5. Re-raises exceptions after recording the failure record.
  6. FiringRecord fields (kind, component_name, layer, tokens_before/after/delta,
     duration_ms, trigger_events) are populated correctly.
  7. Content snapshots for tool_provider kind are lists of tool-def name strings
     (not ContentBlock lists).
  8. With tracer=None no snapshot work is done and no FiringRecord is emitted.
  9. _fire_component increments _next_firing_seq per call (monotonic firing_seq).

The three loops in process_pending (resolver / transformer / tool_provider) are
identical except for kind + method + get/set content — _fire_component centralises
that shared logic.  The transformer-specific extras (execution_count guard,
result.entries buffering, result.events queuing) remain on the *call site* in
process_pending, not inside _fire_component.
"""

from __future__ import annotations

import pytest

from sr2.models import TextBlock, ToolDefinition
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer, FiringRecord


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


def _make_event(name: str = "turn_start") -> Event:
    return Event(name=name, phase=EventPhase.COMPLETED, source_layer="engine")


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


class StubResolver:
    """Self-incrementing resolver that returns preset content."""

    def __init__(
        self,
        name: str = "stub_resolver",
        content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 99,
        raise_on_call: Exception | None = None,
    ):
        self.name = name
        self._content = content or [TextBlock(text="resolved")]
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._raise = raise_on_call

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        if self._raise:
            raise self._raise
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            content=self._content,
        )


class StubTransformer:
    """Transformer that does NOT self-increment execution_count (Layer owns that)."""

    def __init__(
        self,
        name: str = "stub_transformer",
        result_content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 99,
        raise_on_call: Exception | None = None,
    ):
        self.name = name
        self._result_content = result_content  # None = no-op
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._raise = raise_on_call

    async def transform(self, content: list, events: list[Event]) -> TransformationResult:
        if self._raise:
            raise self._raise
        return TransformationResult(
            transformer_name=self.name,
            source_layer="test",
            content=self._result_content,
        )


class StubToolProvider:
    """Tool provider returning a preset list of tool definitions."""

    def __init__(
        self,
        name: str = "stub_tp",
        tools: list[ToolDefinition] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 99,
        raise_on_call: Exception | None = None,
    ):
        self.name = name
        self._tools = tools or []
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._raise = raise_on_call

    async def provide(self, events: list[Event]) -> list[ToolDefinition]:
        self.execution_count += 1
        if self._raise:
            raise self._raise
        return list(self._tools)


# ---------------------------------------------------------------------------
# Layer factory
# ---------------------------------------------------------------------------


def make_layer(
    name: str = "test_layer",
    resolvers: list | None = None,
    transformers: list | None = None,
    tool_providers: list | None = None,
    tracer=None,
    bus: EventBus | None = None,
):
    from sr2.pipeline.layer import Layer

    _bus = bus or EventBus()
    layer = Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=resolvers or [],
        transformers=transformers or [],
        tool_providers=tool_providers or [],
        token_counter=CharacterTokenCounter(),
        event_bus=_bus,
        tracer=tracer,
    )
    # Wire up a simple monotonic seq counter (engine normally does this)
    seq = [0]

    def _next_seq() -> int:
        v = seq[0]
        seq[0] += 1
        return v

    layer._next_firing_seq = _next_seq
    return layer, _bus


# ---------------------------------------------------------------------------
# Helper: assert _fire_component exists
# ---------------------------------------------------------------------------


def assert_has_fire_component(layer) -> None:
    """Raise a descriptive AssertionError if _fire_component is missing."""
    assert hasattr(layer, "_fire_component"), (
        "Layer._fire_component does not exist — the DRY refactor (sr2-15) has not been applied yet. "
        "Extract the shared try/except/FiringRecord logic from process_pending into _fire_component."
    )


# ---------------------------------------------------------------------------
# 1. Method existence — fails immediately on unfactored code
# ---------------------------------------------------------------------------


class TestFireComponentExists:
    """_fire_component must exist as a method on Layer."""

    def test_layer_has_fire_component_method(self):
        """SR2-15: Layer must expose _fire_component() after the DRY refactor."""
        layer, _ = make_layer()
        assert_has_fire_component(layer)

    def test_fire_component_is_callable(self):
        """_fire_component must be callable (a coroutine function)."""
        import inspect

        layer, _ = make_layer()
        assert_has_fire_component(layer)
        assert inspect.iscoroutinefunction(layer._fire_component), (
            "_fire_component must be an async method"
        )


# ---------------------------------------------------------------------------
# 2. Resolver kind — _fire_component calls resolve()
# ---------------------------------------------------------------------------


class TestFireComponentResolver:
    """When kind='resolver', _fire_component must call comp.resolve(events)."""

    @pytest.mark.asyncio
    async def test_resolver_component_method_is_called(self):
        """_fire_component(resolver, 'resolver', ...) calls resolver.resolve()."""
        resolver = StubResolver()
        layer, _ = make_layer(resolvers=[resolver])
        assert_has_fire_component(layer)

        events = [_make_event()]
        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=events,
        )

        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_resolver_content_is_added_to_layer(self):
        """After _fire_component(resolver, ...), resolved content appears in layer._content."""
        block = TextBlock(text="from resolver")
        resolver = StubResolver(content=[block])
        layer, _ = make_layer(resolvers=[resolver])
        assert_has_fire_component(layer)

        events = [_make_event()]
        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=events,
        )

        assert block in layer._content

    @pytest.mark.asyncio
    async def test_resolver_firing_record_kind_is_resolver(self):
        """FiringRecord emitted by _fire_component(resolver) has kind='resolver'."""
        tracer = CollectingTracer()
        resolver = StubResolver(content=[TextBlock(text="data")])
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event()],
        )

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].kind == "resolver"

    @pytest.mark.asyncio
    async def test_resolver_firing_record_status_ok_on_success(self):
        """Successful resolver firing produces status='ok' record."""
        tracer = CollectingTracer()
        resolver = StubResolver(content=[TextBlock(text="ok")])
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        assert record.status == "ok"
        assert record.error is None

    @pytest.mark.asyncio
    async def test_resolver_firing_record_component_name_and_layer(self):
        """FiringRecord.component_name and .layer are set from comp.name and self.name."""
        tracer = CollectingTracer()
        resolver = StubResolver(name="my_resolver", content=[TextBlock(text="x")])
        layer, _ = make_layer(name="my_layer", resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        assert record.component_name == "my_resolver"
        assert record.layer == "my_layer"

    @pytest.mark.asyncio
    async def test_resolver_firing_record_tokens_delta_positive(self):
        """FiringRecord.tokens_delta > 0 when resolver adds content."""
        tracer = CollectingTracer()
        resolver = StubResolver(content=[TextBlock(text="abcdefgh")])  # 8 chars = 2 tokens
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        assert record.tokens_delta > 0
        assert record.tokens_after == record.tokens_before + record.tokens_delta

    @pytest.mark.asyncio
    async def test_resolver_firing_record_trigger_events_contains_event_name(self):
        """FiringRecord.trigger_events lists the triggering event names."""
        tracer = CollectingTracer()
        resolver = StubResolver(content=[TextBlock(text="x")])
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event("turn_start")],
        )

        record = tracer.get_trace()[0]
        assert "turn_start" in record.trigger_events

    @pytest.mark.asyncio
    async def test_resolver_firing_record_duration_ms_non_negative(self):
        """FiringRecord.duration_ms is a non-negative float."""
        tracer = CollectingTracer()
        resolver = StubResolver(content=[TextBlock(text="x")])
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        assert isinstance(record.duration_ms, float)
        assert record.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# 3. Transformer kind — _fire_component calls transform()
# ---------------------------------------------------------------------------


class TestFireComponentTransformer:
    """When kind='transformer', _fire_component calls comp.transform()."""

    @pytest.mark.asyncio
    async def test_transformer_transform_method_is_called(self):
        """_fire_component(transformer, 'transformer', ...) calls transformer.transform()."""

        calls = []

        class SpyTransformer(StubTransformer):
            async def transform(self, content: list, events: list[Event]) -> TransformationResult:
                calls.append(True)
                return TransformationResult(
                    transformer_name=self.name, source_layer="test", content=None
                )

        transformer = SpyTransformer()
        layer, _ = make_layer(transformers=[transformer])
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=transformer,
            kind="transformer",
            events=[_make_event()],
        )

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_transformer_firing_record_kind_is_transformer(self):
        """FiringRecord emitted by _fire_component(transformer) has kind='transformer'."""
        tracer = CollectingTracer()
        transformer = StubTransformer(result_content=[TextBlock(text="transformed")])
        layer, _ = make_layer(transformers=[transformer], tracer=tracer)
        assert_has_fire_component(layer)

        # Pre-load some content so content_before is non-empty
        layer._content = [TextBlock(text="original")]

        await layer._fire_component(
            comp=transformer,
            kind="transformer",
            events=[_make_event()],
        )

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].kind == "transformer"

    @pytest.mark.asyncio
    async def test_transformer_content_before_reflects_layer_state_before_call(self):
        """FiringRecord.content_before is snapshotted before transform() is called."""
        tracer = CollectingTracer()
        original = TextBlock(text="before transform")
        transformer = StubTransformer(result_content=[TextBlock(text="after")])
        layer, _ = make_layer(transformers=[transformer], tracer=tracer)
        assert_has_fire_component(layer)

        layer._content = [original]

        await layer._fire_component(
            comp=transformer,
            kind="transformer",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        assert original in record.content_before


# ---------------------------------------------------------------------------
# 4. Tool-provider kind — _fire_component calls provide()
# ---------------------------------------------------------------------------


class TestFireComponentToolProvider:
    """When kind='tool_provider', _fire_component calls comp.provide()."""

    @pytest.mark.asyncio
    async def test_tool_provider_provide_method_is_called(self):
        """_fire_component(tp, 'tool_provider', ...) calls tp.provide()."""
        tp = StubToolProvider(tools=[_make_tool("search")])
        layer, _ = make_layer(tool_providers=[tp])
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=tp,
            kind="tool_provider",
            events=[_make_event()],
        )

        assert tp.execution_count == 1

    @pytest.mark.asyncio
    async def test_tool_provider_tools_are_added_to_layer(self):
        """After _fire_component(tp, ...), tool definitions appear in layer._tool_definitions."""
        tp = StubToolProvider(tools=[_make_tool("calculate"), _make_tool("search")])
        layer, _ = make_layer(tool_providers=[tp])
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=tp,
            kind="tool_provider",
            events=[_make_event()],
        )

        names = [td.name for td in layer._tool_definitions]
        assert "calculate" in names
        assert "search" in names

    @pytest.mark.asyncio
    async def test_tool_provider_firing_record_kind_is_tool_provider(self):
        """FiringRecord emitted by _fire_component(tp) has kind='tool_provider'."""
        tracer = CollectingTracer()
        tp = StubToolProvider(tools=[_make_tool("my_tool")])
        layer, _ = make_layer(tool_providers=[tp], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=tp,
            kind="tool_provider",
            events=[_make_event()],
        )

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].kind == "tool_provider"

    @pytest.mark.asyncio
    async def test_tool_provider_content_fields_are_string_lists(self):
        """Tool-provider content_before and content_after are lists of tool-def name strings."""
        tracer = CollectingTracer()
        tp = StubToolProvider(tools=[_make_tool("alpha"), _make_tool("beta")])
        layer, _ = make_layer(tool_providers=[tp], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=tp,
            kind="tool_provider",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        # Both fields must be lists of strings — not ContentBlock lists
        assert all(isinstance(n, str) for n in record.content_before)
        assert all(isinstance(n, str) for n in record.content_after)
        assert "alpha" in record.content_after
        assert "beta" in record.content_after

    @pytest.mark.asyncio
    async def test_tool_provider_tokens_are_always_zero(self):
        """Tool-provider records have tokens_before=0, tokens_after=0, tokens_delta=0."""
        tracer = CollectingTracer()
        tp = StubToolProvider(tools=[_make_tool("x")])
        layer, _ = make_layer(tool_providers=[tp], tracer=tracer)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=tp,
            kind="tool_provider",
            events=[_make_event()],
        )

        record = tracer.get_trace()[0]
        assert record.tokens_before == 0
        assert record.tokens_after == 0
        assert record.tokens_delta == 0


# ---------------------------------------------------------------------------
# 5. Failure path — shared across all three kinds
# ---------------------------------------------------------------------------


class TestFireComponentFailurePath:
    """On exception: FiringRecord with status='failed' is emitted and exception re-raises."""

    @pytest.mark.asyncio
    async def test_resolver_failure_emits_failed_record(self):
        """Failing resolver -> status='failed' record + error text."""
        tracer = CollectingTracer()
        resolver = StubResolver(raise_on_call=RuntimeError("resolver boom"))
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        with pytest.raises(RuntimeError, match="resolver boom"):
            await layer._fire_component(
                comp=resolver,
                kind="resolver",
                events=[_make_event()],
            )

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].status == "failed"
        assert "resolver boom" in (records[0].error or "")

    @pytest.mark.asyncio
    async def test_transformer_failure_emits_failed_record(self):
        """Failing transformer -> status='failed' record + error text."""
        tracer = CollectingTracer()
        transformer = StubTransformer(raise_on_call=ValueError("transformer boom"))
        layer, _ = make_layer(transformers=[transformer], tracer=tracer)
        assert_has_fire_component(layer)

        with pytest.raises(ValueError, match="transformer boom"):
            await layer._fire_component(
                comp=transformer,
                kind="transformer",
                events=[_make_event()],
            )

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].status == "failed"
        assert "transformer boom" in (records[0].error or "")

    @pytest.mark.asyncio
    async def test_tool_provider_failure_emits_failed_record(self):
        """Failing tool provider -> status='failed' record + error text."""
        tracer = CollectingTracer()
        tp = StubToolProvider(raise_on_call=IOError("tp boom"))
        layer, _ = make_layer(tool_providers=[tp], tracer=tracer)
        assert_has_fire_component(layer)

        with pytest.raises(IOError, match="tp boom"):
            await layer._fire_component(
                comp=tp,
                kind="tool_provider",
                events=[_make_event()],
            )

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].status == "failed"
        assert "tp boom" in (records[0].error or "")

    @pytest.mark.asyncio
    async def test_failure_record_includes_component_name(self):
        """Failed FiringRecord still has the correct component_name."""
        tracer = CollectingTracer()
        resolver = StubResolver(name="named_resolver", raise_on_call=RuntimeError("err"))
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)
        assert_has_fire_component(layer)

        with pytest.raises(RuntimeError):
            await layer._fire_component(
                comp=resolver,
                kind="resolver",
                events=[_make_event()],
            )

        record = tracer.get_trace()[0]
        assert record.component_name == "named_resolver"

    @pytest.mark.asyncio
    async def test_exception_propagates_even_when_tracer_is_none(self):
        """Exception re-raises even when there is no tracer (no FiringRecord produced)."""
        resolver = StubResolver(raise_on_call=RuntimeError("propagate me"))
        layer, _ = make_layer(resolvers=[resolver], tracer=None)
        assert_has_fire_component(layer)

        with pytest.raises(RuntimeError, match="propagate me"):
            await layer._fire_component(
                comp=resolver,
                kind="resolver",
                events=[_make_event()],
            )


# ---------------------------------------------------------------------------
# 6. Tracer=None — no instrumentation overhead
# ---------------------------------------------------------------------------


class TestFireComponentNoTracer:
    """With tracer=None, _fire_component does no snapshot work and emits no records."""

    @pytest.mark.asyncio
    async def test_resolver_no_tracer_content_still_added(self):
        """With tracer=None, resolver still populates layer content."""
        block = TextBlock(text="resolved data")
        resolver = StubResolver(content=[block])
        layer, _ = make_layer(resolvers=[resolver], tracer=None)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=resolver,
            kind="resolver",
            events=[_make_event()],
        )

        assert block in layer._content

    @pytest.mark.asyncio
    async def test_tool_provider_no_tracer_tools_still_added(self):
        """With tracer=None, tool provider still populates layer._tool_definitions."""
        tp = StubToolProvider(tools=[_make_tool("no_tracer_tool")])
        layer, _ = make_layer(tool_providers=[tp], tracer=None)
        assert_has_fire_component(layer)

        await layer._fire_component(
            comp=tp,
            kind="tool_provider",
            events=[_make_event()],
        )

        names = [td.name for td in layer._tool_definitions]
        assert "no_tracer_tool" in names


# ---------------------------------------------------------------------------
# 7. firing_seq increments monotonically across multiple _fire_component calls
# ---------------------------------------------------------------------------


class TestFireComponentFiringSeq:
    """Each call to _fire_component advances the firing_seq counter."""

    @pytest.mark.asyncio
    async def test_two_calls_produce_sequential_firing_seqs(self):
        """Back-to-back _fire_component calls get firing_seq 0 and 1."""
        tracer = CollectingTracer()
        resolver_a = StubResolver(name="r_a", content=[TextBlock(text="a")])
        resolver_b = StubResolver(name="r_b", content=[TextBlock(text="b")])
        layer, _ = make_layer(resolvers=[resolver_a, resolver_b], tracer=tracer)
        assert_has_fire_component(layer)

        events = [_make_event()]
        await layer._fire_component(comp=resolver_a, kind="resolver", events=events)
        await layer._fire_component(comp=resolver_b, kind="resolver", events=events)

        records = tracer.get_trace()
        assert len(records) == 2
        seqs = sorted(r.firing_seq for r in records)
        assert seqs[1] == seqs[0] + 1

    @pytest.mark.asyncio
    async def test_cross_kind_firing_seqs_are_monotonic(self):
        """firing_seq is monotonic across resolver, transformer, and tool_provider kinds."""
        tracer = CollectingTracer()
        resolver = StubResolver(name="r", content=[TextBlock(text="x")])
        transformer = StubTransformer(name="t", result_content=None)
        tp = StubToolProvider(name="tp", tools=[_make_tool("y")])
        layer, _ = make_layer(
            resolvers=[resolver],
            transformers=[transformer],
            tool_providers=[tp],
            tracer=tracer,
        )
        assert_has_fire_component(layer)

        events = [_make_event()]
        await layer._fire_component(comp=resolver, kind="resolver", events=events)
        await layer._fire_component(comp=transformer, kind="transformer", events=events)
        await layer._fire_component(comp=tp, kind="tool_provider", events=events)

        records = tracer.get_trace()
        assert len(records) == 3
        seqs = [r.firing_seq for r in records]
        # Must be strictly increasing
        assert seqs == sorted(seqs)
        assert seqs == list(range(seqs[0], seqs[0] + 3))
