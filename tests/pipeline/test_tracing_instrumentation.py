"""Tests for FR4-7 / FR10 — Instrumentation of firing sites in Layer.process_pending.

Covers:
  FR4:  Resolver firing emits a FiringRecord via the tracer
  FR5:  Transformer firing emits a FiringRecord via the tracer
  FR6:  Tool-provider firing emits a FiringRecord via the tracer
  FR7:  Failed component captures status="failed" and re-raises
  FR10: With tracer=None, no instrumentation overhead; process_pending works unchanged
"""

import pytest

from sr2.models import ContentBlock, TextBlock, ToolDefinition
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer, FiringRecord


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubResolver:
    """Resolver that returns predetermined text content."""

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


class FailingResolver:
    """Resolver that always raises an exception."""

    def __init__(
        self,
        name: str = "failing_resolver",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        source_layer: str = "test",
        error_message: str = "intentional resolver failure",
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._source_layer = source_layer
        self._error_message = error_message

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        raise RuntimeError(self._error_message)


class StubTransformer:
    """Transformer that optionally applies a transform function."""

    def __init__(
        self,
        name: str = "stub_transformer",
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
        transform_fn=None,
    ):
        self.name = name
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0
        self._transform_fn = transform_fn

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult:
        self.execution_count += 1
        if self._transform_fn:
            result_content = self._transform_fn(content)
        else:
            result_content = content
        return TransformationResult(
            transformer_name=self.name,
            source_layer="test",
            content=result_content,
            events=None,
        )


class StubToolProvider:
    """Tool provider that returns a predetermined list of ToolDefinitions."""

    def __init__(
        self,
        name: str = "stub_tool_provider",
        tools: list[ToolDefinition] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._tools = tools or []
        self.subscriptions = subscriptions or []
        self.max_executions = max_executions
        self.execution_count = 0

    async def provide(self, events: list[Event]) -> list[ToolDefinition]:
        self.execution_count += 1
        return list(self._tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TURN_START_SUB = [EventSubscription(event_name="turn_start")]
_TURN_END_SUB = [EventSubscription(event_name="turn_end")]


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


def make_system_layer(
    name: str = "system_prompt",
    resolvers: list | None = None,
    transformers: list | None = None,
    tool_providers: list | None = None,
    tracer=None,
    event_bus: EventBus | None = None,
):
    """Build a SYSTEM-target Layer with a CharacterTokenCounter."""
    from sr2.pipeline.layer import Layer

    return Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=resolvers or [],
        transformers=transformers or [],
        tool_providers=tool_providers or [],
        token_counter=CharacterTokenCounter(),
        event_bus=event_bus or EventBus(),
        tracer=tracer,
    )


def build_engine(layers: list, tracer=None):
    """Build a PipelineEngine, wiring in the tracer if provided."""
    from sr2.pipeline.engine import PipelineEngine

    return PipelineEngine(
        layers=layers,
        token_counter=CharacterTokenCounter(),
        tracer=tracer,
    )


# ---------------------------------------------------------------------------
# FR4 — Resolver instrumentation
# ---------------------------------------------------------------------------


class TestResolverInstrumentation:
    @pytest.mark.asyncio
    async def test_resolver_firing_produces_exactly_one_record(self):
        """FR4: A resolver that fires produces exactly one FiringRecord with kind='resolver'."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="You are helpful.")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        records = tracer.get_trace()
        resolver_records = [r for r in records if r.kind == "resolver"]
        assert len(resolver_records) == 1

    @pytest.mark.asyncio
    async def test_resolver_record_has_correct_component_and_layer_name(self):
        """FR4: Record has correct component_name and layer."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="my_resolver",
            content=[TextBlock(text="content")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="my_layer", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = tracer.get_trace()[0]
        assert record.component_name == "my_resolver"
        assert record.layer == "my_layer"

    @pytest.mark.asyncio
    async def test_resolver_content_before_is_empty_on_fresh_layer(self):
        """FR4: content_before is empty (no content on a fresh layer before resolver fires)."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="Hello.")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert record.content_before == []

    @pytest.mark.asyncio
    async def test_resolver_content_after_is_non_empty(self):
        """FR4: content_after contains the text added by the resolver."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="Resolver added this.")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert len(record.content_after) > 0

    @pytest.mark.asyncio
    async def test_resolver_tokens_delta_positive(self):
        """FR4: tokens_delta > 0 when resolver adds content."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="abcdefgh")],  # 8 chars = 2 tokens
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert record.tokens_delta > 0

    @pytest.mark.asyncio
    async def test_resolver_tokens_before_and_after_correct(self):
        """FR4: tokens_before == 0 on fresh layer; tokens_after == tokens_delta when starting from empty."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="abcdefgh")],  # 8 chars = 2 tokens with CharacterTokenCounter
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert record.tokens_before == 0  # fresh layer, nothing before
        assert record.tokens_after > 0    # content was added
        assert record.tokens_after == record.tokens_before + record.tokens_delta

    @pytest.mark.asyncio
    async def test_resolver_duration_ms_non_negative(self):
        """FR4: duration_ms is a non-negative float."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="content")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert isinstance(record.duration_ms, float)
        assert record.duration_ms >= 0.0

    @pytest.mark.asyncio
    async def test_resolver_status_ok(self):
        """FR4: status is 'ok' for a successful resolver firing."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="data")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert record.status == "ok"

    @pytest.mark.asyncio
    async def test_resolver_turn_seq_increments_across_runs(self):
        """FR4: turn_seq == 0 for first run, == 1 for second run."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="text")],
            subscriptions=_TURN_START_SUB,
            max_executions=99,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])
        await engine.run([])

        records = [r for r in tracer.get_trace() if r.kind == "resolver"]
        assert len(records) == 2
        assert records[0].turn_seq == 0
        assert records[1].turn_seq == 1

    @pytest.mark.asyncio
    async def test_resolver_first_in_turn_has_firing_seq_zero(self):
        """FR4: firing_seq for the first resolver in a turn is 0."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="first_resolver",
            content=[TextBlock(text="first")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert record.firing_seq == 0

    @pytest.mark.asyncio
    async def test_two_resolvers_in_same_turn_have_monotonic_firing_seq(self):
        """FR4: Two resolvers in the same turn have firing_seq 0 and 1 (no gaps)."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver_a = StubResolver(
            name="resolver_a",
            content=[TextBlock(text="first resolver output")],
            subscriptions=_TURN_START_SUB,
        )
        resolver_b = StubResolver(
            name="resolver_b",
            content=[TextBlock(text="second resolver output")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver_a, resolver_b],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        resolver_records = sorted(
            [r for r in tracer.get_trace() if r.kind == "resolver"],
            key=lambda r: r.firing_seq,
        )
        assert len(resolver_records) == 2
        assert resolver_records[0].firing_seq == 0
        assert resolver_records[1].firing_seq == 1

    @pytest.mark.asyncio
    async def test_resolver_trigger_events_non_empty(self):
        """FR4: trigger_events is non-empty and contains the event name that triggered the resolver."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="data")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "resolver")
        assert len(record.trigger_events) > 0
        assert "turn_start" in record.trigger_events


# ---------------------------------------------------------------------------
# FR5 — Transformer instrumentation
# ---------------------------------------------------------------------------


class TestTransformerInstrumentation:
    @pytest.mark.asyncio
    async def test_transformer_firing_produces_one_record(self):
        """FR5: A transformer that fires produces exactly one FiringRecord with kind='transformer'."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="original content")],
            subscriptions=_TURN_START_SUB,
        )
        transformer = StubTransformer(
            name="my_transformer",
            subscriptions=_TURN_END_SUB,
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            transformers=[transformer],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        transformer_records = [r for r in tracer.get_trace() if r.kind == "transformer"]
        assert len(transformer_records) == 1

    @pytest.mark.asyncio
    async def test_transformer_content_before_reflects_pre_transform_state(self):
        """FR5: content_before matches layer content before transform (non-empty after resolver ran)."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="pre-transform content")],
            subscriptions=_TURN_START_SUB,
        )
        transformer = StubTransformer(
            name="my_transformer",
            subscriptions=_TURN_END_SUB,
            transform_fn=lambda content: [TextBlock(text="transformed content")],
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            transformers=[transformer],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        transformer_record = next(r for r in tracer.get_trace() if r.kind == "transformer")
        # content_before should be non-empty (resolver already ran)
        assert len(transformer_record.content_before) > 0

    @pytest.mark.asyncio
    async def test_transformer_content_after_differs_when_transform_changes_content(self):
        """FR5: content_after differs from content_before when transformer changes content."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="original")],
            subscriptions=_TURN_START_SUB,
        )
        transformer = StubTransformer(
            name="replacer",
            subscriptions=_TURN_END_SUB,
            transform_fn=lambda content: [TextBlock(text="replaced")],
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            transformers=[transformer],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        transformer_record = next(r for r in tracer.get_trace() if r.kind == "transformer")
        assert transformer_record.content_before != transformer_record.content_after


# ---------------------------------------------------------------------------
# FR6 — Tool-provider instrumentation
# ---------------------------------------------------------------------------


class TestToolProviderInstrumentation:
    @pytest.mark.asyncio
    async def test_tool_provider_firing_produces_one_record(self):
        """FR6: A tool provider that fires produces one FiringRecord with kind='tool_provider'."""
        tracer = CollectingTracer()
        bus = EventBus()
        tool_provider = StubToolProvider(
            name="my_tools",
            tools=[_make_tool("search"), _make_tool("calculate")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(
            name="system_prompt",
            tool_providers=[tool_provider],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        tool_records = [r for r in tracer.get_trace() if r.kind == "tool_provider"]
        assert len(tool_records) == 1

    @pytest.mark.asyncio
    async def test_tool_provider_content_fields_are_tool_def_name_lists(self):
        """FR6: content_before and content_after are lists of tool-def names (strings)."""
        tracer = CollectingTracer()
        bus = EventBus()
        tool_provider = StubToolProvider(
            name="my_tools",
            tools=[_make_tool("search"), _make_tool("calculate")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(
            name="system_prompt",
            tool_providers=[tool_provider],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "tool_provider")
        # content_before: names present before provide() ran
        assert isinstance(record.content_before, list)
        assert all(isinstance(n, str) for n in record.content_before)
        # content_after: names after add_tool_definitions()
        assert isinstance(record.content_after, list)
        assert all(isinstance(n, str) for n in record.content_after)
        # After should include the tools that were just provided
        assert "search" in record.content_after
        assert "calculate" in record.content_after

    @pytest.mark.asyncio
    async def test_tool_provider_tokens_delta_is_zero(self):
        """FR6: tokens_delta == 0 for tool-provider records (MVP — no token counting for tools)."""
        tracer = CollectingTracer()
        bus = EventBus()
        tool_provider = StubToolProvider(
            name="my_tools",
            tools=[_make_tool("search")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(
            name="system_prompt",
            tool_providers=[tool_provider],
            tracer=tracer,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=tracer)

        await engine.run([])

        record = next(r for r in tracer.get_trace() if r.kind == "tool_provider")
        assert record.tokens_before == 0
        assert record.tokens_after == 0
        assert record.tokens_delta == 0


# ---------------------------------------------------------------------------
# FR7 — Failed-firing capture
# ---------------------------------------------------------------------------


class TestFailedFiringCapture:
    @pytest.mark.asyncio
    async def test_failing_resolver_record_has_status_failed(self):
        """FR7: A resolver that raises produces a FiringRecord with status='failed'."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = FailingResolver(
            name="bad_resolver",
            subscriptions=_TURN_START_SUB,
            error_message="something went wrong",
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        with pytest.raises(Exception):
            await engine.run([])

        failed_records = [r for r in tracer.get_trace() if r.status == "failed"]
        assert len(failed_records) >= 1
        failed = failed_records[0]
        assert failed.status == "failed"

    @pytest.mark.asyncio
    async def test_failing_resolver_record_contains_error_message(self):
        """FR7: The error field of the record contains the exception message."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = FailingResolver(
            name="bad_resolver",
            subscriptions=_TURN_START_SUB,
            error_message="something went wrong",
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        with pytest.raises(Exception):
            await engine.run([])

        failed_record = next(r for r in tracer.get_trace() if r.status == "failed")
        assert failed_record.error is not None
        assert "something went wrong" in failed_record.error

    @pytest.mark.asyncio
    async def test_failing_resolver_exception_propagates(self):
        """FR7: The exception still propagates — engine.run() raises."""
        tracer = CollectingTracer()
        bus = EventBus()
        resolver = FailingResolver(
            name="bad_resolver",
            subscriptions=_TURN_START_SUB,
            error_message="propagation check",
        )
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=tracer, event_bus=bus)
        engine = build_engine([layer], tracer=tracer)

        raised = False
        try:
            await engine.run([])
        except Exception as exc:
            raised = True
            assert "propagation check" in str(exc)

        assert raised, "Exception should have propagated out of engine.run()"


# ---------------------------------------------------------------------------
# FR10 — Zero-cost seam (tracer=None)
# ---------------------------------------------------------------------------


class TestZeroCostSeam:
    @pytest.mark.asyncio
    async def test_no_tracer_guard_prevents_on_firing_calls(self):
        """FR10: With tracer=None on layers, the guard if self._tracer is not None blocks on_firing."""
        bus = EventBus()

        class SpyTracer:
            """Records whether on_firing was ever called."""
            def __init__(self): self.calls = 0
            def on_firing(self, record: FiringRecord) -> None: self.calls += 1

        spy = SpyTracer()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="content")],
            subscriptions=_TURN_START_SUB,
        )
        # Build layer with spy wired in
        layer = make_system_layer(name="system_prompt", resolvers=[resolver], tracer=spy, event_bus=bus)
        # Build engine with NO tracer — this sets layer._tracer = None (overrides the spy we set above)
        engine = build_engine([layer], tracer=None)
        # layer._tracer is now None — the guard should block spy from being called

        await engine.run([])

        assert spy.calls == 0, f"on_firing was called {spy.calls} time(s) but tracer=None should block it"

    @pytest.mark.asyncio
    async def test_no_tracer_layer_content_still_populated(self):
        """FR10: With tracer=None, process_pending still works — layer content is populated."""
        bus = EventBus()
        resolver = StubResolver(
            name="sys_prompt",
            content=[TextBlock(text="You are helpful.")],
            subscriptions=_TURN_START_SUB,
        )
        layer = make_system_layer(
            name="system_prompt",
            resolvers=[resolver],
            tracer=None,
            event_bus=bus,
        )
        engine = build_engine([layer], tracer=None)

        result = await engine.run([])

        assert result.request is not None
        assert result.request.system is not None
        assert len(result.request.system) >= 1
        assert any(b.text == "You are helpful." for b in result.request.system)
