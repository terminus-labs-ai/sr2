"""Tests for sr2-75 — _eligible() guard generator + strategy dispatch in process_pending.

Covers:
  1. _eligible() filters out exhausted components (execution_count >= max_executions)
  2. _eligible() filters out components whose subscriptions don't match any event
  3. _eligible() yields eligible components in order
  4. process_pending uses _eligible internally (verified via integration tests)
  5. _snapshot_after uses strategy dispatch (no kind string branching)
  6. _fire_component uses strategy dispatch (no kind string branching)
"""

from __future__ import annotations

import pytest

from sr2.models import TextBlock, ToolDefinition
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.tracing import CollectingTracer


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
    """Self-incrementing resolver."""

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
    """Transformer that does NOT self-increment execution_count."""

    def __init__(
        self,
        name: str = "stub_transformer",
        result_content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 99,
        raise_on_call: Exception | None = None,
    ):
        self.name = name
        self._result_content = result_content
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
    """Tool provider returning preset tool definitions."""

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
    seq = [0]

    def _next_seq() -> int:
        v = seq[0]
        seq[0] += 1
        return v

    layer._next_firing_seq = _next_seq
    return layer, _bus


# ---------------------------------------------------------------------------
# 1. _eligible() method existence
# ---------------------------------------------------------------------------


class TestEligibleExists:
    """Layer._eligible must exist as a method."""

    def test_layer_has_eligible_method(self):
        """sr2-75: Layer must expose _eligible() after the DRY refactor."""
        layer, _ = make_layer()
        assert hasattr(layer, "_eligible"), (
            "Layer._eligible does not exist — the DRY refactor (sr2-75) has not been applied yet."
        )

    def test_eligible_returns_a_generator(self):
        """_eligible must return a generator/iterator."""
        import types

        layer, _ = make_layer(resolvers=[StubResolver()])
        result = layer._eligible(layer.resolvers, [_make_event()])
        # Should be a generator
        assert isinstance(result, types.GeneratorType)


# ---------------------------------------------------------------------------
# 2. _eligible() — execution_count guard
# ---------------------------------------------------------------------------


class TestEligibleExecutionGuard:
    """_eligible filters out components where execution_count >= max_executions."""

    def test_eligible_includes_component_under_max(self):
        """Component with execution_count < max_executions is yielded."""
        resolver = StubResolver(
            max_executions=3,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        resolver.execution_count = 1  # under max

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event()]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver in eligible

    def test_eligible_excludes_component_at_max(self):
        """Component with execution_count == max_executions is NOT yielded."""
        resolver = StubResolver(
            max_executions=1,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        resolver.execution_count = 1  # at max

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event()]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver not in eligible

    def test_eligible_excludes_component_over_max(self):
        """Component with execution_count > max_executions is NOT yielded."""
        resolver = StubResolver(
            max_executions=1,
            subscriptions=[EventSubscription(event_name="turn_start")],
        )
        resolver.execution_count = 5  # way over max

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event()]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver not in eligible

    def test_eligible_excludes_never_fired_component(self):
        """A component that has never fired (execution_count=0) IS eligible.
        (idle = eligible — it just may have wrong subscriptions)."""
        resolver = StubResolver(max_executions=1)
        # execution_count = 0 (default, never fired)

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event()]

        # This resolver has no subscriptions, so it won't match events.
        # But it should still pass the execution_count guard.
        # The subscription guard will filter it.
        # We just verify the execution_count guard doesn't block it.
        # To test purely the execution guard, give it a matching subscription:
        resolver.subscriptions = [EventSubscription(event_name="turn_start")]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver in eligible


# ---------------------------------------------------------------------------
# 3. _eligible() — subscription match guard
# ---------------------------------------------------------------------------


class TestEligibleSubscriptionGuard:
    """_eligible filters out components whose subscriptions don't match any event."""

    def test_eligible_includes_matching_subscription(self):
        """Component with a matching subscription is yielded."""
        resolver = StubResolver(
            subscriptions=[EventSubscription(event_name="turn_start")]
        )

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event("turn_start")]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver in eligible

    def test_eligible_excludes_non_matching_subscription(self):
        """Component with no matching subscription is NOT yielded."""
        resolver = StubResolver(
            subscriptions=[EventSubscription(event_name="some_other_event")]
        )

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event("turn_start")]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver not in eligible

    def test_eligible_excludes_empty_subscriptions(self):
        """Component with no subscriptions is NOT yielded."""
        resolver = StubResolver(subscriptions=[])

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event("turn_start")]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver not in eligible

    def test_eligible_any_subscription_match_is_enough(self):
        """If ANY subscription matches ANY event, the component is eligible."""
        resolver = StubResolver(
            subscriptions=[
                EventSubscription(event_name="non_matching_event"),
                EventSubscription(event_name="turn_start"),  # this one matches
            ]
        )

        layer, _ = make_layer(resolvers=[resolver])
        events = [_make_event("turn_start")]

        eligible = list(layer._eligible(layer.resolvers, events))
        assert resolver in eligible

    def test_eligible_multiple_components_mixed(self):
        """When some components match and some don't, only matching ones are yielded."""
        eligible_resolver = StubResolver(
            subscriptions=[EventSubscription(event_name="turn_start")]
        )
        ineligible_resolver = StubResolver(
            subscriptions=[EventSubscription(event_name="other_event")]
        )
        exhausted_resolver = StubResolver(
            subscriptions=[EventSubscription(event_name="turn_start")],
            max_executions=1,
        )
        exhausted_resolver.execution_count = 1

        layer, _ = make_layer(resolvers=[eligible_resolver, ineligible_resolver, exhausted_resolver])
        events = [_make_event("turn_start")]

        result = list(layer._eligible(layer.resolvers, events))
        assert eligible_resolver in result
        assert ineligible_resolver not in result
        assert exhausted_resolver not in result
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 4. _eligible() preserves order
# ---------------------------------------------------------------------------


class TestEligibleOrder:
    """_eligible yields components in the same order they appear in the list."""

    def test_eligible_preserves_input_order(self):
        """Components are yielded in input list order."""
        r1 = StubResolver(subscriptions=[EventSubscription(event_name="turn_start")])
        r2 = StubResolver(subscriptions=[EventSubscription(event_name="turn_start")])
        r3 = StubResolver(subscriptions=[EventSubscription(event_name="turn_start")])

        layer, _ = make_layer(resolvers=[r1, r2, r3])
        events = [_make_event("turn_start")]

        result = list(layer._eligible(layer.resolvers, events))
        assert result == [r1, r2, r3]

    def test_eligible_preserves_order_after_filtering(self):
        """When filtering, remaining components maintain original relative order."""
        r1 = StubResolver(subscriptions=[EventSubscription(event_name="turn_start")])
        r2 = StubResolver(max_executions=1)
        r2.execution_count = 1  # exhausted, filtered out
        r3 = StubResolver(subscriptions=[EventSubscription(event_name="turn_start")])

        layer, _ = make_layer(resolvers=[r1, r2, r3])
        events = [_make_event("turn_start")]

        result = list(layer._eligible(layer.resolvers, events))
        assert result == [r1, r3]


# ---------------------------------------------------------------------------
# 5. Integration — process_pending uses _eligible
# ---------------------------------------------------------------------------


class TestProcessPendingUsesEligible:
    """process_pending must delegate to _eligible for filtering."""

    @pytest.mark.asyncio
    async def test_exhausted_resolver_not_fired_via_process_pending(self):
        """A resolver at max_executions is not fired during process_pending."""
        resolver = StubResolver(
            name="exhausted",
            subscriptions=[EventSubscription(event_name="turn_start")],
            max_executions=1,
            content=[TextBlock(text="should not fire")],
        )
        resolver.execution_count = 1  # already at max

        layer, _ = make_layer(resolvers=[resolver])

        # Queue an event that would match the resolver
        layer._pending_events = [_make_event("turn_start")]

        await layer.process_pending()

        # Resolver should NOT have fired (execution_count should still be 1)
        assert resolver.execution_count == 1
        # Content should NOT have been added
        assert not any("should not fire" in str(b) for b in layer.get_content())

    @pytest.mark.asyncio
    async def test_non_matching_subscription_not_fired_via_process_pending(self):
        """A resolver whose subscription doesn't match the event is not fired."""
        resolver = StubResolver(
            name="no_match",
            subscriptions=[EventSubscription(event_name="different_event")],
            content=[TextBlock(text="should not fire")],
        )

        layer, _ = make_layer(resolvers=[resolver])

        layer._pending_events = [_make_event("turn_start")]

        await layer.process_pending()

        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_eligible_resolver_fired_via_process_pending(self):
        """An eligible resolver IS fired during process_pending."""
        block = TextBlock(text="fired")
        resolver = StubResolver(
            name="eligible",
            subscriptions=[EventSubscription(event_name="turn_start")],
            content=[block],
        )

        layer, _ = make_layer(resolvers=[resolver])

        layer._pending_events = [_make_event("turn_start")]

        await layer.process_pending()

        assert resolver.execution_count == 1
        assert block in layer.get_content()

    @pytest.mark.asyncio
    async def test_transformer_eligibility_via_process_pending(self):
        """Transformer eligibility filtering works through process_pending."""
        transformer = StubTransformer(
            name="eligible_transformer",
            subscriptions=[EventSubscription(event_name="turn_end")],
            result_content=[TextBlock(text="transformed")],
        )
        # Pre-load content
        layer, _ = make_layer(transformers=[transformer])
        layer._content = [TextBlock(text="original")]

        layer._pending_events = [_make_event("turn_end")]

        await layer.process_pending()

        # Transformer should have been called
        assert layer.get_content()[0].text == "transformed"

    @pytest.mark.asyncio
    async def test_tool_provider_eligibility_via_process_pending(self):
        """Tool provider eligibility filtering works through process_pending."""
        tp = StubToolProvider(
            name="eligible_tp",
            subscriptions=[EventSubscription(event_name="turn_start")],
            tools=[_make_tool("search")],
        )

        layer, _ = make_layer(tool_providers=[tp])

        layer._pending_events = [_make_event("turn_start")]

        await layer.process_pending()

        assert tp.execution_count == 1
        names = [td.name for td in layer.get_tool_definitions()]
        assert "search" in names


# ---------------------------------------------------------------------------
# 6. Strategy dispatch — _fire_component has no kind string branching
# ---------------------------------------------------------------------------


class TestStrategyDispatch:
    """Verify that _fire_component and _snapshot_after use strategy dispatch
    instead of kind string branching."""

    def test_no_kind_string_branching_in_fire_component(self):
        """_fire_component should not use if/elif/else on kind strings."""
        import inspect

        from sr2.pipeline.layer import Layer

        source = inspect.getsource(Layer._fire_component)
        # The refactored version should not have kind == "resolver" etc.
        assert 'kind == "resolver"' not in source, (
            "_fire_component still uses kind string branching. "
            "Refactor to use strategy dispatch."
        )
        assert 'kind == "transformer"' not in source
        assert 'kind == "tool_provider"' not in source

    def test_no_kind_string_branching_in_snapshot_after(self):
        """_snapshot_after should not use kind string branching."""
        import inspect

        from sr2.pipeline.layer import Layer

        source = inspect.getsource(Layer._snapshot_after)
        assert 'kind == "tool_provider"' not in source, (
            "_snapshot_after still uses kind string branching. "
            "Refactor to use strategy dispatch."
        )

    def test_layer_has_component_dispatch_attribute(self):
        """The Layer class should have a COMPONENT_DISPATCH class attribute
        (or similar) for strategy-based dispatch."""
        from sr2.pipeline.layer import Layer

        # Check for a dispatch table — could be named COMPONENT_DISPATCH,
        # _COMPONENT_DISPATCH, _DISPATCH, etc.
        dispatch_attrs = [
            attr for attr in dir(Layer)
            if "dispatch" in attr.lower() or "strategy" in attr.lower()
        ]
        assert len(dispatch_attrs) > 0, (
            "Layer should have a dispatch table for strategy-based component dispatch. "
            "Found no attributes with 'dispatch' or 'strategy' in the name."
        )


# ---------------------------------------------------------------------------
# 7. _fire_component still works correctly after refactor
# ---------------------------------------------------------------------------


class TestFireComponentAfterRefactor:
    """Verify _fire_component behavior is unchanged after the refactor."""

    @pytest.mark.asyncio
    async def test_resolver_fire_component_still_adds_content(self):
        """After refactor, resolver _fire_component still adds content correctly."""
        block = TextBlock(text="refactored resolver")
        resolver = StubResolver(content=[block])
        layer, _ = make_layer(resolvers=[resolver])

        await layer._fire_component(comp=resolver, kind="resolver", events=[_make_event()])

        assert block in layer._content

    @pytest.mark.asyncio
    async def test_transformer_fire_component_still_applies_content(self):
        """After refactor, transformer _fire_component still applies content correctly."""
        transformer = StubTransformer(result_content=[TextBlock(text="refactored transform")])
        layer, _ = make_layer(transformers=[transformer])
        layer._content = [TextBlock(text="original")]

        await layer._fire_component(comp=transformer, kind="transformer", events=[_make_event()])

        assert layer._content[0].text == "refactored transform"

    @pytest.mark.asyncio
    async def test_tool_provider_fire_component_still_adds_tools(self):
        """After refactor, tool_provider _fire_component still adds tools correctly."""
        tp = StubToolProvider(tools=[_make_tool("post_refactor_tool")])
        layer, _ = make_layer(tool_providers=[tp])

        await layer._fire_component(comp=tp, kind="tool_provider", events=[_make_event()])

        names = [td.name for td in layer._tool_definitions]
        assert "post_refactor_tool" in names

    @pytest.mark.asyncio
    async def test_tracer_still_works_after_refactor(self):
        """FiringRecord emission still works correctly after refactor."""
        tracer = CollectingTracer()
        resolver = StubResolver(content=[TextBlock(text="tracked")])
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)

        await layer._fire_component(comp=resolver, kind="resolver", events=[_make_event()])

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].kind == "resolver"
        assert records[0].status == "ok"

    @pytest.mark.asyncio
    async def test_failure_path_still_works_after_refactor(self):
        """Exception handling in _fire_component still works after refactor."""
        tracer = CollectingTracer()
        resolver = StubResolver(raise_on_call=RuntimeError("post refactor failure"))
        layer, _ = make_layer(resolvers=[resolver], tracer=tracer)

        with pytest.raises(RuntimeError, match="post refactor failure"):
            await layer._fire_component(comp=resolver, kind="resolver", events=[_make_event()])

        records = tracer.get_trace()
        assert len(records) == 1
        assert records[0].status == "failed"
        assert "post refactor failure" in (records[0].error or "")
