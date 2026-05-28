"""Tests for sr2-36 — SRP+OCP: split Layer._fire_component, unify Component protocol.

These tests define the desired post-refactor API:

  Component protocol:
    - Any object with `run(layer_view, events) -> ComponentResult` satisfies it
    - Resolver, Transformer, and ToolProvider adapt their existing methods to `run()`
    - Layer dispatches uniformly via Component.run() instead of branching on kind

  OCP guarantee:
    - Adding a new component type requires only implementing Component.run()
    - No changes to Layer are required
    - A stub not extending any existing class gets executed by process_pending()

All tests in this file FAIL until the refactor is implemented.
"""

from __future__ import annotations

import pytest

from sr2.models import TextBlock, ToolDefinition
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent, TransformationResult
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(name: str = "turn_start") -> Event:
    return Event(name=name, phase=EventPhase.COMPLETED, source_layer="engine")


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


def _make_layer(
    name: str = "test_layer",
    resolvers: list | None = None,
    transformers: list | None = None,
    tool_providers: list | None = None,
    components: list | None = None,
    bus: EventBus | None = None,
    tracer=None,
):
    """Factory for a Layer with optional uniform component list support."""
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
    if components is not None:
        # Post-refactor: Layer accepts a uniform `components` list
        layer.components = components
    seq = [0]

    def _next_seq() -> int:
        v = seq[0]
        seq[0] += 1
        return v

    layer._next_firing_seq = _next_seq
    return layer, _bus


# ---------------------------------------------------------------------------
# Minimal stubs used across many tests — mimic existing component shapes
# ---------------------------------------------------------------------------


class StubResolver:
    """Minimal Resolver-protocol stub."""

    def __init__(
        self,
        name: str = "stub_resolver",
        content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._content = content or [TextBlock(text="resolved")]
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start")
        ]
        self.max_executions = max_executions
        self.execution_count = 0

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            content=list(self._content),
        )


class StubTransformer:
    """Minimal Transformer-protocol stub."""

    def __init__(
        self,
        name: str = "stub_transformer",
        result_content: list | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._result_content = result_content
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start")
        ]
        self.max_executions = max_executions
        self.execution_count = 0

    async def transform(
        self, content: list, events: list[Event]
    ) -> TransformationResult:
        self.execution_count += 1
        return TransformationResult(
            transformer_name=self.name,
            source_layer="test",
            content=self._result_content,
        )


class StubToolProvider:
    """Minimal ToolProvider-protocol stub."""

    def __init__(
        self,
        name: str = "stub_tp",
        tools: list[ToolDefinition] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._tools = tools or [_make_tool("default_tool")]
        self.subscriptions = subscriptions or [
            EventSubscription(event_name="turn_start")
        ]
        self.max_executions = max_executions
        self.execution_count = 0

    async def provide(self, events: list[Event]) -> list[ToolDefinition]:
        self.execution_count += 1
        return list(self._tools)


# ---------------------------------------------------------------------------
# TestComponentProtocol
# ---------------------------------------------------------------------------


class TestComponentProtocol:
    """Verify that Component is a real protocol and all adapters satisfy it."""

    def test_class_with_only_run_satisfies_component_protocol(self):
        """A class implementing run(layer_view, events) -> ComponentResult is a Component."""
        from sr2.pipeline.protocols import Component, ComponentResult  # type: ignore[attr-defined]

        class MinimalComponent:
            name = "minimal"
            subscriptions: list[EventSubscription] = []
            max_executions: int = 1
            execution_count: int = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                return ComponentResult(component_name=self.name, source_layer="test")

        assert isinstance(MinimalComponent(), Component), (
            "MinimalComponent implements run() but is not recognised as Component. "
            "Component protocol is not yet defined or the run() signature doesn't match."
        )

    def test_resolver_satisfies_component_protocol(self):
        """Resolver (or its adapter) satisfies the Component protocol."""
        from sr2.pipeline.protocols import Component  # type: ignore[attr-defined]

        resolver = StubResolver()
        assert isinstance(resolver, Component), (
            "Resolver does not satisfy Component protocol. "
            "The Resolver adapter wrapping resolve()->run() is not yet implemented."
        )

    def test_transformer_satisfies_component_protocol(self):
        """Transformer (or its adapter) satisfies the Component protocol."""
        from sr2.pipeline.protocols import Component  # type: ignore[attr-defined]

        transformer = StubTransformer()
        assert isinstance(transformer, Component), (
            "Transformer does not satisfy Component protocol. "
            "The Transformer adapter wrapping transform()->run() is not yet implemented."
        )

    def test_tool_provider_satisfies_component_protocol(self):
        """ToolProvider (or its adapter) satisfies the Component protocol."""
        from sr2.pipeline.protocols import Component  # type: ignore[attr-defined]

        tp = StubToolProvider()
        assert isinstance(tp, Component), (
            "ToolProvider does not satisfy Component protocol. "
            "The ToolProvider adapter wrapping provide()->run() is not yet implemented."
        )


# ---------------------------------------------------------------------------
# TestUniformDispatch
# ---------------------------------------------------------------------------


class TestUniformDispatch:
    """Layer dispatches through Component.run() uniformly for all component types."""

    @pytest.mark.asyncio
    async def test_custom_component_processed_same_as_builtin_resolver(self):
        """A custom Component implementation is processed identically to a Resolver."""
        from sr2.pipeline.protocols import Component, ComponentResult  # type: ignore[attr-defined]

        fired = []

        class CustomComponent:
            name = "custom"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                fired.append(True)
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                    content=[TextBlock(text="from custom")],
                )

        assert isinstance(CustomComponent(), Component)

        custom = CustomComponent()
        layer, bus = _make_layer(components=[custom])

        bus.queue(_make_event("turn_start"))
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert len(fired) == 1, (
            "Custom Component.run() was not called during process_pending(). "
            "Layer must dispatch uniformly through Component.run()."
        )

    @pytest.mark.asyncio
    async def test_stub_component_not_extending_any_base_gets_executed(self):
        """A stub implementing only Component.run() (no base class) is executed by Layer."""
        from sr2.pipeline.protocols import Component, ComponentResult  # type: ignore[attr-defined]

        run_calls = []

        class PureStub:
            """No base class, no existing protocol — just run()."""

            name = "pure_stub"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                run_calls.append(events)
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                )

        stub = PureStub()
        assert isinstance(stub, Component)

        layer, bus = _make_layer(components=[stub])
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert len(run_calls) == 1, (
            "PureStub.run() was not called. OCP violation: Layer must not require "
            "components to extend Resolver, Transformer, or ToolProvider."
        )

    @pytest.mark.asyncio
    async def test_uniform_dispatch_produces_same_result_as_resolver_kind(self):
        """Uniform Component dispatch produces same observable content as resolver kind dispatch."""
        block = TextBlock(text="uniform result")
        resolver = StubResolver(content=[block])

        layer, bus = _make_layer(resolvers=[resolver])
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        content = layer.get_content()
        assert block in content, (
            "Resolver content not present after process_pending(). "
            "Uniform dispatch must preserve existing resolver behavior."
        )

    @pytest.mark.asyncio
    async def test_uniform_dispatch_produces_same_result_as_transformer_kind(self):
        """Uniform Component dispatch produces same content replacement as transformer kind."""
        original = TextBlock(text="original")
        replaced = TextBlock(text="replaced")
        transformer = StubTransformer(result_content=[replaced])

        layer, bus = _make_layer(transformers=[transformer])
        layer._content = [original]
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        content = layer.get_content()
        assert replaced in content, (
            "Transformer did not replace content. "
            "Uniform dispatch must preserve existing transformer behavior."
        )
        assert original not in content, (
            "Original content still present after transformer ran. "
            "set_content() must be applied during uniform dispatch."
        )

    @pytest.mark.asyncio
    async def test_uniform_dispatch_produces_same_result_as_tool_provider_kind(self):
        """Uniform Component dispatch produces same tool definitions as tool_provider kind."""
        tool = _make_tool("my_tool")
        tp = StubToolProvider(tools=[tool])

        layer, bus = _make_layer(tool_providers=[tp])
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        names = [td.name for td in layer._tool_definitions]
        assert "my_tool" in names, (
            "ToolProvider tools not present after process_pending(). "
            "Uniform dispatch must preserve existing tool_provider behavior."
        )

    @pytest.mark.asyncio
    async def test_new_component_type_in_layer_gets_executed_during_process_pending(self):
        """A Component registered on Layer runs when its subscribed event fires."""
        from sr2.pipeline.protocols import Component, ComponentResult  # type: ignore[attr-defined]

        fired_events: list[list[Event]] = []

        class NewKindComponent:
            name = "new_kind"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                fired_events.append(list(events))
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                )

        comp = NewKindComponent()
        layer, bus = _make_layer(components=[comp])
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert len(fired_events) == 1
        assert fired_events[0][0].name == "turn_start"


# ---------------------------------------------------------------------------
# TestOCPGuarantee
# ---------------------------------------------------------------------------


class TestOCPGuarantee:
    """Verify OCP: new component types need no changes to Layer."""

    @pytest.mark.asyncio
    async def test_layer_fires_all_resolvers_transformers_and_tool_providers_per_turn(self):
        """Existing behavior preserved: all three kinds fire when their event arrives."""
        resolver = StubResolver()
        transformer = StubTransformer()
        tp = StubToolProvider()

        layer, bus = _make_layer(
            resolvers=[resolver],
            transformers=[transformer],
            tool_providers=[tp],
        )
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert resolver.execution_count == 1, "Resolver was not executed"
        assert transformer.execution_count == 1, "Transformer was not executed"
        assert tp.execution_count == 1, "ToolProvider was not executed"

    @pytest.mark.asyncio
    async def test_stub_component_without_base_class_is_executed(self):
        """OCP: A stub implementing only run() (no base class) is executed by Layer."""
        from sr2.pipeline.protocols import ComponentResult  # type: ignore[attr-defined]

        call_log: list[str] = []

        class FourthKind:
            """Not a Resolver, not a Transformer, not a ToolProvider."""

            name = "fourth_kind"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                call_log.append("ran")
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                )

        fourth = FourthKind()
        layer, bus = _make_layer(components=[fourth])
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert call_log == ["ran"], (
            "FourthKind.run() was never called. "
            "Layer must not require isinstance(comp, Resolver|Transformer|ToolProvider). "
            "Any Component.run() implementor must be dispatchable without modifying Layer."
        )

    @pytest.mark.asyncio
    async def test_layer_with_mixed_components_fires_all(self):
        """Layer with Resolver + Transformer + ToolProvider + custom Component fires all four."""
        from sr2.pipeline.protocols import ComponentResult  # type: ignore[attr-defined]

        resolver = StubResolver()
        transformer = StubTransformer()
        tp = StubToolProvider()
        custom_calls: list[bool] = []

        class CustomFifth:
            name = "custom_fifth"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                custom_calls.append(True)
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                )

        fifth = CustomFifth()
        layer, bus = _make_layer(
            resolvers=[resolver],
            transformers=[transformer],
            tool_providers=[tp],
            components=[fifth],
        )
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert resolver.execution_count == 1, "Resolver not fired"
        assert transformer.execution_count == 1, "Transformer not fired"
        assert tp.execution_count == 1, "ToolProvider not fired"
        assert custom_calls == [True], "Custom fifth component not fired"

    @pytest.mark.asyncio
    async def test_component_max_executions_respected_for_custom_component(self):
        """max_executions guard applies uniformly to custom Component implementations."""
        from sr2.pipeline.protocols import ComponentResult  # type: ignore[attr-defined]

        call_count = [0]

        class LimitedComponent:
            name = "limited"
            subscriptions = [EventSubscription(event_name="turn_start")]
            max_executions = 1
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                call_count[0] += 1
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                )

        comp = LimitedComponent()
        layer, bus = _make_layer(components=[comp])

        # Fire the event twice
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert call_count[0] == 1, (
            f"LimitedComponent ran {call_count[0]} times; expected 1. "
            "max_executions must be enforced for custom Component types too."
        )

    @pytest.mark.asyncio
    async def test_component_subscription_filter_respected(self):
        """Components only run when their subscribed event fires — not on arbitrary events."""
        from sr2.pipeline.protocols import ComponentResult  # type: ignore[attr-defined]

        fired_on: list[str] = []

        class SelectiveComponent:
            name = "selective"
            subscriptions = [EventSubscription(event_name="specific_event")]
            max_executions = 10
            execution_count = 0

            async def run(
                self, layer_view: object, events: list[Event]
            ) -> ComponentResult:
                fired_on.extend(e.name for e in events)
                self.execution_count += 1
                return ComponentResult(
                    component_name=self.name,
                    source_layer="test",
                )

        comp = SelectiveComponent()
        layer, bus = _make_layer(components=[comp])

        # Wrong event — should not fire
        layer.handle_event(_make_event("wrong_event"))
        await layer.process_pending()
        assert fired_on == [], "Component fired on wrong event"

        # Right event — should fire
        layer.handle_event(_make_event("specific_event"))
        await layer.process_pending()
        assert "specific_event" in fired_on, "Component did not fire on subscribed event"
