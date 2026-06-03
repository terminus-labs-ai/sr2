"""Tests for sr2-67 — Component protocol removal (dead code cut).

The Component protocol, ComponentResult dataclass, and Layer.components
extension point were unreachable: no LayerConfig field, no entry-point
group, no wiring through the public SR2() constructor.  They existed
only as tests that constructed Layer(components=[...]) directly.

This test file verifies the cut is clean:
  1. Component and ComponentResult are no longer importable from protocols
  2. Layer.__init__ no longer accepts components parameter
  3. Layer still works correctly with just resolvers, transformers, tool_providers
"""

import pytest

from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.events import EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.layer import Layer
from sr2.models import TextBlock


class TestComponentProtocolRemoved:
    """Component protocol and ComponentResult have been removed."""

    def test_component_not_in_protocols(self):
        with pytest.raises(ImportError):
            from sr2.pipeline.protocols import Component  # noqa: F401

    def test_component_result_not_in_protocols(self):
        with pytest.raises(ImportError):
            from sr2.pipeline.protocols import ComponentResult  # noqa: F401


class TestLayerNoComponentsParam:
    """Layer.__init__ no longer accepts a components parameter."""

    def test_layer_no_components_attribute(self):
        class DummyCounter:
            def count(self, content):
                return len(content)

        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1000,
            resolvers=[],
            transformers=[],
            token_counter=DummyCounter(),
        )
        assert not hasattr(layer, "components")


class TestLayerWithoutComponents:
    """Layer works correctly without the components extension point."""

    def test_subscriptions_without_components(self):
        class DummyResolver:
            name = "dummy"
            subscriptions = [EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)]
            max_executions = 1
            execution_count = 0

            async def resolve(self, events):
                return ResolvedContent(content=[TextBlock(text="test")])

        class DummyCounter:
            def count(self, content):
                return len(content)

        resolver = DummyResolver()
        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1000,
            resolvers=[resolver],
            transformers=[],
            token_counter=DummyCounter(),
        )

        subs = layer.subscriptions
        assert len(subs) == 1
        assert subs[0].event_name == "turn_start"

    def test_is_done_without_components(self):
        class DummyResolver:
            name = "dummy"
            subscriptions = []
            max_executions = 1
            execution_count = 0

        class DummyCounter:
            def count(self, content):
                return len(content)

        resolver = DummyResolver()
        layer = Layer(
            name="test",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=1000,
            resolvers=[resolver],
            transformers=[],
            token_counter=DummyCounter(),
        )

        assert layer.is_done() is True  # never fired = done

        resolver.execution_count = 1
        assert layer.is_done() is True  # hit max = done
