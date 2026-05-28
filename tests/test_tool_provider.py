"""Tests for Step 4: ToolProvider kind.

Covers:
  A. Config models — ToolProviderConfig and LayerConfig.tool_providers
  B. Layer — tool provider routing, subscriptions, is_done, reset
  C. Orchestrator — _TOOL_PROVIDERS registry
  D. End-to-end — ToolProvider -> CompletionRequest.tools
"""

from __future__ import annotations

import importlib.metadata
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sr2.models import TextBlock, ToolDefinition, TokenUsage
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.plugins.registry import PluginRegistry
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def reset_plugin_registries():
    """Reset _RESOLVERS, _TRANSFORMERS, and _TOOL_PROVIDERS between tests.

    Prevents discovery state from leaking between tests that patch entry_points.
    """
    import sr2.orchestrator as orch

    def _reset():
        orch._RESOLVERS._discovered = False
        orch._RESOLVERS._classes = {}
        orch._RESOLVERS._collisions = {}
        orch._TRANSFORMERS._discovered = False
        orch._TRANSFORMERS._classes = {}
        orch._TRANSFORMERS._collisions = {}
        if hasattr(orch, "_TOOL_PROVIDERS"):
            orch._TOOL_PROVIDERS._discovered = False
            orch._TOOL_PROVIDERS._classes = {}
            orch._TOOL_PROVIDERS._collisions = {}

    _reset()
    yield
    _reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_definition(name: str = "do_thing") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="A test tool",
        input_schema={"type": "object", "properties": {}},
    )


def _make_event(name: str = "turn_start") -> Event:
    return Event(name=name, phase=EventPhase.COMPLETED, source_layer="engine")


def _make_layer(
    name: str = "tools",
    target: CompilationTarget = CompilationTarget.TOOLS,
    tool_providers: list | None = None,
) -> Any:
    """Build a bare Layer with no resolvers, no transformers, optional tool_providers."""
    from sr2.pipeline.layer import Layer

    return Layer(
        name=name,
        target=target,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
        tool_providers=tool_providers or [],
    )


class _MockLLM:
    """Minimal LLMCallable that records the request it was given."""

    def __init__(self) -> None:
        self.last_request: CompletionRequest | None = None

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.last_request = request
        return CompletionResponse(
            id="mock-resp",
            content=[TextBlock(text="ok")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest):
        self.last_request = request
        yield StreamEvent(type="text", text="ok")
        yield StreamEvent(type="end")


# ---------------------------------------------------------------------------
# A. Config models
# ---------------------------------------------------------------------------


class TestToolProviderConfig:
    """A1-A2: ToolProviderConfig construction and defaults."""

    def test_construct_with_type_only(self):
        """A1: ToolProviderConfig can be constructed with just type=."""
        from sr2.config.models import ToolProviderConfig

        cfg = ToolProviderConfig(type="my_provider")
        assert cfg.type == "my_provider"

    def test_default_config_is_empty_dict(self):
        """A2a: Default config is {}."""
        from sr2.config.models import ToolProviderConfig

        cfg = ToolProviderConfig(type="x")
        assert cfg.config == {}

    def test_default_subscriptions_is_empty_list(self):
        """A2b: Default subscriptions is []."""
        from sr2.config.models import ToolProviderConfig

        cfg = ToolProviderConfig(type="x")
        assert cfg.subscriptions == []

    def test_default_max_executions_is_one(self):
        """A2c: Default max_executions is 1."""
        from sr2.config.models import ToolProviderConfig

        cfg = ToolProviderConfig(type="x")
        assert cfg.max_executions == 1

    def test_explicit_values_are_stored(self):
        """A2d: Explicit config, subscriptions, and max_executions are preserved."""
        from sr2.config.models import EventSubscriptionConfig, ToolProviderConfig

        cfg = ToolProviderConfig(
            type="fancy",
            config={"key": "val"},
            subscriptions=[EventSubscriptionConfig(event="turn_start")],
            max_executions=5,
        )
        assert cfg.config == {"key": "val"}
        assert len(cfg.subscriptions) == 1
        assert cfg.subscriptions[0].event == "turn_start"
        assert cfg.max_executions == 5


class TestLayerConfigToolProviders:
    """A3-A4: LayerConfig.tool_providers field."""

    def test_layer_config_accepts_tool_providers(self):
        """A3: LayerConfig accepts tool_providers=[ToolProviderConfig(type='x')]."""
        from sr2.config.models import LayerConfig, ResolverConfig, ToolProviderConfig

        cfg = LayerConfig(
            name="tools",
            target="tools",
            resolvers=[],
            tool_providers=[ToolProviderConfig(type="x")],
        )
        assert len(cfg.tool_providers) == 1
        assert cfg.tool_providers[0].type == "x"

    def test_layer_config_tool_providers_defaults_to_none(self):
        """A4: LayerConfig.tool_providers defaults to None when omitted."""
        from sr2.config.models import LayerConfig

        cfg = LayerConfig(name="system", target="system", resolvers=[])
        assert cfg.tool_providers is None


# ---------------------------------------------------------------------------
# B. Layer — tool provider routing
# ---------------------------------------------------------------------------


class _SpyToolProvider:
    """Minimal ToolProvider that records calls and returns fixed definitions."""

    name: str = "spy"

    def __init__(
        self,
        tool_defs: list[ToolDefinition] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ) -> None:
        self.tool_defs = tool_defs or [_make_tool_definition()]
        self.subscriptions: list[EventSubscription] = subscriptions or [
            EventSubscription(event_name="turn_start")
        ]
        self.max_executions = max_executions
        self.execution_count = 0
        self.calls: list[list[Event]] = []

    async def provide(self, events: list[Event]) -> list[ToolDefinition]:
        # ToolProvider must increment execution_count itself inside provide(),
        # the same way Resolver increments it inside resolve(). Layer.process_pending
        # checks execution_count BEFORE calling provide(), so the count must be
        # updated by the component to signal exhaustion on the next event.
        self.calls.append(events)
        self.execution_count += 1
        return self.tool_defs

    @classmethod
    def build(cls, config: Any, deps: Any) -> "_SpyToolProvider":
        return cls()


class TestLayerAcceptsToolProviders:
    """B5: Layer accepts a tool_providers list in __init__."""

    def test_layer_accepts_tool_providers_kwarg(self):
        """Layer can be constructed with tool_providers=[<provider>]."""
        provider = _SpyToolProvider()
        layer = _make_layer(tool_providers=[provider])
        assert provider in layer.tool_providers

    def test_layer_tool_providers_defaults_to_empty(self):
        """Layer defaults to empty tool_providers when not passed."""
        layer = _make_layer()
        assert layer.tool_providers == []


class TestLayerToolProviderRouting:
    """B6-B7: process_pending routes matching events to provide(), output to _tool_definitions."""

    @pytest.mark.asyncio
    async def test_matching_event_calls_provide(self):
        """B6: A subscribed event triggers provide() on the tool provider."""
        provider = _SpyToolProvider()
        layer = _make_layer(tool_providers=[provider])

        event = _make_event("turn_start")
        layer.handle_event(event)
        await layer.process_pending()

        assert len(provider.calls) == 1

    @pytest.mark.asyncio
    async def test_provide_output_added_to_tool_definitions(self):
        """B6: ToolDefinition returned by provide() is stored in _tool_definitions."""
        defs = [_make_tool_definition("search"), _make_tool_definition("write")]
        provider = _SpyToolProvider(tool_defs=defs)
        layer = _make_layer(tool_providers=[provider])

        event = _make_event("turn_start")
        layer.handle_event(event)
        await layer.process_pending()

        assert layer._tool_definitions == defs

    @pytest.mark.asyncio
    async def test_non_matching_event_does_not_call_provide(self):
        """B6: An event that doesn't match the subscription is not routed to provide()."""
        provider = _SpyToolProvider(
            subscriptions=[EventSubscription(event_name="turn_end")]
        )
        layer = _make_layer(tool_providers=[provider])

        event = _make_event("turn_start")  # doesn't match "turn_end"
        layer.handle_event(event)
        await layer.process_pending()

        assert provider.calls == []
        assert layer._tool_definitions == []

    @pytest.mark.asyncio
    async def test_compile_returns_tool_definitions_from_providers(self):
        """B7: compile() on a TOOLS layer returns the definitions accumulated from providers."""
        tool_def = _make_tool_definition("summarize")
        provider = _SpyToolProvider(tool_defs=[tool_def])
        layer = _make_layer(target=CompilationTarget.TOOLS, tool_providers=[provider])

        event = _make_event("turn_start")
        layer.handle_event(event)
        await layer.process_pending()

        result = layer.compile()
        assert result == [tool_def]


class TestLayerSubscriptionsIncludeToolProviders:
    """B8: Layer.subscriptions includes tool provider subscriptions."""

    def test_subscriptions_includes_tool_provider_subs(self):
        """B8: subscriptions property returns tool provider subscriptions alongside resolver subs."""
        sub = EventSubscription(event_name="custom_event")
        provider = _SpyToolProvider(subscriptions=[sub])
        layer = _make_layer(tool_providers=[provider])

        all_subs = layer.subscriptions
        assert sub in all_subs

    def test_subscriptions_includes_all_tool_providers(self):
        """B8: subscriptions aggregates across multiple tool providers."""
        sub_a = EventSubscription(event_name="alpha")
        sub_b = EventSubscription(event_name="beta")
        provider_a = _SpyToolProvider(subscriptions=[sub_a])
        provider_b = _SpyToolProvider(subscriptions=[sub_b])
        layer = _make_layer(tool_providers=[provider_a, provider_b])

        all_subs = layer.subscriptions
        assert sub_a in all_subs
        assert sub_b in all_subs


class TestLayerIsDoneIncludesToolProviders:
    """B9: is_done() returns False when a tool provider is mid-execution."""

    def test_is_done_true_when_no_providers_fired(self):
        """B9a: is_done() is True before any provider fires (execution_count == 0)."""
        provider = _SpyToolProvider(max_executions=2)
        assert provider.execution_count == 0
        layer = _make_layer(tool_providers=[provider])
        assert layer.is_done() is True

    def test_is_done_false_when_provider_partially_executed(self):
        """B9b: is_done() is False when provider has fired but not reached max_executions."""
        provider = _SpyToolProvider(max_executions=3)
        provider.execution_count = 1  # mid-run
        layer = _make_layer(tool_providers=[provider])
        assert layer.is_done() is False

    def test_is_done_true_when_provider_fully_exhausted(self):
        """B9c: is_done() is True when execution_count == max_executions."""
        provider = _SpyToolProvider(max_executions=2)
        provider.execution_count = 2  # exhausted
        layer = _make_layer(tool_providers=[provider])
        assert layer.is_done() is True


class TestLayerToolProviderExecutionCount:
    """B10-B11: execution_count incremented; exhausted providers are not called again."""

    @pytest.mark.asyncio
    async def test_execution_count_incremented_after_provide(self):
        """B10: execution_count is incremented after provide() is called."""
        provider = _SpyToolProvider()
        layer = _make_layer(tool_providers=[provider])

        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert provider.execution_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_provider_not_called_again(self):
        """B11: A provider at max_executions is skipped on subsequent events."""
        provider = _SpyToolProvider(max_executions=1)
        layer = _make_layer(tool_providers=[provider])

        # First trigger — should fire
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()
        assert provider.execution_count == 1

        # Second trigger — should be skipped
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()
        assert provider.execution_count == 1  # not incremented again
        assert len(provider.calls) == 1


class TestLayerToolDefinitionsReset:
    """B12: _tool_definitions is reset between turns."""

    @pytest.mark.asyncio
    async def test_tool_definitions_reset_between_turns(self):
        """B12: After a reset, _tool_definitions is empty even if providers fired last turn."""
        provider = _SpyToolProvider(max_executions=10)
        layer = _make_layer(tool_providers=[provider])

        # First turn
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()
        assert len(layer._tool_definitions) == 1

        # Reset (as engine does between turns)
        layer.reset_tools()

        assert layer._tool_definitions == []

    @pytest.mark.asyncio
    async def test_tool_definitions_accumulate_fresh_after_reset(self):
        """B12: After reset, a new turn accumulates fresh definitions."""
        new_def = _make_tool_definition("fresh_tool")
        provider = _SpyToolProvider(tool_defs=[new_def], max_executions=10)
        layer = _make_layer(tool_providers=[provider])

        # Turn 1
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        # Reset between turns
        layer.reset_tools()
        provider.execution_count = 0  # engine resets execution counts too

        # Turn 2
        layer.handle_event(_make_event("turn_start"))
        await layer.process_pending()

        assert layer._tool_definitions == [new_def]
        # Must not contain stale definitions from turn 1
        assert len(layer._tool_definitions) == 1


# ---------------------------------------------------------------------------
# C. Orchestrator — _TOOL_PROVIDERS registry
# ---------------------------------------------------------------------------


class TestOrchestratorToolProvidersRegistry:
    """C13: orchestrator._TOOL_PROVIDERS is a PluginRegistry instance."""

    def test_tool_providers_is_plugin_registry(self):
        """_TOOL_PROVIDERS exists on the orchestrator module and is a PluginRegistry."""
        import sr2.orchestrator as orch

        assert hasattr(orch, "_TOOL_PROVIDERS"), (
            "orchestrator._TOOL_PROVIDERS does not exist; "
            "add it alongside _RESOLVERS and _TRANSFORMERS"
        )
        assert isinstance(orch._TOOL_PROVIDERS, PluginRegistry)

    def test_tool_providers_registry_group(self):
        """_TOOL_PROVIDERS uses the 'sr2.tool_providers' entry-point group."""
        import sr2.orchestrator as orch

        assert orch._TOOL_PROVIDERS._group == "sr2.tool_providers"


# ---------------------------------------------------------------------------
# D. End-to-end test
# ---------------------------------------------------------------------------


class _E2EToolProvider:
    """Full ToolProvider implementation for end-to-end validation."""

    name: str = "e2e_spy"

    def __init__(self) -> None:
        self.subscriptions: list[EventSubscription] = [
            EventSubscription(event_name="turn_start")
        ]
        self.max_executions: int = 1
        self.execution_count: int = 0
        self._tool_defs = [
            ToolDefinition(
                name="e2e_tool",
                description="End-to-end test tool",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

    async def provide(self, events: list[Event]) -> list[ToolDefinition]:
        self.execution_count += 1
        return self._tool_defs

    @classmethod
    def build(cls, config: Any, deps: Any) -> "_E2EToolProvider":
        return cls()


def _make_e2e_ep_side_effect(tool_provider_cls: type):
    """Return a group-aware side_effect for entry_points patching.

    - sr2.resolvers  → StaticResolver
    - sr2.transformers → []
    - sr2.tool_providers → _E2EToolProvider (or given class)
    - anything else → []
    """

    def _side_effect(group: str):
        if group == "sr2.resolvers":
            from sr2.pipeline.resolvers.static import StaticResolver

            ep = MagicMock(spec=importlib.metadata.EntryPoint)
            ep.name = "static"
            ep.load.return_value = StaticResolver
            dist = MagicMock()
            dist.name = "sr2"
            ep.dist = dist
            return [ep]

        if group == "sr2.tool_providers":
            ep = MagicMock(spec=importlib.metadata.EntryPoint)
            ep.name = "e2e_spy"
            ep.load.return_value = tool_provider_cls
            dist = MagicMock()
            dist.name = "sr2-e2e-test"
            ep.dist = dist
            return [ep]

        return []

    return _side_effect


class TestEndToEnd:
    """D14: ToolProvider registered via entry_points -> CompletionRequest.tools."""

    @pytest.mark.asyncio
    async def test_tool_provider_tools_appear_on_completion_request(
        self, reset_plugin_registries
    ):
        """D14: A ToolProvider on a TOOLS layer populates CompletionRequest.tools.

        Setup:
          - Two-layer config: 'system' (SYSTEM target, static resolver) and
            'tools' (TOOLS target, e2e_spy tool provider).
          - entry_points patched: static → StaticResolver, e2e_spy → _E2EToolProvider.
          - SR2 is constructed and engine.run() is driven through a full turn.
          - The resulting CompletionRequest.tools must contain the ToolDefinition
            returned by _E2EToolProvider.provide().
        """
        from sr2.config.models import (
            LayerConfig,
            PipelineConfig,
            ResolverConfig,
            ToolProviderConfig,
        )
        from sr2.orchestrator import SR2

        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are a helpful assistant."},
                        )
                    ],
                ),
                LayerConfig(
                    name="tools",
                    target="tools",
                    resolvers=[],
                    tool_providers=[ToolProviderConfig(type="e2e_spy")],
                ),
            ]
        )

        mock_llm = _MockLLM()

        side_effect = _make_e2e_ep_side_effect(_E2EToolProvider)

        with patch(
            "sr2.plugins.registry.entry_points", side_effect=side_effect
        ):
            sr2 = SR2(
                pipeline_config=pipeline_config,
                llm={"default": mock_llm},
                token_counter=CharacterTokenCounter(),
            )

            # Run one turn — collect the stream to drive the generator
            stream = sr2.turn([TextBlock(text="hello")])
            async for _ in stream:
                pass

        # The LLM was called with a request — check tools
        assert mock_llm.last_request is not None
        assert mock_llm.last_request.tools is not None
        assert len(mock_llm.last_request.tools) == 1
        assert mock_llm.last_request.tools[0].name == "e2e_tool"

    @pytest.mark.asyncio
    async def test_tool_provider_fires_on_second_turn(self, reset_plugin_registries):
        """D15: Tools appear in CompletionRequest.tools on the SECOND turn() call.

        Regression for SR2 orchestrator not resetting execution_count for
        tool_providers between turns. Without the fix, a max_executions=1 provider
        fires only on turn 1; on turn 2 it is skipped (execution_count >= max_executions)
        and the model loses tool access.
        """
        from sr2.config.models import (
            LayerConfig,
            PipelineConfig,
            ResolverConfig,
            ToolProviderConfig,
        )
        from sr2.orchestrator import SR2

        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are a helpful assistant."},
                        )
                    ],
                ),
                LayerConfig(
                    name="tools",
                    target="tools",
                    resolvers=[],
                    tool_providers=[ToolProviderConfig(type="e2e_spy")],
                ),
            ]
        )

        mock_llm = _MockLLM()
        side_effect = _make_e2e_ep_side_effect(_E2EToolProvider)

        with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
            sr2 = SR2(
                pipeline_config=pipeline_config,
                llm={"default": mock_llm},
                token_counter=CharacterTokenCounter(),
            )

            # Turn 1
            stream = sr2.turn([TextBlock(text="hello")])
            async for _ in stream:
                pass
            turn1_tools = mock_llm.last_request.tools

            # Turn 2 — execution_count must be reset so provider fires again
            stream = sr2.turn([TextBlock(text="again")])
            async for _ in stream:
                pass
            turn2_tools = mock_llm.last_request.tools

        assert turn1_tools is not None and len(turn1_tools) == 1, (
            "Turn 1 should have tools"
        )
        assert turn2_tools is not None and len(turn2_tools) == 1, (
            "Turn 2 must also have tools — execution_count reset missing in SR2.turn()"
        )

    @pytest.mark.asyncio
    async def test_tool_definitions_absent_when_no_tool_providers(
        self, reset_plugin_registries
    ):
        """D14 (negative): Without tool_providers, CompletionRequest.tools is None or empty."""
        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig
        from sr2.orchestrator import SR2

        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(
                            type="static",
                            config={"text": "You are a helpful assistant."},
                        )
                    ],
                ),
            ]
        )

        mock_llm = _MockLLM()

        from sr2.pipeline.resolvers.static import StaticResolver

        def _static_only(group: str):
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "static"
                ep.load.return_value = StaticResolver
                dist = MagicMock()
                dist.name = "sr2"
                ep.dist = dist
                return [ep]
            return []

        with patch("sr2.plugins.registry.entry_points", side_effect=_static_only):
            sr2 = SR2(
                pipeline_config=pipeline_config,
                llm={"default": mock_llm},
                token_counter=CharacterTokenCounter(),
            )

            stream = sr2.turn([TextBlock(text="hello")])
            async for _ in stream:
                pass

        # tools should be None or empty (engine emits None when list is empty)
        assert mock_llm.last_request is not None
        tools = mock_llm.last_request.tools
        assert not tools  # None or []
