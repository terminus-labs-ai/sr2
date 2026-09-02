"""Tests for run-context conditions that control layer participation."""

from __future__ import annotations

import pytest

from conftest import run_engine
from sr2.config.models import LayerConfig
from sr2.models import TextBlock
from sr2.orchestrator import _build_layer
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.token_counting import CharacterTokenCounter


class StubResolver:
    """A resolver that returns predetermined content."""

    def __init__(
        self,
        name: str = "stub_resolver",
        content: list | None = None,
        max_executions: int = 1,
    ) -> None:
        self.name = name
        self._content = content or []
        self.subscriptions = [
            EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)
        ]
        self.max_executions = max_executions
        self.execution_count = 0

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            content=self._content,
        )


def make_layer(
    name: str = "location",
    condition: str | None = None,
    run_context_provider=None,
    text: str = "location content",
) -> Layer:
    return Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[StubResolver(content=[TextBlock(text=text)])],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
        condition=condition,
        run_context_provider=run_context_provider,
    )


class TestConditionedLayers:
    @pytest.mark.asyncio
    async def test_included_when_context_value_is_non_empty(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"area": "fractured-roots"},
        )

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert layer.is_active() is True
        assert "location content" in "".join(
            block.text for block in result.request.system
        )
        assert layer.resolvers[0].execution_count == 1

    def test_whitespace_value_is_non_empty(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"area": " "},
        )

        assert layer.is_active() is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider",
        [
            lambda: {"area": ""},
            lambda: {"area": None},
            lambda: {"mode": "headless"},
            lambda: None,
            None,
        ],
    )
    async def test_excluded_when_context_does_not_supply_value(self, provider):
        layer = make_layer(condition="area", run_context_provider=provider)

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert layer.is_active() is False
        assert result.request.system is None
        assert layer.resolvers[0].execution_count == 0

    @pytest.mark.asyncio
    async def test_inactive_layer_is_absent_from_metrics(self):
        active_layer = make_layer(name="active", text="active content")
        inactive_layer = make_layer(
            name="conditioned",
            condition="area",
            run_context_provider=lambda: {"area": ""},
        )

        engine = PipelineEngine(
            layers=[active_layer, inactive_layer],
            token_counter=CharacterTokenCounter(),
        )
        result = await run_engine(engine, [])

        assert set(result.metrics.layers) == {"active"}
        assert inactive_layer.resolvers[0].execution_count == 0

    @pytest.mark.asyncio
    async def test_condition_is_read_again_on_later_turn(self):
        context = {"area": "fractured-roots"}
        layer = make_layer(condition="area", run_context_provider=lambda: context)
        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())

        first_result = await run_engine(engine, [])
        context["area"] = ""
        second_result = await run_engine(engine, [])

        assert first_result.request.system is not None
        assert second_result.request.system is None
        assert layer.resolvers[0].execution_count == 1


class TestUnconditionedLayers:
    @pytest.mark.asyncio
    async def test_unconditioned_layer_remains_active_without_provider(self):
        layer = make_layer(condition=None, run_context_provider=None)

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert layer.is_active() is True
        assert "location content" in "".join(
            block.text for block in result.request.system
        )
        assert layer.resolvers[0].execution_count == 1

    @pytest.mark.asyncio
    async def test_unconditioned_layer_ignores_provider_value(self):
        layer = make_layer(
            condition=None,
            run_context_provider=lambda: {"area": ""},
        )

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert result.request.system is not None


class TestLayerConfigCondition:
    def test_condition_defaults_to_none(self):
        config = LayerConfig(name="location", target="system", resolvers=[])

        assert config.condition is None

    def test_condition_accepts_run_context_key(self):
        config = LayerConfig(
            name="location",
            target="system",
            resolvers=[],
            condition="area",
        )

        assert config.condition == "area"

    def test_build_layer_passes_condition_and_run_context_provider(self):
        def provider():
            return {"area": "fractured-roots"}

        config = LayerConfig(
            name="location",
            target="system",
            resolvers=[],
            condition="area",
        )

        layer = _build_layer(
            config,
            CharacterTokenCounter(),
            Dependencies(run_context_provider=provider),
        )

        assert layer.condition == "area"
        assert layer.is_active() is True
