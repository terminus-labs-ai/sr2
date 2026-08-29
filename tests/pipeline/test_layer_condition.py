"""Tests for obsidian-8h0z: layer conditions.

A layer with ``condition="area"`` is included in the pipeline only when the
run context supplies a non-empty string for that key.  Layers without a
condition behave exactly as before (no regression).

Covers:
  AC1: condition layer included when RunContext.area is a non-empty string
  AC2: same layer excluded when area is "", None, or absent
  AC3: layers without a condition behave exactly as today (no regression)
  AC4: tests cover included / excluded / no-condition
"""

from __future__ import annotations

import pytest

from conftest import run_engine
from sr2.config.models import LayerConfig
from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubResolver:
    """A resolver that returns predetermined content."""

    def __init__(
        self,
        name: str = "stub_resolver",
        content: list | None = None,
        max_executions: int = 1,
    ):
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


# ---------------------------------------------------------------------------
# AC1 + AC2: condition layer included / excluded
# ---------------------------------------------------------------------------


class TestLayerConditionIncluded:
    """AC1: layer with condition area-is-not-empty is included when area is a non-empty string."""

    @pytest.mark.asyncio
    async def test_included_when_area_non_empty(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"area": "fractured-roots"},
        )
        assert layer.is_active() is True

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        system_text = "".join(b.text for b in result.request.system)
        assert "location content" in system_text
        assert layer.resolvers[0].execution_count == 1

    @pytest.mark.asyncio
    async def test_included_when_area_whitespace_is_not_enough(self):
        """Whitespace-only area is not a non-empty meaningful value — but the
        spec says 'non-empty string', so whitespace-only counts as non-empty."""
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"area": " "},
        )
        assert layer.is_active() is True


class TestLayerConditionExcluded:
    """AC2: same layer is excluded when area is '', None, or absent."""

    @pytest.mark.asyncio
    async def test_excluded_when_area_empty_string(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"area": ""},
        )
        assert layer.is_active() is False

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert result.request.system is None
        assert layer.resolvers[0].execution_count == 0

    @pytest.mark.asyncio
    async def test_excluded_when_area_none(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"area": None},
        )
        assert layer.is_active() is False

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert result.request.system is None
        assert layer.resolvers[0].execution_count == 0

    @pytest.mark.asyncio
    async def test_excluded_when_area_absent(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: {"mode": "headless"},
        )
        assert layer.is_active() is False

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert result.request.system is None
        assert layer.resolvers[0].execution_count == 0

    @pytest.mark.asyncio
    async def test_excluded_when_run_context_is_none(self):
        layer = make_layer(
            condition="area",
            run_context_provider=lambda: None,
        )
        assert layer.is_active() is False

    @pytest.mark.asyncio
    async def test_excluded_when_no_provider(self):
        """Condition set but no provider wired → cannot be satisfied → excluded."""
        layer = make_layer(condition="area", run_context_provider=None)
        assert layer.is_active() is False

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        assert result.request.system is None
        assert layer.resolvers[0].execution_count == 0


# ---------------------------------------------------------------------------
# AC3: no-condition layers behave exactly as today
# ---------------------------------------------------------------------------


class TestNoConditionRegression:
    """AC3: layers without a condition behave exactly as today (no regression)."""

    @pytest.mark.asyncio
    async def test_no_condition_always_active(self):
        layer = make_layer(condition=None, run_context_provider=None)
        assert layer.is_active() is True

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        system_text = "".join(b.text for b in result.request.system)
        assert "location content" in system_text
        assert layer.resolvers[0].execution_count == 1

    @pytest.mark.asyncio
    async def test_no_condition_ignores_provider(self):
        """A provider is irrelevant when no condition is set."""
        layer = make_layer(
            condition=None,
            run_context_provider=lambda: {"area": ""},
        )
        assert layer.is_active() is True

        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())
        result = await run_engine(engine, [])

        system_text = "".join(b.text for b in result.request.system)
        assert "location content" in system_text

    @pytest.mark.asyncio
    async def test_condition_flips_between_turns(self):
        """The same layer instance is active on one turn and excluded on the next."""
        ctx = {"area": "fractured-roots"}
        layer = make_layer(condition="area", run_context_provider=lambda: ctx)
        engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())

        # Turn 1: area present → included
        result1 = await run_engine(engine, [])
        assert result1.request.system is not None
        assert layer.resolvers[0].execution_count == 1

        # Area disappears (DM turn) → excluded
        ctx["area"] = ""
        result2 = await run_engine(engine, [])
        assert result2.request.system is None
        # Resolver did not fire again on the excluded turn
        assert layer.resolvers[0].execution_count == 1


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestLayerConfigCondition:
    """LayerConfig.condition parses and defaults to None."""

    def test_condition_defaults_to_none(self):
        lc = LayerConfig(name="x", target="system", resolvers=[])
        assert lc.condition is None

    def test_condition_accepts_key_name(self):
        lc = LayerConfig(name="location", target="system", resolvers=[], condition="area")
        assert lc.condition == "area"
