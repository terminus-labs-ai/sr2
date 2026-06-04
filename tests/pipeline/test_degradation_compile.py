"""Tests for FR5: compile-time degradation effect + step-down/recompile loop.

Bead: sr2-84

Covers:
  FR5:  Layers whose degradation_category is not in ladder.active_categories()
        are excluded from the compiled CompletionRequest.
  FR7:  Ladder resets to FULL at turn start.
  FR8:  No degradation config → byte-identical output to pre-change baseline.
  D5:   Compile → if over budget → step_down → recompile → until fits or bottom.

All tests exercise the engine's _compile_request with a DegradationLadder.
"""

from __future__ import annotations

import pytest

from sr2.config.models import (
    ConfigError,
    DegradationConfig,
    DegradationLevelConfig,
    DegradationTriggerConfig,
)
from sr2.degradation.ladder import DegradationLadder
from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubResolver:
    """Resolver that returns predetermined content."""

    def __init__(
        self,
        name: str = "stub",
        content: list[TextBlock] | None = None,
        subscriptions: list[EventSubscription] | None = None,
        max_executions: int = 1,
    ):
        self.name = name
        self._content = content or [TextBlock(text=f"from {name}")]
        self.subscriptions = subscriptions or []
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
    name: str,
    target: CompilationTarget,
    content: list[TextBlock],
    degradation_category: str | None = None,
    priority: int = 0,
    token_budget: int | None = None,
) -> Layer:
    """Build a Layer with pre-set content (no resolver needed)."""
    return Layer(
        name=name,
        target=target,
        position=AppendStrategy(),
        token_budget=token_budget,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
        degradation_category=degradation_category,
        priority=priority,
    )


def make_config(levels: list[dict]) -> DegradationConfig:
    """Build a DegradationConfig from level dicts."""
    return DegradationConfig(
        levels=[
            DegradationLevelConfig(name=lv["name"], keep_categories=lv["cats"])
            for lv in levels
        ]
    )


def count_request_tokens(request: CompletionRequest) -> int:
    """Count total tokens in a CompletionRequest using CharacterTokenCounter."""
    counter = CharacterTokenCounter()
    total = 0
    if request.system:
        total += counter.count(request.system)
    if request.messages:
        for msg in request.messages:
            total += counter.count(msg.content)
    return total


# ===========================================================================
# 1. Basic compile-time exclusion (FR5)
# ===========================================================================


class TestCompileTimeExclusion:
    """Layers excluded when their category is not active at current ladder level."""

    def test_layer_with_no_category_always_included(self):
        """Layers with degradation_category=None are always kept (structural)."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory"]},
                {"name": "minimal", "cats": []},
            ])
        )
        ladder.step_down()  # now at minimal — no categories active

        # Structural layer (no category)
        structural = make_layer(
            "system_prompt", CompilationTarget.SYSTEM,
            [TextBlock(text="System prompt")],
            degradation_category=None,
        )
        structural.set_content([TextBlock(text="System prompt")])

        # Degradeable layer
        memory_layer = make_layer(
            "memory", CompilationTarget.MESSAGES,
            [TextBlock(text="memory content")],
            degradation_category="memory",
        )
        memory_layer.set_content([TextBlock(text="memory content")])

        engine = PipelineEngine(
            layers=[structural, memory_layer],
            token_counter=counter,
        )
        engine._ladder = ladder
        request = engine._compile_request()

        # System prompt should be present (structural, no category)
        assert request.system is not None
        assert any(b.text == "System prompt" for b in request.system)

        # Memory content should be absent (category not active)
        all_msg_text = ""
        if request.messages:
            for msg in request.messages:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        all_msg_text += block.text
        assert "memory content" not in all_msg_text

    def test_layer_with_active_category_included(self):
        """Layers whose category IS in active_categories are included."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "context"]},
            ])
        )

        memory_layer = make_layer(
            "memory", CompilationTarget.SYSTEM,
            [TextBlock(text="memory block")],
            degradation_category="memory",
        )
        memory_layer.set_content([TextBlock(text="memory block")])

        context_layer = make_layer(
            "context", CompilationTarget.SYSTEM,
            [TextBlock(text="context block")],
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="context block")])

        engine = PipelineEngine(
            layers=[memory_layer, context_layer],
            token_counter=counter,
        )
        engine._ladder = ladder
        request = engine._compile_request()

        assert request.system is not None
        texts = [b.text for b in request.system]
        assert "memory block" in texts
        assert "context block" in texts

    def test_layer_excluded_when_category_inactive(self):
        """Layer excluded when its category is not in active_categories at current level."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "tools", "context"]},
                {"name": "reduced", "cats": ["context"]},
                {"name": "minimal", "cats": ["context"]},
            ])
        )
        ladder.step_down()  # reduced: only "context" active

        memory_layer = make_layer(
            "memory", CompilationTarget.SYSTEM,
            [TextBlock(text="memory block")],
            degradation_category="memory",
        )
        memory_layer.set_content([TextBlock(text="memory block")])

        context_layer = make_layer(
            "context", CompilationTarget.SYSTEM,
            [TextBlock(text="context block")],
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="context block")])

        tools_layer = make_layer(
            "tools", CompilationTarget.SYSTEM,
            [TextBlock(text="tools block")],
            degradation_category="tools",
        )
        tools_layer.set_content([TextBlock(text="tools block")])

        engine = PipelineEngine(
            layers=[memory_layer, context_layer, tools_layer],
            token_counter=counter,
        )
        engine._ladder = ladder
        request = engine._compile_request()

        texts = [b.text for b in request.system]
        assert "context block" in texts
        assert "memory block" not in texts
        assert "tools block" not in texts

    def test_multiple_layers_same_category_all_excluded(self):
        """When a category is inactive, ALL layers with that category are excluded."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ])
        )
        ladder.step_down()  # only context active

        mem1 = make_layer(
            "memory_recent", CompilationTarget.MESSAGES,
            [TextBlock(text="recent memory")],
            degradation_category="memory",
        )
        mem1.set_content([TextBlock(text="recent memory")])

        mem2 = make_layer(
            "memory_long", CompilationTarget.MESSAGES,
            [TextBlock(text="long-term memory")],
            degradation_category="memory",
        )
        mem2.set_content([TextBlock(text="long-term memory")])

        engine = PipelineEngine(
            layers=[mem1, mem2],
            token_counter=counter,
        )
        engine._ladder = ladder
        request = engine._compile_request()

        all_text = ""
        if request.messages:
            for msg in request.messages:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        all_text += block.text
        assert "recent memory" not in all_text
        assert "long-term memory" not in all_text

    def test_tools_layer_excluded_when_tools_category_inactive(self):
        """A tools-target layer with degradation_category='tools' is excluded when 'tools' is inactive."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["tools", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ])
        )
        ladder.step_down()

        tools_layer = make_layer(
            "tools", CompilationTarget.TOOLS,
            [TextBlock(text="tool_def")],
            degradation_category="tools",
        )
        from sr2.models import ToolDefinition
        tools_layer.add_tool_definitions([
            ToolDefinition(
                name="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ])

        engine = PipelineEngine(layers=[tools_layer], token_counter=counter)
        engine._ladder = ladder
        request = engine._compile_request()

        assert request.tools is None or len(request.tools) == 0


# ===========================================================================
# 2. No degradation config → byte-identical (FR8)
# ===========================================================================


class TestNoDegradationConfig:
    """When no degradation config is present, compile output is identical to baseline."""

    def test_no_ladder_all_layers_compiled(self):
        """Engine without a ladder compiles all layers regardless of category."""
        counter = CharacterTokenCounter()

        memory_layer = make_layer(
            "memory", CompilationTarget.SYSTEM,
            [TextBlock(text="memory block")],
            degradation_category="memory",
        )
        memory_layer.set_content([TextBlock(text="memory block")])

        context_layer = make_layer(
            "context", CompilationTarget.SYSTEM,
            [TextBlock(text="context block")],
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="context block")])

        structural = make_layer(
            "system_prompt", CompilationTarget.SYSTEM,
            [TextBlock(text="system prompt")],
            degradation_category=None,
        )
        structural.set_content([TextBlock(text="system prompt")])

        engine = PipelineEngine(
            layers=[structural, memory_layer, context_layer],
            token_counter=counter,
        )
        # No ladder set — should be None
        assert engine._ladder is None

        request = engine._compile_request()

        # All content should be present
        texts = [b.text for b in request.system]
        assert "memory block" in texts
        assert "context block" in texts
        assert "system prompt" in texts

    def test_no_ladder_same_output_with_and_without_categories(self):
        """Two engines — one with categorized layers, one without — produce same output when no ladder."""
        counter = CharacterTokenCounter()

        # Engine A: layers with categories
        layer_a = make_layer(
            "l1", CompilationTarget.SYSTEM,
            [TextBlock(text="abc")],
            degradation_category="memory",
        )
        layer_a.set_content([TextBlock(text="abc")])
        engine_a = PipelineEngine(layers=[layer_a], token_counter=counter)

        # Engine B: identical layers without categories
        layer_b = make_layer(
            "l1", CompilationTarget.SYSTEM,
            [TextBlock(text="abc")],
            degradation_category=None,
        )
        layer_b.set_content([TextBlock(text="abc")])
        engine_b = PipelineEngine(layers=[layer_b], token_counter=counter)

        req_a = engine_a._compile_request()
        req_b = engine_b._compile_request()

        assert req_a.system == req_b.system
        assert req_a.messages == req_b.messages
        assert req_a.tools == req_b.tools


# ===========================================================================
# 3. Step-down recompile loop (D5)
# ===========================================================================


class TestStepDownRecompileLoop:
    """D5: compile → if over budget → step_down → recompile → until fits or bottom."""

    @pytest.mark.asyncio
    async def test_compile_loop_steps_down_when_over_budget(self):
        """When compile produces a request over the engine's token_budget, the ladder
        steps down and recompiles until under budget."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
                {"name": "minimal", "cats": ["context"]},
            ])
        )

        # Memory layer: 20 tokens (80 chars)
        memory_layer = make_layer(
            "memory", CompilationTarget.MESSAGES,
            [TextBlock(text="m" * 80)],
            degradation_category="memory",
        )
        memory_layer.set_content([TextBlock(text="m" * 80)])

        # Context layer: 10 tokens (40 chars)
        context_layer = make_layer(
            "context", CompilationTarget.MESSAGES,
            [TextBlock(text="c" * 40)],
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="c" * 40)])

        bus = EventBus()
        for layer in [memory_layer, context_layer]:
            layer.wire(bus, layer._provenance_store, None)

        engine = PipelineEngine(
            layers=[memory_layer, context_layer],
            token_counter=counter,
            token_budget=15,  # 30 tokens total at FULL → over budget
            bus=bus,
        )
        engine._ladder = ladder

        # Run compile loop
        result = await engine._compile_with_degradation()

        # After stepping down, memory should be excluded (10 tokens from context only)
        assert result.request.messages is not None
        all_text = ""
        for msg in result.request.messages:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    all_text += block.text
        assert "m" not in all_text  # memory excluded
        assert "c" in all_text  # context kept

        # Ladder should have stepped down
        assert not ladder.is_at_full()

    @pytest.mark.asyncio
    async def test_compile_loop_does_not_step_down_when_under_budget(self):
        """When under budget, no step-down occurs."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ])
        )

        context_layer = make_layer(
            "context", CompilationTarget.MESSAGES,
            [TextBlock(text="c" * 40)],  # 10 tokens
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="c" * 40)])

        bus = EventBus()
        context_layer.wire(bus, context_layer._provenance_store, None)

        engine = PipelineEngine(
            layers=[context_layer],
            token_counter=counter,
            token_budget=100,  # plenty of budget
            bus=bus,
        )
        engine._ladder = ladder

        result = await engine._compile_with_degradation()

        # Ladder should remain at FULL
        assert ladder.is_at_full()

    @pytest.mark.asyncio
    async def test_compile_loop_stops_at_bottom_level(self):
        """Even if still over budget at the most-degraded level, the loop stops."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["context"]},
                {"name": "minimal", "cats": ["context"]},
            ])
        )

        # Context layer: 100 tokens (400 chars) — over any reasonable budget
        context_layer = make_layer(
            "context", CompilationTarget.MESSAGES,
            [TextBlock(text="x" * 400)],
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="x" * 400)])

        bus = EventBus()
        context_layer.wire(bus, context_layer._provenance_store, None)

        engine = PipelineEngine(
            layers=[context_layer],
            token_counter=counter,
            token_budget=10,  # 100 tokens >> 10 budget
            bus=bus,
        )
        engine._ladder = ladder

        result = await engine._compile_with_degradation()

        # Should have reached the bottom level
        assert ladder.current_level == 1  # minimal level
        # Request should still be produced (not raise)
        assert result.request is not None

    @pytest.mark.asyncio
    async def test_compile_loop_bounded_by_level_count(self):
        """The step-down/recompile loop runs at most N times where N = len(levels) - 1."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "tools", "context"]},
                {"name": "level1", "cats": ["memory", "context"]},
                {"name": "level2", "cats": ["memory"]},
                {"name": "level3", "cats": ["context"]},
                {"name": "minimal", "cats": []},
            ])
        )

        # All layers over budget
        layers = []
        for cat in ["memory", "tools", "context"]:
            layer = make_layer(
                cat, CompilationTarget.MESSAGES,
                [TextBlock(text=f"{cat}" * 100)],  # lots of tokens
                degradation_category=cat,
            )
            layer.set_content([TextBlock(text=f"{cat}" * 100)])
            layers.append(layer)

        bus = EventBus()
        for layer in layers:
            layer.wire(bus, layer._provenance_store, None)

        engine = PipelineEngine(
            layers=layers,
            token_counter=counter,
            token_budget=1,  # tiny budget — never fits
            bus=bus,
        )
        engine._ladder = ladder

        result = await engine._compile_with_degradation()

        # Ladder should be at bottom (index 4)
        assert ladder.current_level == 4
        assert result.request is not None


# ===========================================================================
# 4. Ladder reset at turn start (FR7)
# ===========================================================================


class TestLadderResetAtTurnStart:
    """FR7: Ladder resets to FULL at the start of every turn."""

    @pytest.mark.asyncio
    async def test_ladder_resets_at_start_turn(self):
        """start_turn() resets the ladder to FULL."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ])
        )

        layer = make_layer(
            "context", CompilationTarget.MESSAGES,
            [TextBlock(text="hello")],
            degradation_category="context",
        )

        bus = EventBus()
        layer.wire(bus, layer._provenance_store, None)

        engine = PipelineEngine(
            layers=[layer],
            token_counter=counter,
            token_budget=200,
            bus=bus,
        )
        engine._ladder = ladder

        # Manually step down the ladder (simulating a previous degraded turn)
        ladder.step_down()
        assert not ladder.is_at_full()

        # Start a new turn — should reset ladder
        await engine.start_turn(turn_seq=1)

        assert ladder.is_at_full()


# ===========================================================================
# 5. Metrics warning on step-down
# ===========================================================================


class TestDegradationMetrics:
    """When degradation causes a step-down, a warning appears in metrics."""

    @pytest.mark.asyncio
    async def test_metrics_warn_on_step_down(self):
        """PipelineMetrics.warnings includes a degradation note when step-down occurred."""
        counter = CharacterTokenCounter()
        ladder = DegradationLadder.from_config(
            make_config([
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ])
        )

        memory_layer = make_layer(
            "memory", CompilationTarget.MESSAGES,
            [TextBlock(text="m" * 80)],  # 20 tokens
            degradation_category="memory",
        )
        memory_layer.set_content([TextBlock(text="m" * 80)])

        context_layer = make_layer(
            "context", CompilationTarget.MESSAGES,
            [TextBlock(text="c" * 40)],  # 10 tokens
            degradation_category="context",
        )
        context_layer.set_content([TextBlock(text="c" * 40)])

        bus = EventBus()
        for layer in [memory_layer, context_layer]:
            layer.wire(bus, layer._provenance_store, None)

        engine = PipelineEngine(
            layers=[memory_layer, context_layer],
            token_counter=counter,
            token_budget=15,  # 30 total > 15 budget → step down
            bus=bus,
        )
        engine._ladder = ladder

        result = await engine._compile_with_degradation()

        # Should have a degradation warning
        assert any("degradation" in w.lower() for w in result.metrics.warnings)
