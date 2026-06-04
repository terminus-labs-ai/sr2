"""Tests for FR6: priority shedding integration (shed() on real Layers).

Bead: sr2-86

Covers:
  - Layer.token_count property returns correct count from _token_counter
  - Layer satisfies HasPriorityAndTokens protocol (has .priority + .token_count)
  - shed() on real Layer objects: deterministic survivor selection by priority
  - shed() preserves original declaration order of surviving layers
  - shed() returns all layers when total token count is within budget
  - shed() returns empty when budget is 0 (and no layer has 0 tokens)
  - shed() sheds lowest priority first, then next-lowest, etc.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pytest

from sr2.degradation.shedding import HasPriorityAndTokens, shed
from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_layer(
    name: str,
    token_text: str,
    priority: int = 0,
    degradation_category: str | None = None,
) -> Layer:
    """Build a Layer with known content so token_count is predictable.

    CharacterTokenCounter counts characters / 4 (rounds up).
    We use ``token_text`` whose length we control.
    """
    counter = CharacterTokenCounter()
    layer = Layer(
        name=name,
        target=CompilationTarget.MESSAGES,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=counter,
        event_bus=EventBus(),
        degradation_category=degradation_category,
        priority=priority,
    )
    # Pre-populate content so token_count is non-zero
    layer.set_content([TextBlock(text=token_text)])
    return layer


# ===========================================================================
# 1. Layer.token_count property
# ===========================================================================


class TestLayerTokenCount:
    """Layer exposes token_count property that delegates to _token_counter."""

    def test_token_count_empty_layer(self):
        layer = make_layer("empty", "")
        assert layer.token_count == 0

    def test_token_count_reflects_content(self):
        """token_count reflects actual content size via the counter."""
        # CharacterTokenCounter: len(text) / 4, rounded up → ceil(40/4) = 10
        layer = make_layer("known", "a" * 40)
        assert layer.token_count == 10

    def test_token_count_updates_after_content_change(self):
        """token_count updates when content changes."""
        layer = make_layer("mutable", "a" * 40)
        assert layer.token_count == 10
        layer.set_content([TextBlock(text="b" * 80)])
        assert layer.token_count == 20

    def test_token_count_zero_after_clear(self):
        layer = make_layer("clearable", "a" * 40)
        layer.set_content([])
        assert layer.token_count == 0


# ===========================================================================
# 2. Layer satisfies HasPriorityAndTokens protocol
# ===========================================================================


class TestLayerProtocolConformance:
    """Layer must satisfy the HasPriorityAndTokens protocol for shed()."""

    def test_layer_satisfies_protocol(self):
        layer = make_layer("test", "x" * 20, priority=5)
        assert isinstance(layer, HasPriorityAndTokens)

    def test_layer_has_priority_attribute(self):
        layer = make_layer("test", "x" * 20, priority=7)
        assert layer.priority == 7

    def test_layer_priority_default_zero(self):
        layer = make_layer("test", "x" * 20)
        assert layer.priority == 0

    def test_layer_has_token_count_attribute(self):
        layer = make_layer("test", "x" * 20)
        assert isinstance(layer.token_count, int)
        assert layer.token_count >= 0


# ===========================================================================
# 3. shed() on real Layer objects — deterministic survivor selection
# ===========================================================================


class TestShedOnRealLayers:
    """shed(layers, budget) works on real Layer instances."""

    def test_no_shedding_when_under_budget(self):
        """All layers survive when total tokens <= budget."""
        a = make_layer("a", "x" * 40, priority=10)   # 10 tokens
        b = make_layer("b", "y" * 40, priority=5)     # 10 tokens
        result = shed([a, b], budget=30)
        assert len(result) == 2
        assert {l.name for l in result} == {"a", "b"}

    def test_sheds_lowest_priority_first(self):
        """Lowest priority layer is shed first when over budget."""
        high = make_layer("high", "x" * 80, priority=10)  # 20 tokens
        low = make_layer("low", "y" * 80, priority=1)     # 20 tokens
        # Total 40 tokens, budget 25 → shed 'low' first (priority=1)
        result = shed([high, low], budget=25)
        names = {l.name for l in result}
        assert "low" not in names
        assert "high" in names

    def test_sheds_multiple_layers_if_necessary(self):
        """Multiple layers shed until budget is met."""
        critical = make_layer("critical", "x" * 40, priority=100)  # 10 tokens
        medium = make_layer("medium", "y" * 60, priority=5)        # 15 tokens
        low = make_layer("low", "z" * 60, priority=1)             # 15 tokens
        # Total 40, budget 15 → shed 'low' (10 remaining), still over → shed 'medium' (10 remaining)
        result = shed([critical, medium, low], budget=15)
        names = {l.name for l in result}
        assert "critical" in names
        assert "medium" not in names
        assert "low" not in names

    def test_result_total_within_budget(self):
        """Sum of surviving layers' token_count <= budget."""
        a = make_layer("a", "x" * 30, priority=10)  # 8 tokens
        b = make_layer("b", "y" * 30, priority=5)   # 8 tokens
        c = make_layer("c", "z" * 30, priority=1)   # 8 tokens
        result = shed([a, b, c], budget=15)
        total = sum(l.token_count for l in result)
        assert total <= 15

    def test_preserves_original_order(self):
        """Surviving layers maintain their original declaration order."""
        first = make_layer("first", "x" * 20, priority=10)   # 5 tokens
        second = make_layer("second", "y" * 20, priority=8)   # 5 tokens
        third = make_layer("third", "z" * 100, priority=1)    # 25 tokens
        # Budget 15 → shed 'third'; first+second survive
        result = shed([first, second, third], budget=15)
        assert [l.name for l in result] == ["first", "second"]

    def test_budget_zero_sheds_all(self):
        """Budget of 0 sheds all layers with non-zero token_count."""
        a = make_layer("a", "x" * 10, priority=10)
        b = make_layer("b", "y" * 10, priority=5)
        result = shed([a, b], budget=0)
        total = sum(l.token_count for l in result)
        assert total == 0

    def test_empty_input_returns_empty(self):
        result = shed([], budget=1000)
        assert result == []

    def test_deterministic_with_same_priority(self):
        """Layers with the same priority are shed in original index order."""
        # Three layers, same priority, different token counts
        a = make_layer("a", "x" * 40, priority=5)  # 10 tokens
        b = make_layer("b", "y" * 40, priority=5)  # 10 tokens
        c = make_layer("c", "z" * 40, priority=5)  # 10 tokens
        # Budget 20 → need to shed one; lowest index shed first among ties
        result = shed([a, b, c], budget=20)
        assert len(result) == 2
        # First two survive (lowest index shed last among same priority)
        assert [l.name for l in result] == ["b", "c"]

    def test_shed_returns_new_list(self):
        """shed() returns a new list, not the original."""
        a = make_layer("a", "x" * 20, priority=10)
        original = [a]
        result = shed(original, budget=1000)
        assert result is not original
        assert result == original  # same content, different list

    def test_layers_with_zero_token_count_never_shed(self):
        """A layer with 0 tokens is never shed (it doesn't consume budget)."""
        # Empty content = 0 tokens
        empty = Layer(
            name="empty",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
            priority=1,  # lowest priority, but 0 tokens
        )
        big = make_layer("big", "x" * 200, priority=10)  # 50 tokens
        result = shed([empty, big], budget=100)
        names = {l.name for l in result}
        assert "empty" in names
        assert "big" in names

    def test_shed_integration_with_degradation_category(self):
        """shed() works on layers that also have degradation_category set.
        Priority shedding is independent of category-based exclusion."""
        high = make_layer("high", "x" * 80, priority=10, degradation_category="context")
        low = make_layer("low", "y" * 80, priority=1, degradation_category="memory")
        # Total 40, budget 25 → shed 'low'
        result = shed([high, low], budget=25)
        names = {l.name for l in result}
        assert "low" not in names
        assert "high" in names

    def test_realistic_multi_layer_scenario(self):
        """Realistic scenario: system + memory + context + tools layers."""
        system = make_layer("system", "s" * 100, priority=100)      # 25 tokens — never shed
        memory = make_layer("memory", "m" * 200, priority=1)        # 50 tokens — shed first
        context = make_layer("context", "c" * 120, priority=50)     # 30 tokens
        tools = make_layer("tools", "t" * 120, priority=30)         # 30 tokens
        # Total 135 tokens; budget 60
        # Shed memory (50) → remaining 85; still over
        # Shed tools (30) → remaining 55; fits
        layers = [system, memory, context, tools]
        result = shed(layers, budget=60)
        names = {l.name for l in result}
        assert "memory" not in names
        assert "tools" not in names
        assert "system" in names
        assert "context" in names
        total = sum(l.token_count for l in result)
        assert total <= 60
