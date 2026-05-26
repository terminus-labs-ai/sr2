"""Tests for token_threshold event on Layer.check_budget() and LayerConfig.token_threshold_pct.

Step 1 of the summarization transformer spec:
  - LayerConfig gains token_threshold_pct: float | None = None
  - Layer.check_budget() fires a token_threshold event when used >= token_budget * token_threshold_pct
  - Existing overflow event behavior is unchanged
  - Both events can fire in the same check_budget() call
"""

import pytest

from sr2.config.models import LayerConfig
from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_layer(
    name: str = "test_layer",
    token_budget: int | None = 1000,
    token_threshold_pct: float | None = None,
    bus: EventBus | None = None,
) -> "Layer":
    """Build a minimal Layer for budget-checking tests."""
    from sr2.pipeline.layer import Layer

    return Layer(
        name=name,
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=token_budget,
        token_threshold_pct=token_threshold_pct,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=bus or EventBus(),
    )


def add_tokens(layer: "Layer", token_count: int, label: str = "x") -> None:
    """Add exactly token_count tokens worth of content to layer.

    CharacterTokenCounter: 4 chars = 1 token.
    """
    from sr2.pipeline.layer import Layer

    layer.add_content(
        ResolvedContent(
            resolver_name="stub",
            source_layer=layer.name,
            content=[TextBlock(text=label * (token_count * 4))],
        )
    )


def collect_events(bus: EventBus, event_name: str) -> list[Event]:
    """Subscribe to an event name and return the collected list (mutated in place)."""
    collected: list[Event] = []
    bus.subscribe(
        EventSubscription(event_name=event_name),
        lambda e: collected.append(e),
    )
    return collected


# ---------------------------------------------------------------------------
# FR2 / LayerConfig tests
# ---------------------------------------------------------------------------


class TestLayerConfigTokenThresholdPct:
    def test_token_threshold_pct_defaults_to_none(self):
        """FR2: LayerConfig.token_threshold_pct defaults to None."""
        config = LayerConfig(
            name="test",
            resolvers=[],
        )
        assert config.token_threshold_pct is None

    def test_token_threshold_pct_accepts_float(self):
        """FR2: LayerConfig.token_threshold_pct accepts a float value."""
        config = LayerConfig(
            name="test",
            resolvers=[],
            token_threshold_pct=0.8,
        )
        assert config.token_threshold_pct == 0.8

    def test_token_threshold_pct_accepts_none_explicitly(self):
        """FR2: LayerConfig.token_threshold_pct can be explicitly set to None."""
        config = LayerConfig(
            name="test",
            resolvers=[],
            token_threshold_pct=None,
        )
        assert config.token_threshold_pct is None


# ---------------------------------------------------------------------------
# FR1 / AC2: No threshold event when token_threshold_pct is None
# ---------------------------------------------------------------------------


class TestLayerCheckBudgetNoThreshold:
    def test_no_threshold_pct_never_fires_token_threshold_event(self):
        """AC2: Layer with token_threshold_pct=None never fires token_threshold event,
        even when heavily loaded."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(token_budget=100, token_threshold_pct=None, bus=bus)
        add_tokens(layer, 90)  # 90% full — would trigger at pct=0.8

        layer.check_budget()

        assert threshold_events == []

    def test_no_budget_no_events_fired(self):
        """Layer with no token_budget fires no events (neither threshold nor overflow)."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=None, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 10_000)  # enormous content, but no budget set

        layer.check_budget()

        assert threshold_events == []
        assert overflow_events == []

    def test_existing_config_no_threshold_pct_no_regression(self):
        """AC2: Existing layers without token_threshold_pct continue to work normally."""
        bus = EventBus()
        overflow_events = collect_events(bus, "overflow")
        threshold_events = collect_events(bus, "token_threshold")

        # Simulate a pre-existing layer config: only token_budget, no threshold_pct
        layer = make_layer(token_budget=10, token_threshold_pct=None, bus=bus)
        add_tokens(layer, 20)  # clearly over budget

        layer.check_budget()

        assert len(overflow_events) == 1
        assert threshold_events == []


# ---------------------------------------------------------------------------
# FR1 / AC1: threshold event fires at the right moment
# ---------------------------------------------------------------------------


class TestLayerCheckBudgetThresholdFires:
    def test_threshold_fires_when_at_threshold(self):
        """AC1 + FR1: Layer with token_threshold_pct=0.8 and budget=1000 fires
        token_threshold when content reaches exactly 800 tokens (the boundary)."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(name="layer_a", token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 800)  # exactly at threshold: 800 >= 1000 * 0.8

        layer.check_budget()

        assert len(threshold_events) == 1
        assert threshold_events[0].name == "token_threshold"

    def test_threshold_fires_below_overflow(self):
        """FR1: threshold fires even when content is below overflow boundary.

        800 tokens on a 1000-token budget with pct=0.8: threshold fires, overflow does not.
        """
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 800)  # at threshold, not overflowing

        layer.check_budget()

        assert len(threshold_events) == 1
        assert overflow_events == []

    def test_threshold_does_not_fire_below_threshold(self):
        """FR1: No token_threshold event when content is below the threshold."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 799)  # one token short of 800 threshold

        layer.check_budget()

        assert threshold_events == []

    def test_threshold_fires_above_threshold_but_below_overflow(self):
        """FR1: threshold fires for any used >= budget * pct, not only at exact boundary."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 950)  # 950 >= 800 threshold, but 950 <= 1000 (no overflow)

        layer.check_budget()

        assert len(threshold_events) == 1
        assert overflow_events == []


# ---------------------------------------------------------------------------
# FR1: Both threshold AND overflow can fire in the same call
# ---------------------------------------------------------------------------


class TestLayerCheckBudgetBothEvents:
    def test_both_threshold_and_overflow_fire_when_over_budget(self):
        """FR1: Both token_threshold and overflow fire in the same check_budget()
        call when content exceeds budget and threshold_pct is configured."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 1100)  # 1100 >= 800 (threshold) AND 1100 > 1000 (overflow)

        layer.check_budget()

        assert len(threshold_events) == 1
        assert len(overflow_events) == 1

    def test_overflow_without_threshold_pct_still_works(self):
        """FR1: overflow still fires correctly when no threshold_pct is configured."""
        bus = EventBus()
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=1000, token_threshold_pct=None, bus=bus)
        add_tokens(layer, 1100)

        layer.check_budget()

        assert len(overflow_events) == 1

    def test_threshold_fires_before_overflow_when_both_fire(self):
        """FR1/AC1: token_threshold is queued before overflow in the same call."""
        bus = EventBus()
        all_events: list[str] = []
        bus.subscribe("token_threshold", lambda e: all_events.append(e.name))
        bus.subscribe("overflow", lambda e: all_events.append(e.name))

        layer = make_layer(token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 1100)  # 1100 >= 800 (threshold) AND 1100 > 1000 (overflow)

        layer.check_budget()

        assert all_events == ["token_threshold", "overflow"]


# ---------------------------------------------------------------------------
# FR1: token_threshold event fields
# ---------------------------------------------------------------------------


class TestLayerThresholdEventFields:
    def test_threshold_event_has_correct_name_and_phase(self):
        """FR1: token_threshold event has name='token_threshold' and phase=COMPLETED."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(name="my_layer", token_budget=1000, token_threshold_pct=0.8, bus=bus)
        add_tokens(layer, 900)

        layer.check_budget()

        assert len(threshold_events) == 1
        event = threshold_events[0]
        assert event.name == "token_threshold"
        assert event.phase == EventPhase.COMPLETED

    def test_threshold_event_source_layer_matches_layer_name(self):
        """FR1: token_threshold event source_layer equals the layer's name."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(
            name="conversation_layer", token_budget=1000, token_threshold_pct=0.8, bus=bus
        )
        add_tokens(layer, 900)

        layer.check_budget()

        assert threshold_events[0].source_layer == "conversation_layer"

    def test_threshold_event_data_contains_used_budget_pct(self):
        """FR1: token_threshold event data dict contains 'used', 'budget', and 'pct'."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(
            name="mem_layer", token_budget=1000, token_threshold_pct=0.75, bus=bus
        )
        add_tokens(layer, 800)

        layer.check_budget()

        assert len(threshold_events) == 1
        data = threshold_events[0].data
        assert "used" in data
        assert "budget" in data
        assert "pct" in data

    def test_threshold_event_data_values_are_correct(self):
        """FR1: token_threshold event data values match the layer state at call time."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        layer = make_layer(
            name="data_check_layer", token_budget=1000, token_threshold_pct=0.8, bus=bus
        )
        add_tokens(layer, 900)

        layer.check_budget()

        data = threshold_events[0].data
        assert data["used"] == 900
        assert data["budget"] == 1000
        assert data["pct"] == 0.8

    def test_overflow_event_fields_unchanged(self):
        """FR1: overflow event retains its existing fields (no regression)."""
        bus = EventBus()
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(
            name="overflow_layer", token_budget=100, token_threshold_pct=0.8, bus=bus
        )
        add_tokens(layer, 200)

        layer.check_budget()

        assert len(overflow_events) == 1
        event = overflow_events[0]
        assert event.name == "overflow"
        assert event.phase == EventPhase.COMPLETED
        assert event.source_layer == "overflow_layer"


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------


class TestLayerThresholdBoundary:
    def test_boundary_exactly_at_threshold_fires(self):
        """FR1: used >= threshold is inclusive — fires at exactly the threshold value."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        # budget=100, pct=0.5 → threshold=50 tokens
        layer = make_layer(token_budget=100, token_threshold_pct=0.5, bus=bus)
        add_tokens(layer, 50)  # exactly at boundary

        layer.check_budget()

        assert len(threshold_events) == 1

    def test_one_token_below_threshold_does_not_fire(self):
        """FR1: used < threshold does not fire token_threshold event."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")

        # budget=100, pct=0.5 → threshold=50 tokens
        layer = make_layer(token_budget=100, token_threshold_pct=0.5, bus=bus)
        add_tokens(layer, 49)  # one below boundary

        layer.check_budget()

        assert threshold_events == []

    def test_threshold_pct_of_1_fires_only_at_full_budget(self):
        """Edge case: pct=1.0 means threshold fires when budget is fully consumed
        (same boundary as overflow)."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=100, token_threshold_pct=1.0, bus=bus)
        add_tokens(layer, 100)  # exactly at 100% — threshold fires (>=), overflow does NOT (not >)

        layer.check_budget()

        assert len(threshold_events) == 1
        assert overflow_events == []  # overflow is strictly >, not >=

    def test_zero_content_no_events(self):
        """No events fire when layer has no content, regardless of threshold_pct."""
        bus = EventBus()
        threshold_events = collect_events(bus, "token_threshold")
        overflow_events = collect_events(bus, "overflow")

        layer = make_layer(token_budget=1000, token_threshold_pct=0.8, bus=bus)
        # No content added

        layer.check_budget()

        assert threshold_events == []
        assert overflow_events == []

    def test_different_threshold_pcts_fire_at_correct_amounts(self):
        """FR1: Threshold amount is correctly computed from budget * pct."""
        for pct, load, should_fire in [
            (0.5, 50, True),   # exactly at 50%
            (0.5, 49, False),  # one below 50%
            (0.9, 90, True),   # exactly at 90%
            (0.9, 89, False),  # one below 90%
        ]:
            bus = EventBus()
            threshold_events = collect_events(bus, "token_threshold")

            layer = make_layer(token_budget=100, token_threshold_pct=pct, bus=bus)
            add_tokens(layer, load)
            layer.check_budget()

            if should_fire:
                assert len(threshold_events) == 1, (
                    f"Expected token_threshold to fire at pct={pct}, load={load}"
                )
            else:
                assert threshold_events == [], (
                    f"Expected no token_threshold at pct={pct}, load={load}"
                )
