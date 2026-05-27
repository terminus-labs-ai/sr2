"""Tests for sr2.degradation — ladder, priority shedding, circuit breaker, fallback.

Bead: obsidian-t9t.5 (Agent A — test writer)

Covers:
  1. DegradationLadder — 5 levels, step-down, active providers per level
  2. Priority-based shedding — shed lowest-priority layers until under token budget
  3. CircuitBreaker — opens after N failures, rejects when open, half-opens after timeout
  4. Fallback content — cached / none / static modes

NOTE: All tests must FAIL until implementation exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Helpers — lightweight data types used as test inputs
# These mirror what the implementation is expected to expose.
# ---------------------------------------------------------------------------


@dataclass
class ContentLayer:
    """Minimal representation of a pipeline layer for shedding tests."""

    name: str
    priority: int          # lower number = lower priority (shed first)
    token_count: int
    content: str = ""


# ===========================================================================
# 1. DegradationLadder
# ===========================================================================


class TestDegradationLadder:
    """DegradationLadder tracks which of 5 degradation levels is active
    and knows which providers are enabled at each level.
    """

    def test_ladder_has_five_levels(self):
        """Ladder exposes exactly 5 levels."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        levels = list(DegradationLevel)
        assert len(levels) == 5

    def test_level_names_include_full_and_system_prompt_only(self):
        """Ladder must include a 'full' level and a 'system_prompt_only' level."""
        from sr2.degradation.ladder import DegradationLevel

        names = {level.name.lower() for level in DegradationLevel}
        assert "full" in names
        assert "system_prompt_only" in names

    def test_default_level_is_full(self):
        """A freshly created ladder starts at the 'full' level."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        assert ladder.current_level == DegradationLevel.FULL

    def test_step_down_from_full_moves_to_next_level(self):
        """step_down() moves the ladder one level below FULL."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        ladder.step_down()
        assert ladder.current_level != DegradationLevel.FULL

    def test_step_down_is_monotonically_more_degraded(self):
        """Each step_down() call reduces capability (level index increases)."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        levels_seen = [ladder.current_level]
        for _ in range(4):
            ladder.step_down()
            levels_seen.append(ladder.current_level)

        # Verify ordering is strictly non-decreasing in severity
        # (i.e., the int value or ordering index must increase or stay same)
        all_levels = list(DegradationLevel)
        indices = [all_levels.index(lv) for lv in levels_seen]
        assert indices == sorted(indices)

    def test_step_down_at_lowest_level_stays_at_lowest(self):
        """step_down() called at the lowest level does not raise; stays put."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        # Drive to the bottom
        for _ in range(10):
            ladder.step_down()

        # Must be at the most-degraded level and not raise
        lowest = list(DegradationLevel)[-1]
        assert ladder.current_level == lowest

    def test_is_at_full_true_when_full(self):
        """is_at_full() returns True only when the level is FULL."""
        from sr2.degradation.ladder import DegradationLadder

        ladder = DegradationLadder()
        assert ladder.is_at_full() is True

    def test_is_at_full_false_after_step_down(self):
        from sr2.degradation.ladder import DegradationLadder

        ladder = DegradationLadder()
        ladder.step_down()
        assert ladder.is_at_full() is False

    def test_active_providers_at_full_includes_all(self):
        """At FULL level, all provider categories are active."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        providers = ladder.active_providers()
        # Must include at minimum tools and memory categories
        provider_names = {p.lower() for p in providers}
        assert "tools" in provider_names or any("tool" in p for p in provider_names)

    def test_active_providers_at_system_prompt_only_is_minimal(self):
        """At SYSTEM_PROMPT_ONLY level, only system prompt content is active."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder(initial_level=DegradationLevel.SYSTEM_PROMPT_ONLY)
        providers = ladder.active_providers()
        # Must be a strict subset — at minimum only the system layer remains
        provider_names = {p.lower() for p in providers}
        assert "tools" not in provider_names
        assert len(providers) < 5  # Must be fewer than at FULL level

    def test_active_providers_shrink_at_each_level(self):
        """Each successive degradation level has <= active providers than the one above."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        all_levels = list(DegradationLevel)
        prev_count = None
        for level in all_levels:
            ladder = DegradationLadder(initial_level=level)
            count = len(ladder.active_providers())
            if prev_count is not None:
                assert count <= prev_count, (
                    f"Level {level} has more active providers than the previous level"
                )
            prev_count = count

    def test_reset_returns_to_full(self):
        """reset() brings the ladder back to FULL from any degraded state."""
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        ladder.step_down()
        ladder.step_down()
        ladder.reset()
        assert ladder.current_level == DegradationLevel.FULL


# ===========================================================================
# 2. Priority-Based Shedding
# ===========================================================================


class TestPriorityShedding:
    """shed(layers, budget) removes lowest-priority layers until total token
    count is within the budget. Returns surviving layers.
    """

    def test_no_shedding_when_under_budget(self):
        """When total tokens <= budget, all layers survive unchanged."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="sys", priority=10, token_count=100),
            ContentLayer(name="mem", priority=5, token_count=50),
        ]
        result = shed(layers, budget=200)
        assert len(result) == 2
        assert {l.name for l in result} == {"sys", "mem"}

    def test_no_shedding_when_exactly_at_budget(self):
        """Exactly at budget → no shedding."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="sys", priority=10, token_count=100),
        ]
        result = shed(layers, budget=100)
        assert len(result) == 1

    def test_sheds_lowest_priority_first(self):
        """When over budget, the layer with the lowest priority is shed first."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="high", priority=10, token_count=80),
            ContentLayer(name="low", priority=1, token_count=80),
        ]
        # Total 160; budget 100 → must shed 'low' (priority=1)
        result = shed(layers, budget=100)
        names = {l.name for l in result}
        assert "low" not in names
        assert "high" in names

    def test_sheds_multiple_layers_if_necessary(self):
        """Multiple layers may be shed until total is within budget."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="critical", priority=100, token_count=50),
            ContentLayer(name="medium", priority=5, token_count=60),
            ContentLayer(name="low", priority=1, token_count=60),
        ]
        # Total 170; budget 60 → shed 'low' then 'medium' to survive with just 'critical'
        result = shed(layers, budget=60)
        names = {l.name for l in result}
        assert "critical" in names
        assert "low" not in names
        assert "medium" not in names

    def test_result_total_tokens_within_budget(self):
        """After shedding, sum of surviving token counts <= budget."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="a", priority=10, token_count=30),
            ContentLayer(name="b", priority=5, token_count=30),
            ContentLayer(name="c", priority=1, token_count=30),
        ]
        budget = 55
        result = shed(layers, budget=budget)
        total = sum(l.token_count for l in result)
        assert total <= budget

    def test_empty_layers_returns_empty(self):
        from sr2.degradation.shedding import shed

        result = shed([], budget=1000)
        assert result == []

    def test_preserves_original_list_order_of_survivors(self):
        """Surviving layers maintain original declaration order."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="first", priority=10, token_count=20),
            ContentLayer(name="second", priority=8, token_count=20),
            ContentLayer(name="third", priority=1, token_count=100),
        ]
        # Budget 50 → shed 'third'; 'first' and 'second' survive
        result = shed(layers, budget=50)
        assert [l.name for l in result] == ["first", "second"]

    def test_budget_zero_sheds_all(self):
        """A budget of 0 leaves no layers (unless a layer has 0 tokens)."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="a", priority=10, token_count=10),
            ContentLayer(name="b", priority=5, token_count=5),
        ]
        result = shed(layers, budget=0)
        total = sum(l.token_count for l in result)
        assert total == 0

    def test_ties_in_priority_shed_all_tied_layers_if_needed(self):
        """When priorities tie and budget forces shedding, tied layers may all be shed."""
        from sr2.degradation.shedding import shed

        layers = [
            ContentLayer(name="keep", priority=10, token_count=30),
            ContentLayer(name="tied_a", priority=1, token_count=50),
            ContentLayer(name="tied_b", priority=1, token_count=50),
        ]
        result = shed(layers, budget=30)
        assert any(l.name == "keep" for l in result)
        total = sum(l.token_count for l in result)
        assert total <= 30


# ===========================================================================
# 3. Circuit Breaker
# ===========================================================================


class TestCircuitBreaker:
    """CircuitBreaker per content provider.
    States: CLOSED (normal) → OPEN (rejecting) → HALF_OPEN → CLOSED or OPEN.
    """

    def test_initial_state_is_closed(self):
        """A new circuit breaker starts closed (normal operation)."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.state == CircuitState.CLOSED

    def test_is_open_false_when_closed(self):
        from sr2.degradation.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.is_open() is False

    def test_record_failure_does_not_open_below_threshold(self):
        """Failures below threshold keep breaker closed."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_record_failure_opens_at_threshold(self):
        """Exactly N consecutive failures opens the breaker."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_is_open_true_when_open(self):
        from sr2.degradation.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is True

    def test_open_breaker_rejects_call(self):
        """allow_request() returns False when breaker is OPEN."""
        from sr2.degradation.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is False

    def test_closed_breaker_allows_call(self):
        """allow_request() returns True when breaker is CLOSED."""
        from sr2.degradation.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert cb.allow_request() is True

    def test_success_resets_failure_count_when_closed(self):
        """record_success() while CLOSED resets the consecutive-failure count."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()  # Should be only 1 failure now — not enough to open
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_recovery_timeout(self):
        """After the recovery timeout elapses, an OPEN breaker allows one probe."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.1)  # exceed the 50ms timeout
        # Next allow_request() should transition to HALF_OPEN and return True
        result = cb.allow_request()
        assert result is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """A success in HALF_OPEN state closes the breaker."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)

        cb.allow_request()  # moves to HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """A failure in HALF_OPEN state re-opens the breaker."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)

        cb.allow_request()  # moves to HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_failure_count_resets_after_closing_from_half_open(self):
        """After a successful close from HALF_OPEN, failure count starts fresh."""
        from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)

        cb.allow_request()   # HALF_OPEN
        cb.record_success()  # CLOSED

        # One failure should not re-open
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED


# ===========================================================================
# 4. Fallback Content
# ===========================================================================


class TestFallbackContent:
    """FallbackProvider selects content based on configured fallback mode:
    - 'cached': return previously cached content for the provider
    - 'none': return None (layer is skipped / empty)
    - 'static': return a configured static string
    """

    def test_none_mode_returns_none(self):
        """Fallback mode 'none' always returns None."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="none")
        result = provider.get_fallback(provider_name="memory")
        assert result is None

    def test_static_mode_returns_configured_string(self):
        """Fallback mode 'static' returns the configured static value."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="static", static_value="[memory unavailable]")
        result = provider.get_fallback(provider_name="memory")
        assert result == "[memory unavailable]"

    def test_cached_mode_returns_none_when_no_cache(self):
        """Fallback mode 'cached' returns None when no cached value has been stored."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="cached")
        result = provider.get_fallback(provider_name="memory")
        assert result is None

    def test_cached_mode_returns_stored_value(self):
        """Fallback mode 'cached' returns previously stored content for a provider."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="cached")
        provider.update_cache(provider_name="memory", content="last known memory output")
        result = provider.get_fallback(provider_name="memory")
        assert result == "last known memory output"

    def test_cached_mode_is_per_provider(self):
        """Cache is scoped per provider name — different providers don't share a slot."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="cached")
        provider.update_cache(provider_name="memory", content="memory content")
        provider.update_cache(provider_name="tools", content="tools content")

        assert provider.get_fallback("memory") == "memory content"
        assert provider.get_fallback("tools") == "tools content"

    def test_cached_mode_cache_is_updated_by_update_cache(self):
        """update_cache() overwrites the previous cached value."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="cached")
        provider.update_cache("memory", "first value")
        provider.update_cache("memory", "second value")
        assert provider.get_fallback("memory") == "second value"

    def test_static_mode_ignores_provider_name(self):
        """Static fallback returns the same value regardless of provider name."""
        from sr2.degradation.fallback import FallbackProvider

        provider = FallbackProvider(mode="static", static_value="STATIC")
        assert provider.get_fallback("memory") == "STATIC"
        assert provider.get_fallback("tools") == "STATIC"
        assert provider.get_fallback("anything_else") == "STATIC"

    def test_invalid_mode_raises_value_error(self):
        """Constructing a FallbackProvider with an unknown mode raises ValueError."""
        from sr2.degradation.fallback import FallbackProvider

        with pytest.raises((ValueError, TypeError)):
            FallbackProvider(mode="bogus_mode")


# ===========================================================================
# 5. Policy Registry
# ===========================================================================


class TestPolicyRegistry:
    """DegradationPolicyRegistry stores and retrieves per-provider policies."""

    def test_register_and_retrieve_policy(self):
        """A registered policy can be retrieved by provider name."""
        from sr2.degradation.registry import DegradationPolicyRegistry, DegradationPolicy

        registry = DegradationPolicyRegistry()
        policy = DegradationPolicy(
            provider_name="memory",
            fallback_mode="none",
            circuit_breaker_threshold=3,
            priority=5,
        )
        registry.register(policy)
        retrieved = registry.get("memory")
        assert retrieved is not None
        assert retrieved.provider_name == "memory"

    def test_get_unknown_provider_returns_none(self):
        """Retrieving a policy for an unknown provider returns None."""
        from sr2.degradation.registry import DegradationPolicyRegistry

        registry = DegradationPolicyRegistry()
        assert registry.get("nonexistent") is None

    def test_register_overwrites_existing(self):
        """Re-registering a provider replaces the old policy."""
        from sr2.degradation.registry import DegradationPolicyRegistry, DegradationPolicy

        registry = DegradationPolicyRegistry()
        old_policy = DegradationPolicy(
            provider_name="tools",
            fallback_mode="none",
            circuit_breaker_threshold=3,
            priority=1,
        )
        new_policy = DegradationPolicy(
            provider_name="tools",
            fallback_mode="static",
            circuit_breaker_threshold=5,
            priority=2,
        )
        registry.register(old_policy)
        registry.register(new_policy)
        result = registry.get("tools")
        assert result.circuit_breaker_threshold == 5

    def test_list_all_returns_registered_policies(self):
        """list_all() returns all registered policies."""
        from sr2.degradation.registry import DegradationPolicyRegistry, DegradationPolicy

        registry = DegradationPolicyRegistry()
        p1 = DegradationPolicy("memory", "none", 3, priority=5)
        p2 = DegradationPolicy("tools", "static", 5, priority=2)
        registry.register(p1)
        registry.register(p2)
        all_policies = registry.list_all()
        assert len(all_policies) == 2
        names = {p.provider_name for p in all_policies}
        assert names == {"memory", "tools"}
