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


# ===========================================================================
# Tests from test_degradation_registry_refactor.py — SR2-13: two registry patterns
# DegradationPolicyRegistry must delegate to PluginRegistry internally.
# ===========================================================================

import importlib.metadata as _deg_importlib
from typing import Protocol as _Protocol, runtime_checkable as _runtime_checkable
from unittest.mock import MagicMock as _MagicMock, patch as _patch_deg

from sr2.degradation.registry import DegradationPolicy, DegradationPolicyRegistry
from sr2.plugins.errors import PluginCollisionError, PluginNotFoundError
from sr2.plugins.registry import PluginRegistry


@_runtime_checkable
class _DegradationStrategyProtocol(_Protocol):
    """Minimal protocol a degradation strategy class must satisfy."""
    @classmethod
    def build(cls, config: dict) -> "_DegradationStrategyProtocol": ...


class _ConformingStrategy:
    @classmethod
    def build(cls, config: dict) -> "_ConformingStrategy":
        return cls()


class _NonConformingStrategy:
    """Missing build() — does not satisfy _DegradationStrategyProtocol."""
    pass


def _make_entry_point_deg(name: str, cls: type, dist_name: str = "sr2-core") -> _MagicMock:
    ep = _MagicMock(spec=_deg_importlib.EntryPoint)
    ep.name = name
    ep.load.return_value = cls
    dist = _MagicMock()
    dist.name = dist_name
    ep.dist = dist
    return ep


class TestRegistryUsesPluginRegistry:
    def test_registry_has_plugin_registry_attribute(self):
        """DegradationPolicyRegistry must expose an internal PluginRegistry instance."""
        registry = DegradationPolicyRegistry()
        inner = (
            getattr(registry, "_registry", None)
            or getattr(registry, "_plugin_registry", None)
            or getattr(registry, "_entry_point_registry", None)
        )
        assert isinstance(inner, PluginRegistry), (
            "DegradationPolicyRegistry must delegate to a PluginRegistry instance."
        )

    def test_get_strategy_class_uses_sr2_degradation_policies_group(self):
        """get_strategy() must query the 'sr2.degradation_policies' entry-point group."""
        ep = _make_entry_point_deg("circuit_breaker", _ConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = [ep]
            registry = DegradationPolicyRegistry()
            try:
                registry.get_strategy("circuit_breaker")
            except (AttributeError, PluginNotFoundError, TypeError):
                pass
            calls = [str(call) for call in mock_ep.call_args_list]
            assert any("sr2.degradation_policies" in c for c in calls), (
                "PluginRegistry must be initialized with 'sr2.degradation_policies' group."
            )

    def test_get_strategy_returns_class_not_instance(self):
        """get_strategy() returns the strategy *class*, not an instance."""
        ep = _make_entry_point_deg("my_strategy", _ConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            result = registry.get_strategy("my_strategy")
        assert result is _ConformingStrategy

    def test_list_strategy_names_returns_entry_point_names(self):
        """list_strategy_names() returns names from entry-point discovery."""
        eps = [
            _make_entry_point_deg("circuit_breaker", _ConformingStrategy),
            _make_entry_point_deg("sla_breaker", _ConformingStrategy),
        ]
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=eps):
            registry = DegradationPolicyRegistry()
            names = registry.list_strategy_names()
        assert sorted(names) == ["circuit_breaker", "sla_breaker"]


class TestStrategyCollisionDetection:
    def test_collision_raises_plugin_collision_error(self):
        """Two distributions registering the same strategy name must raise PluginCollisionError."""
        ep1 = _make_entry_point_deg("breaker", _ConformingStrategy, dist_name="dist-a")
        ep2 = _make_entry_point_deg("breaker", _ConformingStrategy, dist_name="dist-b")
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(PluginCollisionError):
                registry.get_strategy("breaker")

    def test_collision_error_names_both_distributions(self):
        """The collision error message must identify both conflicting distributions."""
        ep1 = _make_entry_point_deg("shared", _ConformingStrategy, dist_name="pkg-alpha")
        ep2 = _make_entry_point_deg("shared", _ConformingStrategy, dist_name="pkg-beta")
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(PluginCollisionError) as exc_info:
                registry.get_strategy("shared")
        msg = str(exc_info.value)
        assert "pkg-alpha" in msg and "pkg-beta" in msg

    def test_non_colliding_strategy_resolves_despite_collision_elsewhere(self):
        """A clean name resolves even when another name collides in the same scan."""
        ep1 = _make_entry_point_deg("clash", _ConformingStrategy, dist_name="dist-a")
        ep2 = _make_entry_point_deg("clash", _ConformingStrategy, dist_name="dist-b")
        ep3 = _make_entry_point_deg("unique", _ConformingStrategy, dist_name="dist-c")
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep1, ep2, ep3]):
            registry = DegradationPolicyRegistry()
            result = registry.get_strategy("unique")
        assert result is _ConformingStrategy


class TestStrategyProtocolValidation:
    def test_non_conforming_strategy_raises_type_error(self):
        """A strategy class missing required protocol members must raise TypeError."""
        ep = _make_entry_point_deg("bad_strategy", _NonConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(TypeError):
                registry.get_strategy("bad_strategy")

    def test_non_conforming_strategy_error_is_not_not_found(self):
        """TypeError (bad class) must be distinct from PluginNotFoundError (missing name)."""
        ep = _make_entry_point_deg("bad_strategy", _NonConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(Exception) as exc_info:
                registry.get_strategy("bad_strategy")
        assert not isinstance(exc_info.value, PluginNotFoundError)

    def test_unknown_strategy_raises_plugin_not_found_error(self):
        """A name that has no entry point raises PluginNotFoundError."""
        ep = _make_entry_point_deg("known", _ConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(PluginNotFoundError):
                registry.get_strategy("unknown_strategy")

    def test_conforming_strategy_does_not_raise(self):
        """A properly conforming strategy class is returned without error."""
        ep = _make_entry_point_deg("good_strategy", _ConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            result = registry.get_strategy("good_strategy")
        assert result is _ConformingStrategy


class TestDegradationPolicyDataclassRetained:
    def test_degradation_policy_dataclass_still_exists(self):
        """DegradationPolicy must still be importable as a dataclass."""
        policy = DegradationPolicy(
            provider_name="memory",
            fallback_mode="static",
            circuit_breaker_threshold=5,
            priority=3,
        )
        assert policy.provider_name == "memory"
        assert policy.fallback_mode == "static"
        assert policy.circuit_breaker_threshold == 5
        assert policy.priority == 3

    def test_degradation_policy_is_not_discovered_via_entry_points(self):
        """DegradationPolicy instances are registered directly, not via entry points."""
        registry = DegradationPolicyRegistry()
        policy = DegradationPolicy(
            provider_name="tools",
            fallback_mode="cached",
            circuit_breaker_threshold=3,
            priority=1,
        )
        registry.register(policy)
        retrieved = registry.get("tools")
        assert retrieved is not None
        assert retrieved.provider_name == "tools"

    def test_register_and_get_policy_config_unchanged(self):
        """Existing policy config store API (register/get) continues to work."""
        registry = DegradationPolicyRegistry()
        p1 = DegradationPolicy("memory", "none", 3, priority=5)
        p2 = DegradationPolicy("tools", "static", 5, priority=2)
        registry.register(p1)
        registry.register(p2)
        assert registry.get("memory").fallback_mode == "none"
        assert registry.get("tools").circuit_breaker_threshold == 5

    def test_list_all_returns_config_objects_not_strategy_classes(self):
        """list_all() returns DegradationPolicy config objects, not strategy classes."""
        registry = DegradationPolicyRegistry()
        p1 = DegradationPolicy("memory", "none", 3, priority=5)
        registry.register(p1)
        results = registry.list_all()
        assert len(results) == 1
        assert isinstance(results[0], DegradationPolicy)


class TestDegradationPublicApiStability:
    def test_degradation_policy_importable_from_sr2_degradation(self):
        """DegradationPolicy is importable from sr2.degradation."""
        from sr2.degradation import DegradationPolicy as DP
        assert DP is not None

    def test_degradation_policy_registry_importable_from_sr2_degradation(self):
        """DegradationPolicyRegistry is importable from sr2.degradation."""
        from sr2.degradation import DegradationPolicyRegistry as DPR
        assert DPR is not None

    def test_plugin_registry_importable_from_sr2_plugins(self):
        """PluginRegistry remains accessible."""
        from sr2.plugins.registry import PluginRegistry as PR
        assert PR is not None

    def test_degradation_module_does_not_export_plugin_registry_directly(self):
        """PluginRegistry is an impl detail; sr2.degradation should not re-export it."""
        import sr2.degradation as deg_module
        if hasattr(deg_module, "__all__"):
            assert "PluginRegistry" not in deg_module.__all__


class TestStrategyLazyDiscovery:
    def test_entry_points_not_called_at_construction(self):
        """entry_points() must not be called when DegradationPolicyRegistry is constructed."""
        with _patch_deg("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            _registry = DegradationPolicyRegistry()
            mock_ep.assert_not_called()

    def test_entry_points_called_at_most_once_across_multiple_get_strategy_calls(self):
        """entry_points() is called once regardless of how many get_strategy() calls are made."""
        ep = _make_entry_point_deg("cb", _ConformingStrategy)
        with _patch_deg("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = [ep]
            registry = DegradationPolicyRegistry()
            registry.get_strategy("cb")
            registry.get_strategy("cb")
            try:
                registry.get_strategy("nonexistent")
            except PluginNotFoundError:
                pass
        mock_ep.assert_called_once()

    def test_entry_points_called_with_correct_group(self):
        """entry_points() is called with group='sr2.degradation_policies'."""
        with _patch_deg("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            registry = DegradationPolicyRegistry()
            try:
                registry.get_strategy("any")
            except PluginNotFoundError:
                pass
        mock_ep.assert_called_once_with(group="sr2.degradation_policies")
