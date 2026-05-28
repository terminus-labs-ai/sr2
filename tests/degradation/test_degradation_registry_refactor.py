"""Tests pinning the desired post-refactor behavior for sr2-13.

Issue: Two registry patterns — PluginRegistry (entry-point) vs
DegradationPolicyRegistry (hand-rolled dict). These tests assert the
*desired* state after the refactor:

  1. DegradationPolicyRegistry must delegate to PluginRegistry internally,
     using the 'sr2.degradation_policies' entry-point group.
  2. Collision detection must be present (currently missing — hand-rolled
     dict silently overwrites on name collision across distributions).
  3. Protocol validation must be present (currently absent — hand-rolled
     dict stores any object).
  4. DegradationPolicy dataclass (runtime config) remains separate from the
     strategy-class registry.
  5. The public API surface (sr2.degradation exports) stays stable.

All tests currently FAIL because DegradationPolicyRegistry is hand-rolled
and has none of these properties.
"""

from __future__ import annotations

import importlib.metadata
from typing import Protocol, runtime_checkable
from unittest.mock import MagicMock, patch

import pytest

from sr2.degradation.registry import DegradationPolicy, DegradationPolicyRegistry
from sr2.plugins.errors import PluginCollisionError, PluginNotFoundError
from sr2.plugins.registry import PluginRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@runtime_checkable
class DegradationStrategyProtocol(Protocol):
    """Minimal protocol a degradation strategy class must satisfy."""

    @classmethod
    def build(cls, config: dict) -> "DegradationStrategyProtocol": ...


class ConformingStrategy:
    @classmethod
    def build(cls, config: dict) -> "ConformingStrategy":
        return cls()


class NonConformingStrategy:
    """Missing build() — does not satisfy DegradationStrategyProtocol."""
    pass


def _make_entry_point(name: str, cls: type, dist_name: str = "sr2-core") -> MagicMock:
    ep = MagicMock(spec=importlib.metadata.EntryPoint)
    ep.name = name
    ep.load.return_value = cls
    dist = MagicMock()
    dist.name = dist_name
    ep.dist = dist
    return ep


# ---------------------------------------------------------------------------
# 1. DegradationPolicyRegistry must use PluginRegistry internally
# ---------------------------------------------------------------------------

class TestRegistryUsesPluginRegistry:
    def test_registry_has_plugin_registry_attribute(self):
        """DegradationPolicyRegistry must expose an internal PluginRegistry instance."""
        registry = DegradationPolicyRegistry()
        # The refactored registry must have a PluginRegistry as an attribute
        # (any reasonable attribute name: _registry, _plugin_registry, etc.)
        inner = getattr(registry, "_registry", None) or \
                getattr(registry, "_plugin_registry", None) or \
                getattr(registry, "_entry_point_registry", None)
        assert isinstance(inner, PluginRegistry), (
            "DegradationPolicyRegistry must delegate to a PluginRegistry instance. "
            "Currently it is a hand-rolled dict with no entry-point discovery."
        )

    def test_get_strategy_class_uses_sr2_degradation_policies_group(self):
        """get_strategy() must query the 'sr2.degradation_policies' entry-point group."""
        ep = _make_entry_point("circuit_breaker", ConformingStrategy)
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = [ep]
            registry = DegradationPolicyRegistry()
            # Trigger discovery by requesting a strategy class
            try:
                registry.get_strategy("circuit_breaker")
            except (AttributeError, PluginNotFoundError, TypeError):
                pass  # May raise — we just need the entry_points call to happen
            # The entry-point call must use the correct group
            calls = [str(call) for call in mock_ep.call_args_list]
            assert any("sr2.degradation_policies" in c for c in calls), (
                "PluginRegistry must be initialized with 'sr2.degradation_policies' group. "
                "Currently the registry never calls entry_points at all."
            )

    def test_get_strategy_returns_class_not_instance(self):
        """get_strategy() returns the strategy *class*, not an instance."""
        ep = _make_entry_point("my_strategy", ConformingStrategy)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            result = registry.get_strategy("my_strategy")
        assert result is ConformingStrategy, (
            "get_strategy() must return the class (for the caller to instantiate), "
            "not an instance."
        )

    def test_list_strategy_names_returns_entry_point_names(self):
        """list_strategy_names() returns names from entry-point discovery."""
        eps = [
            _make_entry_point("circuit_breaker", ConformingStrategy),
            _make_entry_point("sla_breaker", ConformingStrategy),
        ]
        with patch("sr2.plugins.registry.entry_points", return_value=eps):
            registry = DegradationPolicyRegistry()
            names = registry.list_strategy_names()
        assert sorted(names) == ["circuit_breaker", "sla_breaker"], (
            "list_strategy_names() must enumerate entry-point-discovered strategy names, "
            "not a hand-rolled internal dict."
        )


# ---------------------------------------------------------------------------
# 2. Collision detection (currently absent)
# ---------------------------------------------------------------------------

class TestStrategyCollisionDetection:
    def test_collision_raises_plugin_collision_error(self):
        """Two distributions registering the same strategy name must raise PluginCollisionError."""
        ep1 = _make_entry_point("breaker", ConformingStrategy, dist_name="dist-a")
        ep2 = _make_entry_point("breaker", ConformingStrategy, dist_name="dist-b")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(PluginCollisionError):
                registry.get_strategy("breaker")

    def test_collision_error_names_both_distributions(self):
        """The collision error message must identify both conflicting distributions."""
        ep1 = _make_entry_point("shared", ConformingStrategy, dist_name="pkg-alpha")
        ep2 = _make_entry_point("shared", ConformingStrategy, dist_name="pkg-beta")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(PluginCollisionError) as exc_info:
                registry.get_strategy("shared")
        msg = str(exc_info.value)
        assert "pkg-alpha" in msg and "pkg-beta" in msg

    def test_non_colliding_strategy_resolves_despite_collision_elsewhere(self):
        """A clean name resolves even when another name collides in the same scan."""
        ep1 = _make_entry_point("clash", ConformingStrategy, dist_name="dist-a")
        ep2 = _make_entry_point("clash", ConformingStrategy, dist_name="dist-b")
        ep3 = _make_entry_point("unique", ConformingStrategy, dist_name="dist-c")
        with patch("sr2.plugins.registry.entry_points", return_value=[ep1, ep2, ep3]):
            registry = DegradationPolicyRegistry()
            result = registry.get_strategy("unique")
        assert result is ConformingStrategy


# ---------------------------------------------------------------------------
# 3. Protocol validation (currently absent)
# ---------------------------------------------------------------------------

class TestStrategyProtocolValidation:
    def test_non_conforming_strategy_raises_type_error(self):
        """A strategy class missing required protocol members must raise TypeError, not pass silently."""
        ep = _make_entry_point("bad_strategy", NonConformingStrategy)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(TypeError):
                registry.get_strategy("bad_strategy")

    def test_non_conforming_strategy_error_is_not_not_found(self):
        """TypeError (bad class) must be distinct from PluginNotFoundError (missing name)."""
        ep = _make_entry_point("bad_strategy", NonConformingStrategy)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(Exception) as exc_info:
                registry.get_strategy("bad_strategy")
        assert not isinstance(exc_info.value, PluginNotFoundError)

    def test_unknown_strategy_raises_plugin_not_found_error(self):
        """A name that has no entry point raises PluginNotFoundError."""
        ep = _make_entry_point("known", ConformingStrategy)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            with pytest.raises(PluginNotFoundError):
                registry.get_strategy("unknown_strategy")

    def test_conforming_strategy_does_not_raise(self):
        """A properly conforming strategy class is returned without error."""
        ep = _make_entry_point("good_strategy", ConformingStrategy)
        with patch("sr2.plugins.registry.entry_points", return_value=[ep]):
            registry = DegradationPolicyRegistry()
            result = registry.get_strategy("good_strategy")
        assert result is ConformingStrategy


# ---------------------------------------------------------------------------
# 4. DegradationPolicy dataclass stays separate (runtime config, not plugin)
# ---------------------------------------------------------------------------

class TestDegradationPolicyDataclassRetained:
    def test_degradation_policy_dataclass_still_exists(self):
        """DegradationPolicy must still be importable as a dataclass."""
        from sr2.degradation.registry import DegradationPolicy as DP
        policy = DP(
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
        # Policy config objects are registered directly (not via entry points)
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


# ---------------------------------------------------------------------------
# 5. Public API surface (sr2.degradation exports)
# ---------------------------------------------------------------------------

class TestPublicApiStability:
    def test_degradation_policy_importable_from_sr2_degradation(self):
        """DegradationPolicy is importable from sr2.degradation (not broken by refactor)."""
        from sr2.degradation import DegradationPolicy as DP
        assert DP is not None

    def test_degradation_policy_registry_importable_from_sr2_degradation(self):
        """DegradationPolicyRegistry is importable from sr2.degradation (not broken by refactor)."""
        from sr2.degradation import DegradationPolicyRegistry as DPR
        assert DPR is not None

    def test_plugin_registry_importable_from_sr2_plugins(self):
        """PluginRegistry remains accessible (sanity check, not broken by collapse)."""
        from sr2.plugins.registry import PluginRegistry as PR
        assert PR is not None

    def test_degradation_module_does_not_export_plugin_registry_directly(self):
        """PluginRegistry is an impl detail; sr2.degradation should not re-export it."""
        import sr2.degradation as deg_module
        # PluginRegistry should NOT appear in degradation's public __all__
        if hasattr(deg_module, "__all__"):
            assert "PluginRegistry" not in deg_module.__all__, (
                "PluginRegistry is a plugin infrastructure detail and must not "
                "appear in sr2.degradation.__all__."
            )


# ---------------------------------------------------------------------------
# 6. Lazy discovery — entry-point scan happens at most once per registry
# ---------------------------------------------------------------------------

class TestStrategyLazyDiscovery:
    def test_entry_points_not_called_at_construction(self):
        """entry_points() must not be called when DegradationPolicyRegistry is constructed."""
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            _registry = DegradationPolicyRegistry()
            mock_ep.assert_not_called()

    def test_entry_points_called_at_most_once_across_multiple_get_strategy_calls(self):
        """entry_points() is called once regardless of how many get_strategy() calls are made."""
        ep = _make_entry_point("cb", ConformingStrategy)
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
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
        with patch("sr2.plugins.registry.entry_points") as mock_ep:
            mock_ep.return_value = []
            registry = DegradationPolicyRegistry()
            try:
                registry.get_strategy("any")
            except PluginNotFoundError:
                pass
        mock_ep.assert_called_once_with(group="sr2.degradation_policies")
