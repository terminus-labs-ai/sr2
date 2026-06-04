"""Tests for sr2.degradation — ladder state machine + public API stability.

Covers:
  1. DegradationLadder — step-down, reset, is_at_full, active_providers
  2. Public API stability — exports remain importable

Removed (dead code / covered elsewhere):
  - TestPriorityShedding → tests/degradation/test_shedding_on_layers.py
  - TestCircuitBreaker → tests/sr2/test_circuit_breaker_and_timeouts.py
  - TestFallbackContent → FallbackProvider never wired into orchestrator
  - TestPolicyRegistry → DegradationPolicyRegistry never wired
  - TestRegistryUsesPluginRegistry → unwired registry's plugin discovery
  - TestStrategyCollisionDetection → unwired
  - TestStrategyProtocolValidation → unwired
  - TestDegradationPolicyDataclassRetained → backward compat for unwired registry
  - TestStrategyLazyDiscovery → unwired
"""

from __future__ import annotations

import pytest


# ===========================================================================
# 1. DegradationLadder
# ===========================================================================


class TestDegradationLadder:
    """DegradationLadder tracks which degradation level is active."""

    def test_ladder_has_five_levels(self):
        """Ladder exposes exactly 5 legacy levels."""
        from sr2.degradation.ladder import DegradationLevel

        levels = list(DegradationLevel)
        assert len(levels) == 5

    def test_level_names_include_full_and_system_prompt_only(self):
        from sr2.degradation.ladder import DegradationLevel

        names = {level.name.lower() for level in DegradationLevel}
        assert "full" in names
        assert "system_prompt_only" in names

    def test_default_level_is_full(self):
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        assert ladder.current_level == DegradationLevel.FULL

    def test_step_down_from_full_moves_to_next_level(self):
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        ladder.step_down()
        assert ladder.current_level != DegradationLevel.FULL

    def test_step_down_is_monotonically_more_degraded(self):
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        levels_seen = [ladder.current_level]
        for _ in range(4):
            ladder.step_down()
            levels_seen.append(ladder.current_level)

        all_levels = list(DegradationLevel)
        indices = [all_levels.index(lv) for lv in levels_seen]
        assert indices == sorted(indices)

    def test_step_down_at_lowest_level_stays_at_lowest(self):
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        for _ in range(10):
            ladder.step_down()

        lowest = list(DegradationLevel)[-1]
        assert ladder.current_level == lowest

    def test_is_at_full_true_when_full(self):
        from sr2.degradation.ladder import DegradationLadder

        ladder = DegradationLadder()
        assert ladder.is_at_full() is True

    def test_is_at_full_false_after_step_down(self):
        from sr2.degradation.ladder import DegradationLadder

        ladder = DegradationLadder()
        ladder.step_down()
        assert ladder.is_at_full() is False

    def test_active_providers_at_full_includes_all(self):
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        providers = ladder.active_providers()
        provider_names = {p.lower() for p in providers}
        assert "tools" in provider_names or any("tool" in p for p in provider_names)

    def test_active_providers_at_system_prompt_only_is_minimal(self):
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder(initial_level=DegradationLevel.SYSTEM_PROMPT_ONLY)
        providers = ladder.active_providers()
        provider_names = {p.lower() for p in providers}
        assert "tools" not in provider_names
        assert len(providers) < 5

    def test_active_providers_shrink_at_each_level(self):
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
        from sr2.degradation.ladder import DegradationLadder, DegradationLevel

        ladder = DegradationLadder()
        ladder.step_down()
        ladder.step_down()
        ladder.reset()
        assert ladder.current_level == DegradationLevel.FULL


# ===========================================================================
# 2. Public API Stability
# ===========================================================================


class TestDegradationPublicApiStability:
    def test_degradation_policy_importable_from_sr2_degradation(self):
        """DegradationPolicy is importable from sr2.degradation."""
        from sr2.degradation import DegradationPolicy as DP
        assert DP is not None

    def test_degradation_policy_registry_importable_from_sr2_degradation(self):
        """DegradationPolicyRegistry is importable from sr2.degradation."""
        from sr2.degradation import DegradationPolicyRegistry as DPR
        assert DPR is not None

    def test_degradation_policy_store_importable_from_sr2_degradation(self):
        """DegradationPolicyStore is importable from sr2.degradation."""
        from sr2.degradation import DegradationPolicyStore as DPS
        assert DPS is not None

    def test_plugin_registry_importable_from_sr2_plugins(self):
        """PluginRegistry remains accessible."""
        from sr2.plugins.registry import PluginRegistry as PR
        assert PR is not None

    def test_degradation_module_does_not_export_plugin_registry_directly(self):
        """PluginRegistry is an impl detail; sr2.degradation should not re-export it."""
        import sr2.degradation as deg_module
        if hasattr(deg_module, "__all__"):
            assert "PluginRegistry" not in deg_module.__all__
