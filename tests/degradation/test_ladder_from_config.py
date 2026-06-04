"""Tests for DegradationLadder.from_config — FR3 (sr2-83).

Replaces the hardcoded DegradationLevel enum + _ACTIVE_PROVIDERS map
with a config-driven level table.  The ladder keeps step_down/reset/
current_level/is_at_full semantics but is constructed from DegradationConfig.

The legacy constructor __init__(initial_level=...) still works for backwards
compatibility, but new code should use from_config().
"""

import pytest

from sr2.config.models import (
    ConfigError,
    DegradationConfig,
    DegradationLevelConfig,
    DegradationTriggerConfig,
)
from sr2.degradation.ladder import DegradationLadder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _full_config() -> DegradationConfig:
    """A typical 3-level config: full → reduced → minimal."""
    return DegradationConfig(
        levels=[
            DegradationLevelConfig(
                name="full",
                keep_categories=["memory", "tools", "context", "history"],
            ),
            DegradationLevelConfig(
                name="reduced",
                keep_categories=["context", "history"],
            ),
            DegradationLevelConfig(
                name="minimal",
                keep_categories=["context"],
            ),
        ],
        triggers=[DegradationTriggerConfig(type="overflow", threshold=1.2)],
    )


def _single_level_config() -> DegradationConfig:
    """Config with only one level — effectively always FULL."""
    return DegradationConfig(
        levels=[
            DegradationLevelConfig(
                name="full",
                keep_categories=["memory", "tools", "context"],
            ),
        ]
    )


# ===========================================================================
# 1. Construction
# ===========================================================================


class TestFromConfig:
    """DegradationLadder.from_config(cls, cfg) builds a ladder from config data."""

    def test_from_config_creates_ladder(self):
        ladder = DegradationLadder.from_config(_full_config())
        assert ladder is not None

    def test_starts_at_full(self):
        ladder = DegradationLadder.from_config(_full_config())
        assert ladder.is_at_full() is True
        assert ladder.current_level == 0

    def test_active_categories_at_full(self):
        ladder = DegradationLadder.from_config(_full_config())
        cats = ladder.active_categories()
        assert cats == {"memory", "tools", "context", "history"}

    def test_from_config_returns_class_method_result(self):
        """from_config is a classmethod."""
        assert callable(DegradationLadder.from_config)


# ===========================================================================
# 2. Step-down semantics
# ===========================================================================


class TestStepDown:
    """step_down() moves through config-defined levels."""

    def test_step_down_moves_to_next_level(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        assert ladder.current_level == 1

    def test_step_down_updates_active_categories(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        # Should be at 'reduced' level: context, history
        cats = ladder.active_categories()
        assert cats == {"context", "history"}

    def test_step_down_to_bottom(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        ladder.step_down()
        assert ladder.current_level == 2
        cats = ladder.active_categories()
        assert cats == {"context"}

    def test_step_down_at_bottom_is_noop(self):
        ladder = DegradationLadder.from_config(_full_config())
        # Drive to bottom (3 levels, indices 0, 1, 2)
        ladder.step_down()
        ladder.step_down()
        # Already at bottom — step_down should stay there
        ladder.step_down()
        ladder.step_down()
        assert ladder.current_level == 2

    def test_many_step_downs_do_not_raise(self):
        ladder = DegradationLadder.from_config(_full_config())
        for _ in range(100):
            ladder.step_down()
        assert ladder.current_level == 2

    def test_step_down_is_monotonic(self):
        """Level index can only increase (or stay same at bottom)."""
        ladder = DegradationLadder.from_config(_full_config())
        prev = 0
        for _ in range(5):
            ladder.step_down()
            assert ladder.current_level >= prev
            prev = ladder.current_level


# ===========================================================================
# 3. Reset
# ===========================================================================


class TestReset:
    """reset() returns to level 0 (FULL) regardless of current state."""

    def test_reset_from_bottom(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        ladder.step_down()
        ladder.reset()
        assert ladder.current_level == 0
        assert ladder.is_at_full() is True

    def test_reset_restores_all_categories(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        ladder.reset()
        cats = ladder.active_categories()
        assert cats == {"memory", "tools", "context", "history"}

    def test_reset_at_full_is_noop(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.reset()
        assert ladder.current_level == 0
        assert ladder.is_at_full() is True


# ===========================================================================
# 4. active_categories returns set[str]
# ===========================================================================


class TestActiveCategories:
    """active_categories() returns the set of category strings active at current level."""

    def test_returns_set(self):
        ladder = DegradationLadder.from_config(_full_config())
        cats = ladder.active_categories()
        assert isinstance(cats, set)

    def test_empty_keep_categories(self):
        """A level with empty keep_categories returns an empty set."""
        cfg = DegradationConfig(
            levels=[
                DegradationLevelConfig(name="full", keep_categories=["context"]),
                DegradationLevelConfig(name="empty", keep_categories=[]),
            ]
        )
        ladder = DegradationLadder.from_config(cfg)
        ladder.step_down()
        assert ladder.active_categories() == set()

    def test_categories_change_per_level(self):
        """Different levels have different active categories."""
        ladder = DegradationLadder.from_config(_full_config())
        full_cats = ladder.active_categories()
        ladder.step_down()
        reduced_cats = ladder.active_categories()
        assert reduced_cats < full_cats  # strict subset


# ===========================================================================
# 5. Single-level config (no-op ladder)
# ===========================================================================


class TestSingleLevelConfig:
    """A ladder with a single level is stuck at FULL — step_down is a no-op."""

    def test_step_down_is_noop(self):
        ladder = DegradationLadder.from_config(_single_level_config())
        ladder.step_down()
        assert ladder.current_level == 0

    def test_always_at_full(self):
        ladder = DegradationLadder.from_config(_single_level_config())
        for _ in range(10):
            ladder.step_down()
        assert ladder.is_at_full() is True

    def test_active_categories_stay_same(self):
        ladder = DegradationLadder.from_config(_single_level_config())
        cats = ladder.active_categories()
        ladder.step_down()
        assert ladder.active_categories() == cats


# ===========================================================================
# 6. Backwards compatibility: __init__ still works
# ===========================================================================


class TestLegacyInit:
    """The old __init__(initial_level=DegradationLevel.FULL) still works for
    backwards compatibility, but the enum is deprecated."""

    def test_legacy_init_still_works(self):
        """Old code that constructs DegradationLadder() directly still works."""
        ladder = DegradationLadder()
        assert ladder.is_at_full() is True

    def test_legacy_current_level_returns_enum(self):
        """Legacy construction returns DegradationLevel enum for current_level."""
        from sr2.degradation.ladder import DegradationLevel

        ladder = DegradationLadder()
        assert ladder.current_level == DegradationLevel.FULL


# ===========================================================================
# 7. Backwards compatibility: active_providers still works
# ===========================================================================


class TestActiveProvidersCompat:
    """active_providers() remains as an alias for active_categories() for backwards
    compatibility, returning a list instead of a set."""

    def test_active_providers_returns_list(self):
        ladder = DegradationLadder.from_config(_full_config())
        result = ladder.active_providers()
        assert isinstance(result, list)

    def test_active_providers_matches_categories(self):
        ladder = DegradationLadder.from_config(_full_config())
        assert set(ladder.active_providers()) == ladder.active_categories()


# ===========================================================================
# 8. is_at_full
# ===========================================================================


class TestIsAtFull:
    def test_true_at_start(self):
        ladder = DegradationLadder.from_config(_full_config())
        assert ladder.is_at_full() is True

    def test_false_after_step_down(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        assert ladder.is_at_full() is False

    def test_true_after_reset(self):
        ladder = DegradationLadder.from_config(_full_config())
        ladder.step_down()
        ladder.reset()
        assert ladder.is_at_full() is True
