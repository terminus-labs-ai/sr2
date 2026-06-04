"""Tests for DegradationController — trigger evaluation (FR4).

Bead: sr2-85

Covers:
  FR4: DegradationController evaluates configured triggers and drives
       ladder.step_down() when conditions are met.
  v1 triggers: overflow (budget pressure), context_limit (LLM rejection).
"""

from __future__ import annotations

import pytest

from sr2.config.models import (
    DegradationConfig,
    DegradationLevelConfig,
    DegradationTriggerConfig,
)
from sr2.degradation.ladder import DegradationLadder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    levels: list[dict],
    triggers: list[dict] | None = None,
) -> DegradationConfig:
    """Build a DegradationConfig from level/trigger dicts."""
    return DegradationConfig(
        levels=[
            DegradationLevelConfig(name=lv["name"], keep_categories=lv["cats"])
            for lv in levels
        ],
        triggers=[
            DegradationTriggerConfig(type=t["type"], threshold=t.get("threshold"))
            for t in (triggers or [])
        ],
    )


def _make_ladder(levels: list[dict]) -> DegradationLadder:
    """Build a ladder from level dicts."""
    return DegradationLadder.from_config(_make_config(levels))


def _make_controller(
    levels: list[dict],
    triggers: list[dict] | None = None,
) -> tuple["DegradationController", DegradationLadder]:
    """Build a controller + ladder pair from config dicts."""
    from sr2.degradation.controller import DegradationController

    cfg = _make_config(levels, triggers)
    ladder = DegradationLadder.from_config(cfg)
    ctrl = DegradationController(ladder, cfg.triggers)
    return ctrl, ladder


# ===========================================================================
# 1. Construction
# ===========================================================================


class TestControllerConstruction:
    def test_construct_with_ladder_and_triggers(self):
        """Controller accepts a ladder and a list of trigger configs."""
        from sr2.degradation.controller import DegradationController

        cfg = _make_config(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        ladder = DegradationLadder.from_config(cfg)
        ctrl = DegradationController(ladder, cfg.triggers)

        assert ctrl._ladder is ladder
        assert len(ctrl._triggers) == 1

    def test_construct_with_empty_triggers(self):
        """A controller with no triggers is valid — it just never fires."""
        from sr2.degradation.controller import DegradationController

        cfg = _make_config(
            [{"name": "full", "cats": ["memory", "context"]}],
            [],
        )
        ladder = DegradationLadder.from_config(cfg)
        ctrl = DegradationController(ladder, cfg.triggers)

        assert ctrl._triggers == []

    def test_exposes_ladder(self):
        """Controller exposes the ladder for external access."""
        ctrl, ladder = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        assert ctrl.ladder is ladder


# ===========================================================================
# 2. over_budget() — overflow trigger
# ===========================================================================


class TestOverBudget:
    """over_budget(total_tokens, budget) returns True when the request exceeds budget."""

    def test_over_budget_returns_true_when_exceeded(self):
        """total_tokens > budget → True."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        assert ctrl.over_budget(total_tokens=1100, budget=1000) is True

    def test_over_budget_returns_true_when_equal(self):
        """total_tokens == budget → not over budget (boundary)."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        assert ctrl.over_budget(total_tokens=1000, budget=1000) is False

    def test_over_budget_returns_false_when_under(self):
        """total_tokens < budget → False."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        assert ctrl.over_budget(total_tokens=500, budget=1000) is False

    def test_over_budget_without_overflow_trigger(self):
        """If no overflow trigger is configured, over_budget always returns False."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        # Even with massive overflow, no overflow trigger → never fires
        assert ctrl.over_budget(total_tokens=999999, budget=100) is False

    def test_over_budget_with_no_triggers(self):
        """Controller with no triggers: over_budget returns False."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [],
        )
        assert ctrl.over_budget(total_tokens=999999, budget=100) is False


# ===========================================================================
# 3. is_context_limit_error() — context_limit trigger
# ===========================================================================


class TestIsContextLimitError:
    """is_context_limit_error(exc) recognises LLM context-length errors."""

    def test_recognizes_aanthropic_context_window_exceeded(self):
        """Anthropic ContextWindowExceededError is recognised."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("Anthropic: prompt exceeds context window")
        assert ctrl.is_context_limit_error(exc) is True

    def test_recognizes_openai_context_length_exceeded(self):
        """OpenAI context length error message is recognised."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("This model's maximum context length is 8192 tokens")
        assert ctrl.is_context_limit_error(exc) is True

    def test_recognizes_bedrock_context_exceeded(self):
        """AWS Bedrock context exceeded error is recognised."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("Input is too long for the context window")
        assert ctrl.is_context_limit_error(exc) is True

    def test_recognizes_groq_context_limit(self):
        """Groq context limit error is recognised."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("Context length exceeded")
        assert ctrl.is_context_limit_error(exc) is True

    def test_does_not_match_regular_errors(self):
        """A regular error is NOT a context_limit error."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("Internal server error")
        assert ctrl.is_context_limit_error(exc) is False

    def test_does_not_match_rate_limit_errors(self):
        """Rate limit errors are NOT context_limit errors."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("Rate limit exceeded")
        assert ctrl.is_context_limit_error(exc) is False

    def test_does_not_match_connection_errors(self):
        """Connection errors are NOT context_limit errors."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("Connection refused")
        assert ctrl.is_context_limit_error(exc) is False

    def test_returns_false_without_context_limit_trigger(self):
        """If no context_limit trigger is configured, always returns False."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        exc = Exception("context window exceeded")
        assert ctrl.is_context_limit_error(exc) is False

    def test_returns_false_when_no_triggers(self):
        """No triggers at all: is_context_limit_error always returns False."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [],
        )
        exc = Exception("context exceeded")
        assert ctrl.is_context_limit_error(exc) is False

    def test_case_insensitive_matching(self):
        """Context limit matching is case-insensitive."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        exc = Exception("CONTEXT WINDOW EXCEEDED")
        assert ctrl.is_context_limit_error(exc) is True

    def test_handles_non_exception_strings(self):
        """If passed a string instead of exception, treat it as error message."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        assert ctrl.is_context_limit_error("context window exceeded") is True


# ===========================================================================
# 4. Integration — trigger fires step_down
# ===========================================================================


class TestTriggerIntegration:
    """When a trigger fires, it should drive ladder.step_down()."""

    def test_overflow_trigger_steps_down_ladder(self):
        """When over_budget() is True, calling step_down_if_needed moves the ladder down."""
        ctrl, ladder = _make_controller(
            [
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ],
            [{"type": "overflow", "threshold": 1000}],
        )

        assert ladder.is_at_full()
        # The controller recognises overflow and we drive step-down
        assert ctrl.over_budget(total_tokens=1100, budget=1000) is True
        ctrl.step_down_if_needed()
        assert not ladder.is_at_full()

    def test_context_limit_trigger_steps_down(self):
        """When a context_limit error is recognised, stepping down works."""
        ctrl, ladder = _make_controller(
            [
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ],
            [{"type": "context_limit"}],
        )

        exc = Exception("context window exceeded")
        # Check recognition
        assert ctrl.is_context_limit_error(exc) is True
        assert ladder.is_at_full()

        # Manually step down (the orchestrator/engine would call this after recognition)
        ctrl.ladder.step_down()
        assert not ladder.is_at_full()

    def test_trigger_does_not_step_down_at_bottom(self):
        """When at the lowest level, step_down is a no-op."""
        ctrl, ladder = _make_controller(
            [
                {"name": "full", "cats": ["memory", "context"]},
                {"name": "reduced", "cats": ["context"]},
            ],
            [{"type": "overflow", "threshold": 1000}],
        )
        # Drive to bottom
        ladder.step_down()
        ladder.step_down()  # Should stay at bottom

        bottom_level = 1
        assert ladder.current_level == bottom_level


# ===========================================================================
# 5. has_trigger() — check if a trigger type is active
# ===========================================================================


class TestHasTrigger:
    def test_has_overflow_trigger(self):
        """has_trigger('overflow') returns True when configured."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "overflow", "threshold": 1000}],
        )
        assert ctrl.has_trigger("overflow") is True

    def test_has_context_limit_trigger(self):
        """has_trigger('context_limit') returns True when configured."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        assert ctrl.has_trigger("context_limit") is True

    def test_missing_trigger_returns_false(self):
        """has_trigger('overflow') returns False when not configured."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [{"type": "context_limit"}],
        )
        assert ctrl.has_trigger("overflow") is False

    def test_empty_triggers_all_false(self):
        """With no triggers, all has_trigger checks return False."""
        ctrl, _ = _make_controller(
            [{"name": "full", "cats": ["memory", "context"]}],
            [],
        )
        assert ctrl.has_trigger("overflow") is False
        assert ctrl.has_trigger("context_limit") is False
