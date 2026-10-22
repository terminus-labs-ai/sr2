"""sr2-94 / AC4 — ContinuityConfig schema (unit layer, config half of the cage).

Case-table rows covered here:
  - AC4-r7: ContinuityConfig(recent_token_budget=0) fails validation — the
    recent window must be positive.
  - AC4-r8: ContinuityConfig(recent_token_budget=12000, hard_token_budget=16000)
    parses; fields accessible with expected values.
  - AC4-r9: PipelineConfig without a continuity section loads unchanged —
    continuity config is absent (None / not set), i.e. absent continuity
    means disabled (implementation note: continuity-disabled configurations
    must retain current replay behavior — the resolver-side half of that row
    is AC4-r6 in tests/pipeline/resolvers/test_working_set.py).

Red mechanism (deliberate): ``ContinuityConfig`` is imported at module level
from its declared landing spot, ``sr2.config.models``. Until the implementer
adds the model, collection fails with ImportError naming exactly that symbol
— the expected red, not scaffolding.

The exception type raised by a failed model validation is deliberately NOT
pinned beyond pydantic's ValidationError umbrella: Pydantic wraps the repo's
ConfigError (like DegradationConfig's validators do) or raises directly; both
are acceptable. What IS pinned: instantiation fails, and the message names
the offending field so the failure is diagnosable (config-validation
requirement of the case table).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sr2.config.models import ContinuityConfig


# ---------------------------------------------------------------------------
# AC4-r7 — zero recent window is invalid
# ---------------------------------------------------------------------------


class TestRecentBudgetMustBePositive:
    def test_zero_recent_token_budget_rejected(self):
        """AC4-r7: ContinuityConfig(recent_token_budget=0) raises — the
        recent window budget must be positive."""
        with pytest.raises(ValidationError) as excinfo:
            ContinuityConfig(recent_token_budget=0)
        assert "recent_token_budget" in str(excinfo.value), (
            "the validation failure must name recent_token_budget so the "
            "operator can locate the offending field"
        )

    def test_negative_recent_token_budget_rejected(self):
        """Same rule as AC4-r7, negative side: a negative window is not a
        positive budget either."""
        with pytest.raises(ValidationError):
            ContinuityConfig(recent_token_budget=-1)


# ---------------------------------------------------------------------------
# AC4-r8 — valid config parses with expected field values
# ---------------------------------------------------------------------------


class TestContinuityConfigParsing:
    def test_valid_budgets_parse_with_expected_values(self):
        """AC4-r8: both budgets accessible with the values configured."""
        cfg = ContinuityConfig(recent_token_budget=12000, hard_token_budget=16000)
        assert cfg.recent_token_budget == 12000
        assert cfg.hard_token_budget == 16000

    def test_recent_budget_may_equal_hard_budget(self):
        """Degenerate but valid: the whole hard budget is the recent window."""
        cfg = ContinuityConfig(recent_token_budget=16000, hard_token_budget=16000)
        assert cfg.recent_token_budget == 16000
        assert cfg.hard_token_budget == 16000


# ---------------------------------------------------------------------------
# AC4-r9 — absent continuity means disabled (config side)
# ---------------------------------------------------------------------------


class TestPipelineConfigBackCompat:
    def test_pipeline_config_without_continuity_loads_and_field_absent(self):
        """AC4-r9 (config side): an existing-style PipelineConfig with no
        continuity section loads without error and carries no continuity
        config (None / absent)."""
        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig

        cfg = PipelineConfig(
            layers=[
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[ResolverConfig(type="session")],
                )
            ]
        )
        assert cfg.layers[0].name == "conversation"
        continuity = getattr(cfg, "continuity", None)
        assert continuity is None, (
            "a PipelineConfig with no continuity section must carry no "
            f"continuity config; got {continuity!r}"
        )
