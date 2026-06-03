"""Tests for DegradationConfig schema — FR1 (sr2-81).

Validates the Pydantic config models for the degradation subsystem:
- DegradationLevelConfig (name + keep_categories)
- DegradationTriggerConfig (type in {overflow, context_limit} + threshold)
- DegradationConfig (levels + triggers)
- PipelineConfig.degradation field
- ConfigError on invalid input
"""

import pytest
from pydantic import ValidationError

from sr2.config.models import (
    ConfigError,
    DegradationConfig,
    DegradationLevelConfig,
    DegradationTriggerConfig,
    PipelineConfig,
)


# ---------------------------------------------------------------------------
# 1. DegradationLevelConfig
# ---------------------------------------------------------------------------


class TestDegradationLevelConfig:
    def test_minimal_level(self):
        level = DegradationLevelConfig(name="full", keep_categories=["memory", "tools", "context", "history"])
        assert level.name == "full"
        assert level.keep_categories == ["memory", "tools", "context", "history"]

    def test_empty_keep_categories(self):
        """Bottom level may keep nothing."""
        level = DegradationLevelConfig(name="minimal", keep_categories=[])
        assert level.name == "minimal"
        assert level.keep_categories == []

    def test_name_is_required(self):
        with pytest.raises(ValidationError):
            DegradationLevelConfig(keep_categories=["memory"])

    def test_keep_categories_is_required(self):
        with pytest.raises(ValidationError):
            DegradationLevelConfig(name="full")

    def test_single_category(self):
        level = DegradationLevelConfig(name="context_only", keep_categories=["context"])
        assert level.keep_categories == ["context"]

    def test_model_dump_round_trip(self):
        level = DegradationLevelConfig(name="reduced", keep_categories=["context", "history"])
        dumped = level.model_dump()
        level2 = DegradationLevelConfig(**dumped)
        assert level == level2


# ---------------------------------------------------------------------------
# 2. DegradationTriggerConfig
# ---------------------------------------------------------------------------


class TestDegradationTriggerConfig:
    def test_overflow_trigger(self):
        trigger = DegradationTriggerConfig(type="overflow", threshold=1.2)
        assert trigger.type == "overflow"
        assert trigger.threshold == 1.2

    def test_context_limit_trigger(self):
        trigger = DegradationTriggerConfig(type="context_limit")
        assert trigger.type == "context_limit"
        assert trigger.threshold is None

    def test_threshold_optional(self):
        """threshold defaults to None."""
        trigger = DegradationTriggerConfig(type="overflow")
        assert trigger.threshold is None

    def test_threshold_int(self):
        trigger = DegradationTriggerConfig(type="overflow", threshold=150000)
        assert trigger.threshold == 150000

    def test_unknown_trigger_type_raises(self):
        with pytest.raises((ValidationError, ConfigError)):
            DegradationTriggerConfig(type="circuit_breaker", threshold=3)

    def test_unknown_trigger_type_custom_string(self):
        with pytest.raises((ValidationError, ConfigError)):
            DegradationTriggerConfig(type="my_custom_trigger")

    def test_type_is_required(self):
        with pytest.raises(ValidationError):
            DegradationTriggerConfig(threshold=1.5)

    def test_model_dump_round_trip(self):
        trigger = DegradationTriggerConfig(type="overflow", threshold=1.5)
        dumped = trigger.model_dump()
        trigger2 = DegradationTriggerConfig(**dumped)
        assert trigger == trigger2


# ---------------------------------------------------------------------------
# 3. DegradationConfig
# ---------------------------------------------------------------------------


class TestDegradationConfig:
    def test_minimal_config(self):
        levels = [
            DegradationLevelConfig(name="full", keep_categories=["memory", "tools", "context", "history"]),
            DegradationLevelConfig(name="reduced", keep_categories=["context", "history"]),
        ]
        cfg = DegradationConfig(levels=levels)
        assert len(cfg.levels) == 2
        assert cfg.triggers == []

    def test_with_triggers(self):
        levels = [DegradationLevelConfig(name="full", keep_categories=["memory"])]
        triggers = [DegradationTriggerConfig(type="overflow", threshold=1.2)]
        cfg = DegradationConfig(levels=levels, triggers=triggers)
        assert len(cfg.triggers) == 1
        assert cfg.triggers[0].type == "overflow"

    def test_empty_levels_raises(self):
        """At least one level is required."""
        with pytest.raises((ValidationError, ConfigError)):
            DegradationConfig(levels=[])

    def test_duplicate_level_names_raises(self):
        levels = [
            DegradationLevelConfig(name="full", keep_categories=["memory"]),
            DegradationLevelConfig(name="full", keep_categories=["context"]),
        ]
        with pytest.raises((ValidationError, ConfigError)):
            DegradationConfig(levels=levels)

    def test_model_dump_round_trip(self):
        cfg = DegradationConfig(
            levels=[
                DegradationLevelConfig(name="full", keep_categories=["memory", "context"]),
                DegradationLevelConfig(name="reduced", keep_categories=["context"]),
            ],
            triggers=[DegradationTriggerConfig(type="overflow", threshold=1.5)],
        )
        dumped = cfg.model_dump()
        cfg2 = DegradationConfig(**dumped)
        assert cfg == cfg2


# ---------------------------------------------------------------------------
# 4. PipelineConfig.degradation integration
# ---------------------------------------------------------------------------


class TestPipelineConfigDegradation:
    def _minimal_pipeline(self, **overrides):
        """Build a minimal PipelineConfig with one layer."""
        from sr2.config.models import LayerConfig, ResolverConfig

        base = {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static_template"}],
                }
            ]
        }
        base.update(overrides)
        return PipelineConfig(**base)

    def test_degradation_defaults_to_none(self):
        cfg = self._minimal_pipeline()
        assert cfg.degradation is None

    def test_degradation_accepts_config(self):
        cfg = self._minimal_pipeline(
            degradation={
                "levels": [
                    {"name": "full", "keep_categories": ["memory", "context", "history"]},
                    {"name": "reduced", "keep_categories": ["context", "history"]},
                ],
                "triggers": [{"type": "overflow", "threshold": 1.2}],
            }
        )
        assert cfg.degradation is not None
        assert len(cfg.degradation.levels) == 2
        assert cfg.degradation.levels[0].name == "full"
        assert len(cfg.degradation.triggers) == 1
        assert cfg.degradation.triggers[0].type == "overflow"

    def test_no_degradation_means_disabled(self):
        """Regression: without degradation config, pipeline constructs normally."""
        cfg = self._minimal_pipeline()
        assert cfg.degradation is None

    def test_degradation_model_dump_round_trip(self):
        cfg = self._minimal_pipeline(
            degradation={
                "levels": [
                    {"name": "full", "keep_categories": ["memory"]},
                ],
            }
        )
        dumped = cfg.model_dump()
        cfg2 = PipelineConfig(**dumped)
        assert cfg == cfg2

    def test_degradation_yaml_dict_construction(self):
        """Typical YAML input (nested dicts) should construct validly."""
        raw = {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static_template"}],
                },
                {
                    "name": "memory",
                    "target": "messages",
                    "resolvers": [{"type": "retrieval"}],
                },
            ],
            "degradation": {
                "levels": [
                    {
                        "name": "full",
                        "keep_categories": ["memory", "context", "history"],
                    },
                    {
                        "name": "no_memory",
                        "keep_categories": ["context", "history"],
                    },
                    {
                        "name": "context_only",
                        "keep_categories": ["context"],
                    },
                ],
                "triggers": [
                    {"type": "overflow", "threshold": 1.2},
                    {"type": "context_limit"},
                ],
            },
        }
        cfg = PipelineConfig(**raw)
        assert len(cfg.degradation.levels) == 3
        assert cfg.degradation.levels[0].name == "full"
        assert cfg.degradation.levels[2].name == "context_only"
        assert len(cfg.degradation.triggers) == 2
        assert cfg.degradation.triggers[0].type == "overflow"
        assert cfg.degradation.triggers[1].type == "context_limit"


# ---------------------------------------------------------------------------
# 5. ConfigError validation edge cases
# ---------------------------------------------------------------------------


class TestDegradationConfigValidation:
    def test_unknown_trigger_type_in_pipeline(self):
        """Unknown trigger type in a PipelineConfig should raise."""
        from sr2.config.models import LayerConfig, ResolverConfig

        raw = {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static_template"}],
                }
            ],
            "degradation": {
                "levels": [{"name": "full", "keep_categories": ["memory"]}],
                "triggers": [{"type": "circuit_breaker", "threshold": 3}],
            },
        }
        with pytest.raises((ValidationError, ConfigError)):
            PipelineConfig(**raw)

    def test_empty_levels_in_pipeline(self):
        raw = {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static_template"}],
                }
            ],
            "degradation": {
                "levels": [],
            },
        }
        with pytest.raises((ValidationError, ConfigError)):
            PipelineConfig(**raw)


# ---------------------------------------------------------------------------
# 6. Export check
# ---------------------------------------------------------------------------


class TestExports:
    def test_degradation_models_exported_from_config(self):
        """New models should be importable from sr2.config.models."""
        from sr2.config.models import (
            DegradationConfig,
            DegradationLevelConfig,
            DegradationTriggerConfig,
        )

        assert DegradationConfig is not None
        assert DegradationLevelConfig is not None
        assert DegradationTriggerConfig is not None

    def test_degradation_models_in_init(self):
        """Models should also be exported from sr2.config package __init__."""
        from sr2.config import (
            DegradationConfig,
            DegradationLevelConfig,
            DegradationTriggerConfig,
        )

        assert DegradationConfig is not None
        assert DegradationLevelConfig is not None
        assert DegradationTriggerConfig is not None
