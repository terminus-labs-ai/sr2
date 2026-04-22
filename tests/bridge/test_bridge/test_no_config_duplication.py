"""Ensure bridge config never redefines SR2 core config fields.

SR2's PipelineConfig is the single source of truth for pipeline behavior.
Bridge config must only contain bridge-specific concerns (forwarding, sessions,
system_prompt transforms, logging). Any field that duplicates a PipelineConfig
concept is a violation.
"""

import inspect

from sr2.config.models import DegradationConfig, PipelineConfig
from sr2_bridge.config import BridgeConfig


# Fields that are SR2 pipeline concerns and must never appear in BridgeConfig
# or any of its nested models. Add to this list as violations are discovered.
_PIPELINE_FIELD_NAMES = {
    "circuit_breaker_threshold",
    "circuit_breaker_cooldown_seconds",
    "circuit_breaker_cooldown_minutes",
}


def _collect_all_fields(model_cls, prefix="") -> dict[str, str]:
    """Recursively collect all field names from a Pydantic model tree."""
    fields = {}
    for name, field_info in model_cls.model_fields.items():
        full = f"{prefix}.{name}" if prefix else name
        fields[name] = full
        annotation = model_cls.__annotations__.get(name)
        if annotation and hasattr(annotation, "model_fields"):
            fields.update(_collect_all_fields(annotation, full))
    return fields


def test_bridge_config_has_no_degradation_fields():
    """BridgeConfig must not define any circuit breaker / degradation fields."""
    bridge_fields = _collect_all_fields(BridgeConfig)
    violations = _PIPELINE_FIELD_NAMES & set(bridge_fields.keys())
    assert not violations, (
        f"BridgeConfig redefines SR2 pipeline fields: {violations}. "
        f"These belong in PipelineConfig.degradation, not bridge config."
    )


def test_bridge_config_has_no_degradation_submodel():
    """BridgeConfig must not have a 'degradation' field at all."""
    assert "degradation" not in BridgeConfig.model_fields, (
        "BridgeConfig.degradation exists — degradation config belongs in "
        "PipelineConfig.degradation. The bridge reads it from SR2."
    )
