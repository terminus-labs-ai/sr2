"""Tests for ToolOutputValidator."""

import pytest

from sr2.config.models import ToolValidationConfig
from sr2.tools.validation import ToolOutputValidator


SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer"},
        "score": {"type": "number"},
        "active": {"type": "boolean"},
        "tags": {"type": "array"},
    },
    "required": ["name", "count"],
    "additionalProperties": False,
}


class TestValidToolCalls:
    def test_valid_params_pass(self):
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test", "count": 5}, SAMPLE_SCHEMA)
        assert result.valid is True
        assert result.errors == []

    def test_valid_with_optional_fields(self):
        validator = ToolOutputValidator()
        result = validator.validate(
            {"name": "test", "count": 5, "score": 0.9, "active": True, "tags": ["a"]},
            SAMPLE_SCHEMA,
        )
        assert result.valid is True

    def test_empty_schema_passes_anything(self):
        validator = ToolOutputValidator()
        result = validator.validate({"anything": "goes"}, {"type": "object"})
        assert result.valid is True


class TestMissingRequiredFields:
    def test_missing_required_field_detected(self):
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test"}, SAMPLE_SCHEMA)
        assert result.valid is False
        assert any("Missing required field 'count'" in e for e in result.errors)

    def test_missing_required_with_default_repaired(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "default": 0},
            },
            "required": ["name", "count"],
        }
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test"}, schema)
        assert result.valid is True
        assert result.repaired_params["count"] == 0

    def test_missing_required_no_default_in_strict_mode(self):
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer", "default": 0}},
            "required": ["count"],
        }
        validator = ToolOutputValidator(strict=True)
        result = validator.validate({}, schema)
        assert result.valid is False


class TestTypeCoercion:
    def test_string_to_int_repaired(self):
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test", "count": "42"}, SAMPLE_SCHEMA)
        assert result.valid is True
        assert result.repaired_params["count"] == 42

    def test_float_to_int_repaired(self):
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test", "count": 5.0}, SAMPLE_SCHEMA)
        assert result.valid is True
        assert result.repaired_params["count"] == 5

    def test_string_to_number_repaired(self):
        schema = {
            "type": "object",
            "properties": {"value": {"type": "number"}},
        }
        validator = ToolOutputValidator()
        result = validator.validate({"value": "3.14"}, schema)
        assert result.valid is True
        assert result.repaired_params["value"] == 3.14

    def test_invalid_string_to_int_fails(self):
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test", "count": "not a number"}, SAMPLE_SCHEMA)
        assert any("expected integer" in e for e in result.errors)

    def test_strict_mode_rejects_type_mismatch(self):
        validator = ToolOutputValidator(strict=True)
        result = validator.validate({"name": "test", "count": "42"}, SAMPLE_SCHEMA)
        assert result.valid is False


class TestExtraFields:
    def test_extra_fields_stripped(self):
        validator = ToolOutputValidator()
        result = validator.validate(
            {"name": "test", "count": 5, "extra": "field"}, SAMPLE_SCHEMA
        )
        assert result.valid is True
        assert "extra" not in result.repaired_params

    def test_extra_fields_allowed_when_additional_properties_true(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        validator = ToolOutputValidator()
        result = validator.validate({"name": "test", "extra": "ok"}, schema)
        assert result.valid is True
        assert result.errors == []


class TestNullHandling:
    def test_null_non_nullable_flagged(self):
        validator = ToolOutputValidator()
        result = validator.validate({"name": None, "count": 5}, SAMPLE_SCHEMA)
        assert result.valid is False
        assert any("null but not nullable" in e for e in result.errors)

    def test_null_nullable_field_passes(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": ["string", "null"]}},
        }
        validator = ToolOutputValidator()
        result = validator.validate({"name": None}, schema)
        assert result.valid is True


class TestConfigModel:
    def test_defaults(self):
        config = ToolValidationConfig()
        assert config.enabled is True
        assert config.strict is False
        assert config.retry is False
        assert config.max_retries == 1

    def test_custom_config(self):
        config = ToolValidationConfig(strict=True, retry=True, max_retries=3)
        assert config.strict is True
        assert config.retry is True
        assert config.max_retries == 3

    def test_max_retries_bounds(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ToolValidationConfig(max_retries=0)
        with pytest.raises(ValidationError):
            ToolValidationConfig(max_retries=5)
