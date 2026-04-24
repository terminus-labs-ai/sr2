"""Tool output validator — validates LLM tool call parameters against JSON schemas."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating tool call parameters against a schema."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    repaired_params: dict | None = None


class ToolOutputValidator:
    """Validates and optionally repairs LLM tool call parameters against JSON schemas.

    Repair strategies for common LLM errors:
    - Missing required field with a default -> use default
    - String where int/float expected -> attempt conversion
    - Extra fields not in schema -> strip them
    - Null where non-nullable expected -> flag as error (don't guess)
    """

    def __init__(self, strict: bool = False):
        self._strict = strict

    def validate(self, params: dict, schema: dict) -> ValidationResult:
        """Validate parameters against a JSON schema.

        In strict mode, any error is a failure. In lenient mode, repairable
        errors are fixed and the repaired params are returned.
        """
        errors: list[str] = []
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        repaired = dict(params)

        # Check for missing required fields
        for field_name in required:
            if field_name not in repaired:
                prop_schema = properties.get(field_name, {})
                default = prop_schema.get("default")
                if default is not None and not self._strict:
                    repaired[field_name] = default
                    errors.append(f"Missing required field '{field_name}', used default: {default}")
                else:
                    errors.append(f"Missing required field '{field_name}'")

        # Check types and strip extras
        extra_fields = set(repaired.keys()) - set(properties.keys())
        if extra_fields:
            if schema.get("additionalProperties", True) is False:
                for ef in extra_fields:
                    errors.append(f"Extra field '{ef}' not in schema")
                    if not self._strict:
                        del repaired[ef]
            # If additionalProperties is not explicitly False, extras are allowed

        # Type checking and repair
        for field_name, prop_schema in properties.items():
            if field_name not in repaired:
                continue
            value = repaired[field_name]
            expected_type = prop_schema.get("type")
            if expected_type and value is not None:
                converted, repaired_value = self._check_type(
                    field_name, value, expected_type, errors
                )
                if converted and not self._strict:
                    repaired[field_name] = repaired_value

            # Null check for non-nullable
            if value is None:
                nullable = prop_schema.get("nullable", False)
                if not nullable and "null" not in _get_types(prop_schema):
                    errors.append(f"Field '{field_name}' is null but not nullable")

        if self._strict and errors:
            return ValidationResult(valid=False, errors=errors)

        # In lenient mode, if all errors were repairable, return valid with repairs
        has_unrecoverable = any(
            e.startswith("Missing required field") and "used default" not in e for e in errors
        ) or any(e.startswith("Field") and "null but not nullable" in e for e in errors)

        if has_unrecoverable:
            return ValidationResult(valid=False, errors=errors, repaired_params=repaired)

        if errors:
            return ValidationResult(valid=True, errors=errors, repaired_params=repaired)

        return ValidationResult(valid=True, errors=[])

    def _check_type(
        self, field_name: str, value: object, expected: str, errors: list[str]
    ) -> tuple[bool, object]:
        """Check type and attempt repair. Returns (was_converted, repaired_value)."""
        if expected == "integer" and not isinstance(value, int):
            if isinstance(value, str):
                try:
                    converted = int(value)
                    errors.append(f"Field '{field_name}': coerced string to integer")
                    return True, converted
                except (ValueError, TypeError):
                    errors.append(
                        f"Field '{field_name}': expected integer, got {type(value).__name__}"
                    )
            elif isinstance(value, float) and value == int(value):
                errors.append(f"Field '{field_name}': coerced float to integer")
                return True, int(value)
            else:
                errors.append(f"Field '{field_name}': expected integer, got {type(value).__name__}")
        elif expected == "number" and not isinstance(value, (int, float)):
            if isinstance(value, str):
                try:
                    converted = float(value)
                    errors.append(f"Field '{field_name}': coerced string to number")
                    return True, converted
                except (ValueError, TypeError):
                    errors.append(
                        f"Field '{field_name}': expected number, got {type(value).__name__}"
                    )
            else:
                errors.append(f"Field '{field_name}': expected number, got {type(value).__name__}")
        elif expected == "string" and not isinstance(value, str):
            errors.append(f"Field '{field_name}': expected string, got {type(value).__name__}")
        elif expected == "boolean" and not isinstance(value, bool):
            errors.append(f"Field '{field_name}': expected boolean, got {type(value).__name__}")
        elif expected == "array" and not isinstance(value, list):
            errors.append(f"Field '{field_name}': expected array, got {type(value).__name__}")
        elif expected == "object" and not isinstance(value, dict):
            errors.append(f"Field '{field_name}': expected object, got {type(value).__name__}")
        return False, value


def _get_types(prop_schema: dict) -> set[str]:
    """Extract the set of allowed types from a property schema."""
    t = prop_schema.get("type")
    if isinstance(t, list):
        return set(t)
    if isinstance(t, str):
        return {t}
    return set()
