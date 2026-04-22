"""Tests for ToolSchemaBudget — tool schema compression and budget enforcement.

Verifies behavior described in audit-fix-02.md:
- compress() strips verbose/redundant fields without losing functionality
- compress() preserves required-param descriptions and enum values
- compress() preserves overall schema structure
- enforce() returns schemas unchanged when within budget
- enforce() progressively degrades schemas when over budget
- enforce() returns empty list when nothing fits
- ToolBudgetResult carries correct accounting
"""

import json

import pytest

from sr2.tools.budget import ToolBudgetResult, ToolSchemaBudget, estimate_schema_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_schema(name: str = "my_tool") -> dict:
    """A bare-minimum tool schema with no parameters."""
    return {
        "name": name,
        "description": f"Does {name} things.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def _rich_schema(name: str = "rich_tool") -> dict:
    """A verbose schema with required + optional params and all noisy fields."""
    return {
        "name": name,
        "description": "A tool with many parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "description": "This is required and must be kept.",
                    "title": "Required Param",
                    "examples": ["foo", "bar"],
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-z]+$",
                },
                "optional_param": {
                    "type": "integer",
                    "description": "This is optional — description should be stripped.",
                    "title": "Optional Param",
                    "default": 42,
                    "examples": [1, 2, 3],
                },
                "enum_param": {
                    "type": "string",
                    "description": "Non-required but has enum — enum must survive.",
                    "enum": ["a", "b", "c"],
                    "title": "Enum Choice",
                    "default": "a",
                },
            },
            "required": ["required_param"],
        },
    }


def _estimate(schema: dict) -> int:
    """Helper: estimate tokens for a dict via the module function."""
    return estimate_schema_tokens(schema)


# ---------------------------------------------------------------------------
# estimate_schema_tokens
# ---------------------------------------------------------------------------

class TestEstimateSchemaTokens:
    def test_non_empty_schema_returns_positive(self):
        schema = _minimal_schema()
        assert _estimate(schema) >= 1

    def test_larger_schema_costs_more(self):
        small = {"name": "x"}
        big = _rich_schema()
        assert _estimate(big) > _estimate(small)

    def test_empty_dict_returns_at_least_one(self):
        # Edge case: even {} should return >= 1
        assert _estimate({}) >= 1

    def test_uses_json_size_heuristic(self):
        # Manually: json.dumps({}, sep=(",",":")) -> "{}" -> 2 chars -> max(1, 2//4) = 1
        result = _estimate({})
        assert result == max(1, len(json.dumps({}, separators=(",", ":"))) // 4)


# ---------------------------------------------------------------------------
# ToolSchemaBudget.compress()
# ---------------------------------------------------------------------------

class TestCompress:
    def setup_method(self):
        self.budget = ToolSchemaBudget()

    def test_returns_new_list_not_mutating_input(self):
        original = [_rich_schema()]
        result = self.budget.compress(original)
        assert result is not original
        assert result[0] is not original[0]

    def test_preserves_schema_count(self):
        schemas = [_rich_schema("a"), _rich_schema("b"), _minimal_schema("c")]
        result = self.budget.compress(schemas)
        assert len(result) == 3

    def test_preserves_name(self):
        schema = _rich_schema("my_tool")
        result = self.budget.compress([schema])
        assert result[0]["name"] == "my_tool"

    def test_preserves_top_level_description(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        assert "description" in result[0]
        assert result[0]["description"] == schema["description"]

    def test_preserves_required_param_description(self):
        """required_param is in 'required' list — its description must survive."""
        schema = _rich_schema()
        result = self.budget.compress([schema])
        props = result[0]["parameters"]["properties"]
        assert "description" in props["required_param"]
        assert props["required_param"]["description"] == (
            "This is required and must be kept."
        )

    def test_strips_optional_param_description(self):
        """optional_param is NOT required — its description must be stripped."""
        schema = _rich_schema()
        result = self.budget.compress([schema])
        props = result[0]["parameters"]["properties"]
        assert "description" not in props["optional_param"]

    def test_strips_title(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        for prop in result[0]["parameters"]["properties"].values():
            assert "title" not in prop

    def test_strips_examples(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        for prop in result[0]["parameters"]["properties"].values():
            assert "examples" not in prop

    def test_strips_minLength(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        for prop in result[0]["parameters"]["properties"].values():
            assert "minLength" not in prop

    def test_strips_maxLength(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        for prop in result[0]["parameters"]["properties"].values():
            assert "maxLength" not in prop

    def test_strips_pattern(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        for prop in result[0]["parameters"]["properties"].values():
            assert "pattern" not in prop

    def test_strips_default(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        for prop in result[0]["parameters"]["properties"].values():
            assert "default" not in prop

    def test_preserves_enum(self):
        """enum_param is not required, but its enum values must survive."""
        schema = _rich_schema()
        result = self.budget.compress([schema])
        props = result[0]["parameters"]["properties"]
        assert "enum" in props["enum_param"]
        assert props["enum_param"]["enum"] == ["a", "b", "c"]

    def test_preserves_parameter_type(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        props = result[0]["parameters"]["properties"]
        assert props["required_param"]["type"] == "string"
        assert props["optional_param"]["type"] == "integer"
        assert props["enum_param"]["type"] == "string"

    def test_preserves_required_list(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        assert result[0]["parameters"]["required"] == ["required_param"]

    def test_preserves_parameters_type_field(self):
        schema = _rich_schema()
        result = self.budget.compress([schema])
        assert result[0]["parameters"]["type"] == "object"

    def test_empty_input_returns_empty_list(self):
        result = self.budget.compress([])
        assert result == []

    def test_schema_without_parameters_survives(self):
        schema = {"name": "bare", "description": "No params."}
        result = self.budget.compress([schema])
        assert result[0]["name"] == "bare"
        assert result[0]["description"] == "No params."

    def test_compress_is_idempotent(self):
        """Compressing an already-compressed schema produces the same output."""
        schema = _rich_schema()
        once = self.budget.compress([schema])
        twice = self.budget.compress(once)
        assert once == twice

    def test_compressed_schema_smaller_than_original(self):
        schema = _rich_schema()
        original_tokens = _estimate(schema)
        compressed = self.budget.compress([schema])[0]
        compressed_tokens = _estimate(compressed)
        assert compressed_tokens < original_tokens


# ---------------------------------------------------------------------------
# ToolSchemaBudget.enforce()
# ---------------------------------------------------------------------------

class TestEnforce:
    def setup_method(self):
        self.budget = ToolSchemaBudget()

    def _make_schemas(self, n: int) -> list[dict]:
        return [_rich_schema(f"tool_{i}") for i in range(n)]

    def test_no_limit_returns_all_schemas_unchanged(self):
        schemas = self._make_schemas(3)
        result = self.budget.enforce(schemas, max_tokens=None)
        assert result.schemas == schemas
        assert result.final_count == 3
        assert result.truncated is False

    def test_under_budget_returns_all_unchanged(self):
        schemas = self._make_schemas(2)
        generous_budget = sum(_estimate(s) for s in schemas) + 10_000
        result = self.budget.enforce(schemas, max_tokens=generous_budget)
        assert result.schemas == schemas
        assert result.final_count == 2
        assert result.truncated is False

    def test_returns_ToolBudgetResult_instance(self):
        schemas = self._make_schemas(1)
        result = self.budget.enforce(schemas, max_tokens=10_000)
        assert isinstance(result, ToolBudgetResult)

    def test_result_has_correct_original_count(self):
        schemas = self._make_schemas(3)
        result = self.budget.enforce(schemas, max_tokens=10_000)
        assert result.original_count == 3

    def test_result_has_correct_original_tokens(self):
        schemas = self._make_schemas(2)
        expected = sum(_estimate(s) for s in schemas)
        result = self.budget.enforce(schemas, max_tokens=10_000)
        assert result.original_tokens == expected

    def test_empty_schemas_returns_empty_result(self):
        result = self.budget.enforce([], max_tokens=100)
        assert result.schemas == []
        assert result.final_count == 0
        assert result.truncated is False

    def test_tight_budget_drops_descriptions_first(self):
        """With a budget just enough for name+params but not full description."""
        schema = _rich_schema("verbose_tool")
        full_tokens = _estimate(schema)
        # Budget less than full but enough for name+params only
        no_desc = {"name": schema["name"], "parameters": schema["parameters"]}
        no_desc_tokens = _estimate(no_desc)

        # Budget: fits no_desc but not full schema
        budget = no_desc_tokens + 5  # just enough
        if full_tokens <= budget:
            pytest.skip("Schema too small for this test scenario")

        result = self.budget.enforce([schema], max_tokens=budget)
        assert result.final_count == 1
        # enforce() produces {"name": ..., "parameters": ...} — description key absent
        assert "description" not in result.schemas[0]

    def test_tight_budget_drops_params_before_entire_tool(self):
        """Schema fits only as minimal name+empty_params — should appear rather than be dropped.

        Uses a fat schema (many verbose parameters) so that:
          - no_desc form is > 200 tokens (clearing the `remaining > 200` guard)
          - minimal form (name + empty params) is small enough to fit
        This exercises the third branch of the progressive degradation strategy.
        """
        # Build a fat schema where no_desc >> 200 tokens
        props = {
            f"param_{i}": {
                "type": "string",
                "description": f"Parameter {i} with a verbose description to pad token count.",
                "examples": ["example_a", "example_b", "example_c"],
            }
            for i in range(30)
        }
        schema = {
            "name": "fat_tool",
            "description": "A tool with many verbose parameters.",
            "parameters": {"type": "object", "properties": props, "required": []},
        }

        minimal = {
            "name": schema["name"],
            "parameters": {"type": "object", "properties": {}},
        }
        minimal_tokens = _estimate(minimal)

        no_desc = {"name": schema["name"], "parameters": schema["parameters"]}
        no_desc_tokens = _estimate(no_desc)

        # Verify our fat schema has the right shape for this test
        assert no_desc_tokens > 200, "no_desc must exceed 200 to clear the guard"
        assert minimal_tokens < no_desc_tokens, "minimal must be smaller than no_desc"

        # Budget: above 200 (clears guard), above minimal_tokens, below no_desc_tokens
        budget = max(minimal_tokens + 5, 201)
        if no_desc_tokens <= budget:
            pytest.skip("Budget allows no-desc variant; fat schema not fat enough")

        result = self.budget.enforce([schema], max_tokens=budget)
        # Tool must survive as minimal form (params emptied), not be dropped entirely.
        assert result.final_count == 1
        surviving = result.schemas[0]
        assert surviving["name"] == schema["name"]
        # Parameters present but empty (minimal form — all properties stripped)
        assert surviving.get("parameters", {}).get("properties") == {}
        assert _estimate(surviving) <= budget

    def test_very_tight_budget_drops_entire_schema(self):
        """Budget too small for even one tool — result is empty list."""
        schemas = [_rich_schema("big_tool_a"), _rich_schema("big_tool_b")]
        result = self.budget.enforce(schemas, max_tokens=1)
        assert result.schemas == []
        assert result.final_count == 0

    def test_truncated_flag_set_when_budget_path_taken(self):
        """truncated=True any time the budget enforcement path runs (over-budget input)."""
        schemas = [_rich_schema("tool_a"), _rich_schema("tool_b")]
        result = self.budget.enforce(schemas, max_tokens=1)
        assert result.truncated is True

    def test_truncated_flag_set_even_if_all_tools_survive_in_degraded_form(self):
        """truncated=True even if a tool survives with description stripped (budget path taken)."""
        schema = _rich_schema("verbose_tool")
        full_tokens = _estimate(schema)
        no_desc = {"name": schema["name"], "parameters": schema["parameters"]}
        no_desc_tokens = _estimate(no_desc)
        budget = no_desc_tokens + 5
        if full_tokens <= budget:
            pytest.skip("Schema too small for this test scenario")
        result = self.budget.enforce([schema], max_tokens=budget)
        assert result.truncated is True

    def test_truncated_flag_false_when_all_fit(self):
        schemas = self._make_schemas(2)
        generous = sum(_estimate(s) for s in schemas) + 1000
        result = self.budget.enforce(schemas, max_tokens=generous)
        assert result.truncated is False

    def test_final_tokens_within_budget(self):
        """final_tokens must always be <= max_tokens."""
        schemas = self._make_schemas(5)
        total = sum(_estimate(s) for s in schemas)
        budget = total // 2  # Force truncation
        result = self.budget.enforce(schemas, max_tokens=budget)
        assert result.final_tokens <= budget

    def test_final_count_matches_schemas_length(self):
        schemas = self._make_schemas(3)
        result = self.budget.enforce(schemas, max_tokens=10_000)
        assert result.final_count == len(result.schemas)

    def test_result_schemas_are_valid_dicts(self):
        schemas = self._make_schemas(3)
        result = self.budget.enforce(schemas, max_tokens=10_000)
        for s in result.schemas:
            assert isinstance(s, dict)
            assert "name" in s

    def test_multiple_schemas_first_fits_last_dropped(self):
        """With a budget that fits exactly one schema, second should be dropped."""
        schema_a = _rich_schema("tool_a")
        schema_b = _rich_schema("tool_b")
        tokens_a = _estimate(schema_a)
        # Budget: fits a, not both
        budget = tokens_a
        result = self.budget.enforce([schema_a, schema_b], max_tokens=budget)
        # At least the first one should fit (possibly slightly modified)
        assert result.final_count >= 1
        # All surviving schemas fit within budget
        assert result.final_tokens <= budget

    def test_enforce_preserves_tool_name_in_survivors(self):
        schema = _rich_schema("my_tool")
        result = self.budget.enforce([schema], max_tokens=10_000)
        assert result.schemas[0]["name"] == "my_tool"

    def test_original_tokens_preserved_in_over_budget_result(self):
        """original_tokens in ToolBudgetResult reflects pre-truncation token count."""
        schemas = self._make_schemas(3)
        expected_original = sum(_estimate(s) for s in schemas)
        # Force truncation with a tight budget
        result = self.budget.enforce(schemas, max_tokens=1)
        assert result.original_tokens == expected_original

    def test_enforce_with_zero_max_tokens_returns_all(self):
        """max_tokens=0 is falsy — treated as no limit, returns all schemas unchanged."""
        schemas = self._make_schemas(2)
        result = self.budget.enforce(schemas, max_tokens=0)
        # 0 is falsy: `if not max_tokens` guard fires, no truncation applied
        assert result.schemas == schemas
        assert result.truncated is False


# ---------------------------------------------------------------------------
# Integration: compress then enforce
# ---------------------------------------------------------------------------

class TestCompressThenEnforce:
    def setup_method(self):
        self.budget = ToolSchemaBudget()

    def test_compress_then_enforce_within_budget(self):
        schemas = [_rich_schema(f"t{i}") for i in range(4)]
        compressed = self.budget.compress(schemas)
        total_compressed = sum(_estimate(s) for s in compressed)
        result = self.budget.enforce(compressed, max_tokens=total_compressed + 100)
        assert result.final_count == 4
        assert result.truncated is False

    def test_compress_reduces_tokens_before_enforce(self):
        """Compressing before enforcing leaves more budget room."""
        schemas = [_rich_schema("t")]
        compressed = self.budget.compress(schemas)
        original_tokens = _estimate(schemas[0])
        compressed_tokens = _estimate(compressed[0])
        # Use a budget between compressed and original size
        budget = (original_tokens + compressed_tokens) // 2
        if compressed_tokens >= original_tokens:
            pytest.skip("No token reduction in this environment")
        result = self.budget.enforce(compressed, max_tokens=budget)
        # With compressed schemas, the tool should fit
        assert result.final_count == 1

    def test_compress_then_enforce_empty_input(self):
        compressed = self.budget.compress([])
        result = self.budget.enforce(compressed, max_tokens=500)
        assert result.schemas == []
        assert result.final_count == 0
