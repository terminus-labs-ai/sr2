import yaml
from pathlib import Path

from schema_gen import (
    CONFIG_SECTIONS,
    generate_json_schema,
    generate_markdown,
    generate_yaml_schema,
    generate_defaults_yaml,
    _format_type,
    _format_default,
    _generate_yaml_example,
)
from sr2.config.models import PipelineConfig


DEFAULTS_PATH = str(Path(__file__).parent.parent.parent.parent / "configs" / "defaults.yaml")


class TestGenerateJsonSchema:
    def test_returns_dict_with_defs(self):
        schema = generate_json_schema()
        assert isinstance(schema, dict)
        assert "$defs" in schema

    def test_validates_defaults_yaml(self):
        """Generated defaults.yaml should load into PipelineConfig."""
        with open(DEFAULTS_PATH) as f:
            raw = yaml.safe_load(f)
        pipeline_fields = {
            k: v for k, v in raw.items()
            if v is not None and k in PipelineConfig.model_fields
        }
        config = PipelineConfig(**pipeline_fields)
        assert config.token_budget == 32000


class TestGenerateMarkdown:
    def test_includes_all_section_names(self):
        md = generate_markdown()
        for section_name, _, level in CONFIG_SECTIONS:
            heading = "#" * level
            assert f"{heading} {section_name}" in md

    def test_includes_compaction_rules_reference(self):
        md = generate_markdown()
        assert "Compaction Rules Reference" in md
        assert "schema_and_sample" in md
        assert "reference" in md
        assert "result_summary" in md
        assert "supersede" in md
        assert "collapse" in md

    def test_includes_cache_policies_reference(self):
        md = generate_markdown()
        assert "Cache Policies Reference" in md
        assert "immutable" in md
        assert "append_only" in md


class TestGenerateYamlSchema:
    def test_returns_valid_yaml(self):
        yaml_str = generate_yaml_schema()
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert "properties" in parsed


class TestFieldDescriptions:
    def test_every_pipeline_field_has_description(self):
        """Every field in PipelineConfig (and nested models) should have a description."""
        schema = PipelineConfig.model_json_schema()
        properties = schema.get("properties", {})
        missing = []
        for field_name, field_info in properties.items():
            if not field_info.get("description"):
                missing.append(field_name)
        assert missing == [], f"Fields missing description: {missing}"


class TestFormatType:
    def test_handles_literal_enum(self):
        field_info = {"enum": ["a", "b"], "type": "string"}
        assert _format_type(field_info) == "enum"

    def test_handles_list(self):
        field_info = {"type": "array", "items": {"type": "string"}}
        assert _format_type(field_info) == "list[string]"

    def test_handles_optional(self):
        field_info = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = _format_type(field_info)
        assert "null" in result
        assert "string" in result


class TestFormatDefault:
    def test_handles_bool(self):
        assert _format_default({"default": True}) == "`true`"
        assert _format_default({"default": False}) == "`false`"

    def test_handles_str(self):
        assert _format_default({"default": "hello"}) == '`"hello"`'

    def test_handles_int(self):
        assert _format_default({"default": 42}) == "`42`"

    def test_handles_none(self):
        assert _format_default({"default": None}) == "`null`"

    def test_handles_list(self):
        assert _format_default({"default": []}) == "`[]`"
        assert _format_default({"default": ["a", "b"]}) == "`[2 items]`"

    def test_handles_missing(self):
        assert _format_default({}) == "\u2014"


class TestGenerateYamlExample:
    def test_produces_parseable_yaml(self):
        from sr2.config.models import CompactionConfig
        lines = _generate_yaml_example(CompactionConfig)
        yaml_str = "\n".join(lines)
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert "enabled" in parsed


class TestGenerateDefaultsYaml:
    def test_contains_header(self):
        output = generate_defaults_yaml()
        assert "Auto-generated from Pydantic models" in output
        assert "uv run python -m schema_gen --format defaults" in output

    def test_contains_pipeline_section(self):
        output = generate_defaults_yaml()
        assert "token_budget: 32000" in output
        assert "pre_rot_threshold: 0.25" in output

    def test_contains_kv_cache_defaults(self):
        output = generate_defaults_yaml()
        assert "kv_cache:" in output
        assert "strategy: append_only" in output
        assert "compaction_timing: post_llm_async" in output

    def test_contains_descriptions_as_comments(self):
        output = generate_defaults_yaml()
        # Check that field descriptions appear as comments
        assert "# Total token budget for the context window" in output
        assert "# KV-cache optimization strategy" in output
        assert "# Enable compaction" in output

    def test_skips_extends_field(self):
        output = generate_defaults_yaml()
        # extends should not appear as a config field
        assert "\nextends:" not in output
        assert "\n# extends:" not in output

    def test_commented_required_fields(self):
        """Required fields without defaults should be commented out."""
        output = generate_defaults_yaml()
        # CompactionRuleConfig.type and .strategy are required
        assert "# - type: <required>" in output
        assert "#   strategy: <required>" in output

    def test_optional_fields_show_null(self):
        """Optional fields (with None default) should show null."""
        output = generate_defaults_yaml()
        assert "pull_exporter: null" in output

    def test_array_example_shows_schema_defaults(self):
        """Commented array examples should use schema defaults, not placeholders."""
        output = generate_defaults_yaml()
        # CompactionRuleConfig.max_compacted_tokens defaults to 80
        assert "max_compacted_tokens: 80" in output
        # CompactionRuleConfig.recovery_hint defaults to false
        assert "recovery_hint: false" in output

    def test_pipeline_fields_parseable_after_stripping_comments(self):
        """The uncommented pipeline fields should be valid YAML loadable by PipelineConfig."""
        output = generate_defaults_yaml()
        # Extract just the uncommented lines (non-comment, non-empty)
        # and parse as YAML to ensure it's valid
        uncommented_lines = []
        for line in output.split("\n"):
            stripped = line.lstrip()
            if stripped and not stripped.startswith("#"):
                uncommented_lines.append(line)
        yaml_str = "\n".join(uncommented_lines)
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert parsed["token_budget"] == 32000

    def test_defaults_yaml_file_matches_generator(self):
        """configs/defaults.yaml should match the generator output."""
        with open(DEFAULTS_PATH) as f:
            file_content = f.read()
        generated = generate_defaults_yaml()
        assert file_content.strip() == generated.strip()
