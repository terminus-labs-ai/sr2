"""Auto-generates CONFIG.md and schema.json from Pydantic models.

Single source of truth: the Pydantic models in models.py define the config.
This module generates human-readable docs and machine-readable schema from them.

Usage:
    python -m schema_gen --format md > docs/configuration.md
    python -m schema_gen --format json > schema.json
    python -m schema_gen --format yaml > schema.yaml
    python -m schema_gen --format defaults > configs/defaults.yaml
"""

import json
import argparse
from sr2.config.models import (
    PipelineConfig,
    KVCacheConfig,
    CompactionConfig,
    CompactionRuleConfig,
    SummarizationConfig,
    RetrievalConfig,
    IntentDetectionConfig,
    ToolMaskingConfig,
    DegradationConfig,
    LayerConfig,
    ContentItemConfig,
    LLMConfig,
    LLMModelOverride,
)


# Hierarchical config sections for documentation.
# Each entry is (section_name, model_class, heading_level).
# heading_level 2 = ##, 3 = ###, 4 = ####
# Entries with model_class=None are group headers (no table, just a heading).
CONFIG_SECTIONS: list[tuple[str, type | None, int]] = [
    # --- Pipeline Config (per-interface) ---
    ("Pipeline Config", None, 2),
    ("Top-Level", PipelineConfig, 3),
    ("KV-Cache Strategy", KVCacheConfig, 3),
    ("Compaction", CompactionConfig, 3),
    ("Compaction Rule", CompactionRuleConfig, 4),
    ("Summarization", SummarizationConfig, 3),
    ("Retrieval", RetrievalConfig, 3),
    ("Intent Detection", IntentDetectionConfig, 3),
    ("Tool Masking", ToolMaskingConfig, 3),
    ("Per-Interface LLM Override", LLMConfig, 3),
    ("Per-Interface Model Override", LLMModelOverride, 4),
    ("Degradation", DegradationConfig, 3),
    ("Layers", LayerConfig, 3),
    ("Content Item", ContentItemConfig, 4),
]


def generate_json_schema() -> dict:
    """Generate JSON Schema for PipelineConfig."""
    schema = PipelineConfig.model_json_schema()
    schema["title"] = "PipelineConfig"
    return schema


def generate_yaml_schema() -> str:
    """Generate a YAML-formatted schema reference."""
    import yaml

    schema = generate_json_schema()
    return yaml.dump(schema, default_flow_style=False, sort_keys=False)


def generate_markdown() -> str:
    """Generate CONFIG.md — human-readable config reference."""
    lines = [
        "# SR2 Configuration Reference",
        "",
        "Auto-generated from Pydantic models. Single source of truth.",
        "",
        "## Config Inheritance",
        "",
        "```",
        "configs/defaults.yaml          ← Library defaults (all fields have defaults)",
        "  └── agent.yaml               ← Agent overrides (only specify what differs)",
        "       └── interfaces/x.yaml   ← Interface overrides (only specify what differs)",
        "```",
        "",
        "Resolution: deep merge, more specific wins. `extends: defaults` or `extends: agent`.",
        "",
    ]

    for section_name, model_class, level in CONFIG_SECTIONS:
        if model_class is None:
            # Group header — just a heading with no table
            lines.append(f"{'#' * level} {section_name}")
            lines.append("")
        else:
            lines.extend(_document_model(section_name, model_class, level))

    # Reference appendices
    lines.append("## Reference")
    lines.append("")
    lines.extend(_document_compaction_rules())
    lines.extend(_document_cache_policies())

    return "\n".join(lines)


def _document_model(section_name: str, model_class, level: int = 2) -> list[str]:
    """Generate markdown docs for a single Pydantic model."""
    heading = "#" * level
    lines = [
        f"{heading} {section_name}",
        "",
        f"Model: `{model_class.__name__}`",
        "",
    ]

    # Get the JSON schema for this model
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        lines.append("(No configurable fields)")
        lines.append("")
        return lines

    lines.append("| Field | Type | Default | Required | Description |")
    lines.append("|---|---|---|---|---|")

    for field_name, field_info in properties.items():
        field_type = _format_type(field_info)
        default = _format_default(field_info)
        is_required = "✅" if field_name in required else ""
        description = field_info.get("description", "")

        lines.append(
            f"| `{field_name}` | {field_type} | {default} | {is_required} | {description} |"
        )

    lines.append("")

    # Add examples for enum/literal fields
    for field_name, field_info in properties.items():
        if "enum" in field_info:
            values = ", ".join(f"`{v}`" for v in field_info["enum"])
            lines.append(f"**`{field_name}` values:** {values}")
            lines.append("")

    # YAML example
    lines.append("**Example:**")
    lines.append("```yaml")
    lines.extend(_generate_yaml_example(model_class))
    lines.append("```")
    lines.append("")

    return lines


def _format_type(field_info: dict) -> str:
    """Format a JSON Schema type for display."""
    if "anyOf" in field_info:
        types = []
        for opt in field_info["anyOf"]:
            if opt.get("type") == "null":
                continue
            types.append(opt.get("type", "any"))
        return " \\| ".join(types) + " \\| null"

    if "allOf" in field_info:
        ref = field_info["allOf"][0].get("$ref", "")
        return ref.split("/")[-1] if ref else "object"

    t = field_info.get("type", "any")
    if t == "array":
        items = field_info.get("items", {})
        item_type = items.get("type", "any")
        return f"list[{item_type}]"

    if "enum" in field_info:
        return "enum"

    return t


def _format_default(field_info: dict) -> str:
    """Format a default value for display."""
    if "default" not in field_info:
        return "—"
    default = field_info["default"]
    if default is None:
        return "`null`"
    if isinstance(default, bool):
        return f"`{str(default).lower()}`"
    if isinstance(default, str):
        return f'`"{default}"`'
    if isinstance(default, (int, float)):
        return f"`{default}`"
    if isinstance(default, list):
        if not default:
            return "`[]`"
        return f"`[{len(default)} items]`"
    return f"`{default}`"


def _generate_yaml_example(model_class) -> list[str]:
    """Generate a YAML example from a model's defaults."""
    import yaml

    # Build example data from schema defaults, handling models with required fields
    schema = model_class.model_json_schema()
    properties = schema.get("properties", {})
    data = {}
    for field_name, field_info in properties.items():
        if "default" in field_info:
            data[field_name] = field_info["default"]
        elif "enum" in field_info:
            data[field_name] = field_info["enum"][0]
        elif field_info.get("type") == "string":
            data[field_name] = f"<{field_name}>"
        elif field_info.get("type") == "integer":
            data[field_name] = 0
        elif field_info.get("type") == "boolean":
            data[field_name] = False
        elif field_info.get("type") == "array":
            data[field_name] = []
        else:
            data[field_name] = f"<{field_name}>"

    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return yaml_str.strip().split("\n")


def _document_compaction_rules() -> list[str]:
    """Document available compaction rules."""
    return [
        "### Compaction Rules Reference",
        "",
        "Each rule matches a `content_type` on conversation turns and replaces verbose content with a compact reference.",
        "",
        "| Strategy | Content Type | What It Does | Recoverable |",
        "|---|---|---|---|",
        "| `schema_and_sample` | `tool_output` | Replaces with line count + first 3 lines as sample | Yes — re-call the tool |",
        "| `reference` | `file_content` | Replaces with file path + metadata (lines, language, size) | Yes — `read_file(path)` |",
        "| `result_summary` | `code_execution` | Replaces with exit code (✓/✗) + first N lines of output | Yes — read result file |",
        "| `supersede` | `redundant_fetch` | Marks as `(superseded by turn N)` | Already in context from later fetch |",
        "| `collapse` | `confirmation` | Collapses to `→ ✓ tool_name(args)` one-liner | Original action already completed |",
        "",
        "**Options per rule:**",
        "",
        "```yaml",
        "compaction:",
        "  rules:",
        "    - type: tool_output",
        "      strategy: schema_and_sample",
        "      max_compacted_tokens: 80    # Max tokens for the compacted output",
        "      recovery_hint: true          # Adds 'Re-fetch with [tool]' hint",
        "",
        "    - type: file_content",
        "      strategy: reference",
        "      include_metadata:            # Which metadata to include in the reference",
        "        - line_count",
        "        - language",
        "        - size",
        "",
        "    - type: code_execution",
        "      strategy: result_summary",
        "      max_output_lines: 3          # How many stdout lines to keep",
        "",
        "    - type: redundant_fetch",
        "      strategy: supersede",
        "      # No options — metadata.superseded_by_turn provides the turn number",
        "",
        "    - type: confirmation",
        "      strategy: collapse",
        "      # No options — metadata.tool_name and metadata.args_summary used",
        "```",
        "",
        "**What does NOT get compacted:**",
        "- User messages (always preserved in full)",
        "- Agent reasoning / decisions (can't be re-derived)",
        "- Error messages (critical for debugging)",
        "- Content below `min_content_size` tokens",
        "- Turns in the `raw_window` (last N turns kept in full detail)",
        "",
    ]


def _document_cache_policies() -> list[str]:
    """Document available cache policies."""
    return [
        "### Cache Policies Reference",
        "",
        "Each layer declares a cache policy that controls when its content is recomputed.",
        "",
        "| Policy | Behavior | Best For |",
        "|---|---|---|",
        "| `immutable` | Never recompute after first call | System prompt, tool definitions, persona |",
        "| `refresh_on_topic_shift` | Recompute when intent detection flags a topic change | Retrieved memories, contextual data |",
        "| `refresh_on_state_change` | Recompute when state hash changes | Last known state (heartbeat polling) |",
        "| `append_only` | Always recompute (content grows each turn) | Conversation history |",
        "| `always_new` | Always recompute (changes every invocation) | Timestamps, dynamic instructions |",
        "| `per_invocation` | Always recompute, no caching between calls | A2A task input, event payloads |",
        "| `template_reuse` | Same as immutable — reuse across calls of this type | Sub-agent system prompts |",
        "",
        "**KV-Cache impact:** Layers earlier in the config (top) form the cached prefix.",
        "If an early layer changes, everything after it is re-computed by the LLM.",
        "Order layers from most stable to least stable.",
        "",
    ]


def generate_defaults_yaml() -> str:
    """Generate a comprehensive defaults.yaml showing every config option.

    Walks PipelineConfig and AgentYAMLConfig models, emitting commented YAML
    with descriptions above each field. Fields with real defaults are emitted
    uncommented; fields without defaults (required, no value) are commented out
    with a placeholder.
    """
    pipeline_defaults = PipelineConfig().model_dump()

    lines: list[str] = [
        "# SR2 - Pipeline Configuration Defaults",
        "# Auto-generated from Pydantic models. Do not edit by hand.",
        "# Re-generate with: uv run python -m schema_gen --format defaults",
        "#",
        "# Fields with defaults are shown uncommented.",
        "# Fields without defaults (required) are commented out with placeholders.",
        "",
    ]

    pipeline_schema = PipelineConfig.model_json_schema()
    pipeline_defs = pipeline_schema.get("$defs", {})
    _emit_section(
        lines,
        pipeline_schema.get("properties", {}),
        pipeline_defaults,
        pipeline_defs,
        indent=0,
        skip_fields={"extends"},
    )

    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")

    return "\n".join(lines)


def _emit_section(
    lines: list[str],
    properties: dict,
    defaults: dict,
    defs: dict,
    indent: int,
    skip_fields: set[str] | None = None,
) -> None:
    """Recursively emit YAML fields with comments."""
    prefix = "  " * indent
    for field_name, field_info in properties.items():
        if skip_fields and field_name in skip_fields:
            continue

        description = field_info.get("description", "")
        value = defaults.get(field_name)
        resolved = _resolve_ref(field_info, defs)

        # Emit description comment
        if description:
            for desc_line in _wrap_comment(description, prefix):
                lines.append(desc_line)

        # Determine if this is a nested object
        if resolved and resolved.get("type") == "object" and "properties" in resolved:
            _emit_object_field(lines, field_name, resolved, value, defs, indent, prefix)
        elif _is_array_of_objects(field_info, resolved, defs):
            _emit_array_field(lines, field_name, field_info, resolved, value, defs, indent, prefix)
        elif _is_dict_of_objects(field_info, resolved, defs):
            _emit_dict_field(lines, field_name, field_info, resolved, value, defs, indent, prefix)
        else:
            _emit_scalar_field(lines, field_name, field_info, value, indent, prefix)


def _emit_object_field(
    lines: list[str],
    field_name: str,
    resolved: dict,
    value,
    defs: dict,
    indent: int,
    prefix: str,
) -> None:
    """Emit a nested object field."""
    sub_defaults = value if isinstance(value, dict) else {}
    # If value is None or all children are None/commented, comment out the whole object
    if value is None or _all_children_empty(sub_defaults):
        lines.append(f"{prefix}# {field_name}:")
    else:
        lines.append(f"{prefix}{field_name}:")
        _emit_section(
            lines,
            resolved.get("properties", {}),
            sub_defaults,
            defs,
            indent + 1,
        )
    lines.append("")


def _emit_array_field(
    lines: list[str],
    field_name: str,
    field_info: dict,
    resolved: dict | None,
    value,
    defs: dict,
    indent: int,
    prefix: str,
) -> None:
    """Emit an array field, with example entries for arrays of objects."""
    item_schema = _get_array_item_schema(field_info, resolved, defs)

    if value and isinstance(value, list) and len(value) > 0:
        # Emit actual default entries
        lines.append(f"{prefix}{field_name}:")
        for item in value:
            if isinstance(item, dict) and item_schema:
                _emit_array_object_entry(lines, item, item_schema, defs, indent + 1)
            else:
                lines.append(f"{prefix}  - {_format_yaml_value(item)}")
    elif item_schema and item_schema.get("type") == "object" and "properties" in item_schema:
        # No defaults — emit empty list with a commented example below
        lines.append(f"{prefix}{field_name}: []")
        lines.append(f"{prefix}# Example entry:")
        _emit_commented_array_example(lines, item_schema, defs, indent + 1)
    else:
        lines.append(f"{prefix}{field_name}: []")
    lines.append("")


def _emit_dict_field(
    lines: list[str],
    field_name: str,
    field_info: dict,
    resolved: dict | None,
    value,
    defs: dict,
    indent: int,
    prefix: str,
) -> None:
    """Emit a dict field (e.g. interfaces, sessions)."""
    if value and isinstance(value, dict):
        lines.append(f"{prefix}{field_name}:")
        for k, v in value.items():
            lines.append(f"{prefix}  {k}:")
            if isinstance(v, dict):
                for vk, vv in v.items():
                    lines.append(f"{prefix}    {vk}: {_format_yaml_value(vv)}")
    else:
        # Emit commented example — resolve additionalProperties from field_info or resolved
        lines.append(f"{prefix}# {field_name}:")
        additional = None
        # Check field_info directly first
        if "additionalProperties" in field_info:
            additional = _resolve_ref(field_info["additionalProperties"], defs)
        elif resolved and "additionalProperties" in resolved:
            additional = _resolve_ref(resolved["additionalProperties"], defs)
        if additional and additional.get("type") == "object" and "properties" in additional:
            lines.append(f"{prefix}#   <name>:")
            for prop_name, prop_info in additional.get("properties", {}).items():
                desc = prop_info.get("description", "")
                if desc:
                    lines.append(f"{prefix}#     # {desc}")
                lines.append(f"{prefix}#     {prop_name}: {_placeholder_for(prop_info, defs)}")
        else:
            lines.append(f"{prefix}#   <name>: {{}}")
    lines.append("")


def _emit_scalar_field(
    lines: list[str],
    field_name: str,
    field_info: dict,
    value,
    indent: int,
    prefix: str,
) -> None:
    """Emit a scalar (non-object, non-array) field."""
    has_default = "default" in field_info
    is_anyof_with_default = "anyOf" in field_info and "default" in field_info

    if has_default or is_anyof_with_default or value is not None:
        lines.append(f"{prefix}{field_name}: {_format_yaml_value(value)}")
    else:
        # Required field with no default — comment it out
        placeholder = _placeholder_for(field_info, {})
        lines.append(f"{prefix}# {field_name}: {placeholder}")


def _emit_array_object_entry(
    lines: list[str],
    item: dict,
    item_schema: dict,
    defs: dict,
    indent: int,
) -> None:
    """Emit a single array entry that is an object."""
    prefix = "  " * indent
    first = True
    for prop_name in item_schema.get("properties", {}):
        if prop_name in item:
            if first:
                lines.append(f"{prefix}- {prop_name}: {_format_yaml_value(item[prop_name])}")
                first = False
            else:
                lines.append(f"{prefix}  {prop_name}: {_format_yaml_value(item[prop_name])}")


def _emit_commented_array_example(
    lines: list[str],
    item_schema: dict,
    defs: dict,
    indent: int,
) -> None:
    """Emit a single commented-out example entry for an array of objects."""
    prefix = "  " * indent
    first = True
    for prop_name, prop_info in item_schema.get("properties", {}).items():
        if "default" in prop_info:
            value = _format_yaml_value(prop_info["default"])
        else:
            value = _placeholder_for(prop_info, defs)
        if first:
            lines.append(f"{prefix}# - {prop_name}: {value}")
            first = False
        else:
            lines.append(f"{prefix}#   {prop_name}: {value}")


def _all_children_empty(d: dict) -> bool:
    """Check if all values in a dict are None (all children would be commented out)."""
    if not d:
        return True
    return all(v is None for v in d.values())


def _resolve_ref(field_info: dict, defs: dict) -> dict | None:
    """Resolve a $ref or allOf to the actual schema definition."""
    if not isinstance(field_info, dict):
        return None
    if "$ref" in field_info:
        ref_name = field_info["$ref"].split("/")[-1]
        return defs.get(ref_name)

    if "allOf" in field_info:
        for item in field_info["allOf"]:
            if "$ref" in item:
                ref_name = item["$ref"].split("/")[-1]
                return defs.get(ref_name)

    # anyOf with a $ref (Optional[SomeModel])
    if "anyOf" in field_info:
        for opt in field_info["anyOf"]:
            if "$ref" in opt:
                ref_name = opt["$ref"].split("/")[-1]
                return defs.get(ref_name)
            if "allOf" in opt:
                for item in opt["allOf"]:
                    if "$ref" in item:
                        ref_name = item["$ref"].split("/")[-1]
                        return defs.get(ref_name)

    # Inline object
    if field_info.get("type") == "object" and "properties" in field_info:
        return field_info

    return None


def _is_array_of_objects(field_info: dict, resolved: dict | None, defs: dict) -> bool:
    """Check if a field is an array whose items are objects."""
    if field_info.get("type") == "array":
        items = field_info.get("items", {})
        item_resolved = _resolve_ref(items, defs)
        if item_resolved and item_resolved.get("type") == "object":
            return True
        if items.get("type") == "object":
            return True
    return False


def _is_dict_of_objects(field_info: dict, resolved: dict | None, defs: dict) -> bool:
    """Check if a field is a dict (object with additionalProperties)."""
    if field_info.get("type") == "object" and "additionalProperties" in field_info:
        return True
    if resolved and resolved.get("type") == "object" and "additionalProperties" in resolved:
        return True
    return False


def _get_array_item_schema(field_info: dict, resolved: dict | None, defs: dict) -> dict | None:
    """Get the resolved schema for items of an array field."""
    items = field_info.get("items", {})
    item_resolved = _resolve_ref(items, defs)
    if item_resolved:
        return item_resolved
    return items if items else None


def _format_yaml_value(value) -> str:
    """Format a Python value as YAML inline."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if not value:
            return '""'
        # Quote if contains special chars
        if any(c in value for c in ":{}[]#&*!|>'\"%@`"):
            return f'"{value}"'
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        # Simple list of scalars
        items = ", ".join(_format_yaml_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        # For inline dicts, use flow style
        items = ", ".join(f"{k}: {_format_yaml_value(v)}" for k, v in value.items())
        return f"{{{items}}}"
    return str(value)


def _placeholder_for(field_info: dict, defs: dict) -> str:
    """Generate a placeholder value for a required field."""
    # If there's a default, use it
    if "default" in field_info:
        return _format_yaml_value(field_info["default"])
    if "enum" in field_info:
        return str(field_info["enum"][0])
    t = field_info.get("type", "")
    if t == "string":
        return "<required>"
    if t == "integer":
        return "0"
    if t == "number":
        return "0.0"
    if t == "boolean":
        return "false"
    if t == "array":
        return "[]"
    # Check anyOf — if it includes null type, this is Optional
    if "anyOf" in field_info:
        has_null = any(opt.get("type") == "null" for opt in field_info["anyOf"])
        if has_null:
            return "null"
        for opt in field_info["anyOf"]:
            if opt.get("type") == "string":
                return "<required>"
            if opt.get("type") == "integer":
                return "0"
    return "<required>"


def _wrap_comment(text: str, prefix: str, max_width: int = 80) -> list[str]:
    """Wrap a description into comment lines."""
    available = max_width - len(prefix) - 2  # "# " prefix
    if available < 30:
        available = 60
    words = text.split()
    result_lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > available:
            result_lines.append(f"{prefix}# {current}")
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        result_lines.append(f"{prefix}# {current}")
    return result_lines


def main():
    parser = argparse.ArgumentParser(description="Generate config documentation")
    parser.add_argument(
        "--format",
        choices=["md", "json", "yaml", "defaults"],
        default="md",
        help="Output format",
    )
    args = parser.parse_args()

    if args.format == "md":
        print(generate_markdown())
    elif args.format == "json":
        print(json.dumps(generate_json_schema(), indent=2))
    elif args.format == "yaml":
        print(generate_yaml_schema())
    elif args.format == "defaults":
        print(generate_defaults_yaml())


if __name__ == "__main__":
    main()
