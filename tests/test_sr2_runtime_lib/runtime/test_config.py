"""Tests for SR2Runtime agent configuration."""

import pytest
from pydantic import ValidationError

from sr2.runtime.config import (
    AgentConfig,
    ContextConfig,
    ModelConfig,
    OutputConfig,
    PersonaConfig,
    ToolConfig,
)

# -- Fixtures / helpers -------------------------------------------------------

MINIMAL_DICT = {
    "name": "edi",
    "model": {"name": "qwen3:32b"},
    "persona": {"system_prompt": "You are EDI."},
}

FULL_YAML = """\
name: edi
description: A helpful assistant
model:
  provider: ollama
  name: qwen3:32b
  base_url: http://localhost:11434
  temperature: 0.5
  max_tokens: 8192
  extra:
    num_ctx: 32768
persona:
  system_prompt: You are EDI, a helpful assistant.
context:
  context_window: 65536
  conversation:
    active_turns: 20
    buffer_turns: 10
    compaction: summarize
  memory:
    enabled: true
  pipeline_override:
    degradation:
      circuit_breaker_threshold: 5
tools:
  - name: search
    module: tools.search
    config:
      max_results: 10
  - name: read_file
    module: tools.filesystem
output:
  format: structured
  schema_ref: my_schema
  max_tool_iterations: 25
"""


@pytest.fixture
def full_yaml_file(tmp_path):
    p = tmp_path / "agent.yaml"
    p.write_text(FULL_YAML)
    return p


@pytest.fixture
def minimal_yaml_file(tmp_path):
    p = tmp_path / "agent.yaml"
    p.write_text(
        "name: edi\nmodel:\n  name: qwen3:32b\npersona:\n  system_prompt: You are EDI.\n"
    )
    return p


# -- from_yaml / from_dict ---------------------------------------------------


class TestLoadFromYAML:
    def test_full_yaml(self, full_yaml_file):
        cfg = AgentConfig.from_yaml(full_yaml_file)
        assert cfg.name == "edi"
        assert cfg.description == "A helpful assistant"
        assert cfg.model.name == "qwen3:32b"
        assert cfg.model.temperature == 0.5
        assert cfg.model.max_tokens == 8192
        assert cfg.model.extra == {"num_ctx": 32768}
        assert cfg.persona.system_prompt == "You are EDI, a helpful assistant."
        assert cfg.context.context_window == 65536
        assert cfg.context.conversation["active_turns"] == 20
        assert cfg.context.memory["enabled"] is True
        assert cfg.context.pipeline_override["degradation"]["circuit_breaker_threshold"] == 5
        assert len(cfg.tools) == 2
        assert cfg.tools[0].name == "search"
        assert cfg.tools[1].module == "tools.filesystem"
        assert cfg.output.format == "structured"
        assert cfg.output.schema_ref == "my_schema"
        assert cfg.output.max_tool_iterations == 25

    def test_minimal_yaml(self, minimal_yaml_file):
        cfg = AgentConfig.from_yaml(minimal_yaml_file)
        assert cfg.name == "edi"
        assert cfg.model.name == "qwen3:32b"
        assert cfg.persona.system_prompt == "You are EDI."


class TestLoadFromDict:
    def test_minimal_dict(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.name == "edi"
        assert cfg.model.name == "qwen3:32b"
        assert cfg.persona.system_prompt == "You are EDI."

    def test_round_trip_matches(self, full_yaml_file):
        from_yaml = AgentConfig.from_yaml(full_yaml_file)
        from_dict = AgentConfig.from_dict(from_yaml.model_dump())
        assert from_yaml == from_dict


# -- Defaults -----------------------------------------------------------------


class TestDefaults:
    def test_context_defaults(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.context.context_window == 32768
        assert cfg.context.conversation["active_turns"] == 10
        assert cfg.context.conversation["buffer_turns"] == 5
        assert cfg.context.conversation["compaction"] == "summarize"
        assert cfg.context.memory["enabled"] is False
        assert cfg.context.pipeline_override == {}

    def test_tools_default_empty(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.tools == []

    def test_output_defaults(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.output.format == "freeform"
        assert cfg.output.schema_ref is None
        assert cfg.output.max_tool_iterations == 10

    def test_model_defaults(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.model.provider == "ollama"
        assert cfg.model.base_url == "http://localhost:11434"
        assert cfg.model.temperature == 0.3
        assert cfg.model.max_tokens == 4096
        assert cfg.model.extra == {}

    def test_description_defaults_empty(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.description == ""


# -- Validation errors --------------------------------------------------------


class TestValidation:
    def test_missing_name(self):
        data = {
            "model": {"name": "qwen3:32b"},
            "persona": {"system_prompt": "hi"},
        }
        with pytest.raises(ValidationError, match="name"):
            AgentConfig.from_dict(data)

    def test_missing_model_name(self):
        data = {
            "name": "edi",
            "model": {"provider": "ollama"},
            "persona": {"system_prompt": "hi"},
        }
        with pytest.raises(ValidationError, match="name"):
            AgentConfig.from_dict(data)

    def test_missing_persona_system_prompt(self):
        data = {
            "name": "edi",
            "model": {"name": "qwen3:32b"},
            "persona": {},
        }
        with pytest.raises(ValidationError, match="system_prompt"):
            AgentConfig.from_dict(data)

    def test_missing_model_entirely(self):
        data = {
            "name": "edi",
            "persona": {"system_prompt": "hi"},
        }
        with pytest.raises(ValidationError, match="model"):
            AgentConfig.from_dict(data)

    def test_missing_persona_entirely(self):
        data = {
            "name": "edi",
            "model": {"name": "qwen3:32b"},
        }
        with pytest.raises(ValidationError, match="persona"):
            AgentConfig.from_dict(data)


# -- pipeline_override --------------------------------------------------------


class TestPipelineOverride:
    def test_arbitrary_fields_pass_through(self):
        data = {
            **MINIMAL_DICT,
            "context": {
                "pipeline_override": {
                    "degradation": {"circuit_breaker_threshold": 5},
                    "intent_detection": {"enabled": False},
                    "custom_field": [1, 2, 3],
                },
            },
        }
        cfg = AgentConfig.from_dict(data)
        assert cfg.context.pipeline_override["degradation"]["circuit_breaker_threshold"] == 5
        assert cfg.context.pipeline_override["intent_detection"]["enabled"] is False
        assert cfg.context.pipeline_override["custom_field"] == [1, 2, 3]

    def test_empty_override_is_default(self):
        cfg = AgentConfig.from_dict(MINIMAL_DICT)
        assert cfg.context.pipeline_override == {}


# -- ToolConfig ---------------------------------------------------------------


class TestToolConfig:
    def test_construction(self):
        t = ToolConfig(name="search", module="tools.search")
        assert t.name == "search"
        assert t.module == "tools.search"
        assert t.config == {}

    def test_with_config(self):
        t = ToolConfig(name="search", module="tools.search", config={"max_results": 5})
        assert t.config["max_results"] == 5

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError, match="name"):
            ToolConfig(module="tools.search")  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="module"):
            ToolConfig(name="search")  # type: ignore[call-arg]


# -- ContextConfig standalone -------------------------------------------------


class TestContextConfig:
    def test_defaults_standalone(self):
        ctx = ContextConfig()
        assert ctx.context_window == 32768
        assert ctx.conversation["active_turns"] == 10
        assert ctx.memory["enabled"] is False
        assert ctx.pipeline_override == {}

    def test_custom_values(self):
        ctx = ContextConfig(context_window=128000, memory={"enabled": True, "backend": "pg"})
        assert ctx.context_window == 128000
        assert ctx.memory["enabled"] is True
        assert ctx.memory["backend"] == "pg"
