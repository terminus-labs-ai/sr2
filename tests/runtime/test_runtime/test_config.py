"""Tests for runtime config Pydantic models."""

import os

import pytest
import yaml
from pydantic import ValidationError

from sr2_runtime.config import (
    AgentYAMLConfig,
    RuntimeConfig,
    RuntimeDatabaseConfig,
    RuntimeLLMConfig,
    RuntimeLoopConfig,
    RuntimeSessionConfig,
    InterfaceConfig,
    InterfaceSessionConfig,
    LLMModelConfig,
    MCPServerConfig,
    ModelParams,
)


class TestRuntimeConfig:
    """Tests for the top-level RuntimeConfig model."""

    def test_defaults(self):
        config = RuntimeConfig()
        assert config.database.pool_min == 2
        assert config.database.pool_max == 10
        assert config.database.url is None
        assert config.llm.model.name == "claude-sonnet-4-20250514"
        assert config.llm.fast_model.name == "claude-haiku-4-5-20251001"
        assert config.llm.model.max_tokens == 4096
        assert config.loop.max_iterations == 25
        assert config.session.max_turns == 200
        assert config.session.idle_timeout_minutes == 60

    def test_every_field_has_description(self):
        """Every field in RuntimeConfig and sub-models should have a description."""
        models = [
            RuntimeConfig,
            RuntimeDatabaseConfig,
            RuntimeLLMConfig,
            RuntimeLoopConfig,
            RuntimeSessionConfig,
            LLMModelConfig,
            ModelParams,
        ]
        for model in models:
            for name, field in model.model_fields.items():
                assert field.description, f"{model.__name__}.{name} is missing a description"


class TestRuntimeDatabaseConfig:
    def test_allows_none_url(self):
        config = RuntimeDatabaseConfig()
        assert config.url is None

    def test_accepts_url(self):
        config = RuntimeDatabaseConfig(url="postgresql://localhost/db")
        assert config.url == "postgresql://localhost/db"

    def test_invalid_pool_min_zero(self):
        with pytest.raises(ValidationError):
            RuntimeDatabaseConfig(pool_min=0)

    def test_invalid_pool_max_zero(self):
        with pytest.raises(ValidationError):
            RuntimeDatabaseConfig(pool_max=0)


class TestLLMModelConfig:
    """Tests for LLMModelConfig."""

    def test_requires_name(self):
        with pytest.raises(ValidationError):
            LLMModelConfig()

    def test_defaults(self):
        config = LLMModelConfig(name="test-model")
        assert config.name == "test-model"
        assert config.api_base is None
        assert config.max_tokens == 4096
        assert isinstance(config.model_params, ModelParams)

    def test_custom_values(self):
        config = LLMModelConfig(
            name="ollama/qwen2.5:32b",
            api_base="http://localhost:11434",
            max_tokens=8192,
            model_params={"temperature": 0.5},
        )
        assert config.name == "ollama/qwen2.5:32b"
        assert config.api_base == "http://localhost:11434"
        assert config.max_tokens == 8192
        assert config.model_params.temperature == 0.5

    def test_invalid_max_tokens_zero(self):
        with pytest.raises(ValidationError):
            LLMModelConfig(name="test", max_tokens=0)


class TestRuntimeLLMConfig:
    def test_defaults(self):
        config = RuntimeLLMConfig()
        assert config.model.name == "claude-sonnet-4-20250514"
        assert config.model.api_base is None
        assert config.fast_model.name == "claude-haiku-4-5-20251001"
        assert config.fast_model.max_tokens == 1000
        assert config.embedding.name == "text-embedding-3-small"

    def test_custom_nested_values(self):
        config = RuntimeLLMConfig(
            model={
                "name": "ollama/qwen2.5:32b",
                "api_base": "http://localhost:11434",
                "max_tokens": 8192,
            },
        )
        assert config.model.name == "ollama/qwen2.5:32b"
        assert config.model.api_base == "http://localhost:11434"
        assert config.model.max_tokens == 8192

    def test_per_model_params(self):
        config = RuntimeLLMConfig(
            model={"name": "main", "model_params": {"temperature": 0.7}},
            fast_model={"name": "fast", "model_params": {"temperature": 0.3}},
        )
        assert config.model.model_params.temperature == 0.7
        assert config.fast_model.model_params.temperature == 0.3


class TestModelParams:
    """Tests for ModelParams sampling parameter model."""

    def test_defaults_all_none(self):
        params = ModelParams()
        assert params.temperature is None
        assert params.top_p is None
        assert params.top_k is None
        assert params.frequency_penalty is None
        assert params.presence_penalty is None
        assert params.stop is None

    def test_to_api_kwargs_empty_when_all_none(self):
        params = ModelParams()
        assert params.to_api_kwargs() == {}

    def test_to_api_kwargs_returns_only_non_none(self):
        params = ModelParams(temperature=0.5, top_p=0.9)
        kwargs = params.to_api_kwargs()
        assert kwargs == {"temperature": 0.5, "top_p": 0.9}
        assert "top_k" not in kwargs
        assert "frequency_penalty" not in kwargs

    def test_to_api_kwargs_with_stop_sequences(self):
        params = ModelParams(stop=["END", "STOP"])
        kwargs = params.to_api_kwargs()
        assert kwargs == {"stop": ["END", "STOP"]}

    def test_temperature_validation_range(self):
        ModelParams(temperature=0)
        ModelParams(temperature=2)
        with pytest.raises(ValidationError):
            ModelParams(temperature=-0.1)
        with pytest.raises(ValidationError):
            ModelParams(temperature=2.1)

    def test_top_p_validation_range(self):
        ModelParams(top_p=0)
        ModelParams(top_p=1)
        with pytest.raises(ValidationError):
            ModelParams(top_p=-0.1)
        with pytest.raises(ValidationError):
            ModelParams(top_p=1.1)

    def test_top_k_validation_min(self):
        ModelParams(top_k=1)
        with pytest.raises(ValidationError):
            ModelParams(top_k=0)

    def test_frequency_penalty_validation_range(self):
        ModelParams(frequency_penalty=-2)
        ModelParams(frequency_penalty=2)
        with pytest.raises(ValidationError):
            ModelParams(frequency_penalty=-2.1)
        with pytest.raises(ValidationError):
            ModelParams(frequency_penalty=2.1)

    def test_presence_penalty_validation_range(self):
        ModelParams(presence_penalty=-2)
        ModelParams(presence_penalty=2)
        with pytest.raises(ValidationError):
            ModelParams(presence_penalty=-2.1)
        with pytest.raises(ValidationError):
            ModelParams(presence_penalty=2.1)


class TestInterfaceConfig:
    def test_basic(self):
        config = InterfaceConfig(plugin="telegram")
        assert config.plugin == "telegram"
        assert config.session is None
        assert config.pipeline is None

    def test_with_session(self):
        config = InterfaceConfig(
            plugin="telegram",
            session={"name": "main_chat", "lifecycle": "persistent"},
        )
        assert config.session.name == "main_chat"
        assert config.session.lifecycle == "persistent"

    def test_allows_extra_fields(self):
        """Plugin-specific fields (interval_seconds, port, etc.) should pass through."""
        config = InterfaceConfig(
            plugin="timer",
            interval_seconds=300,
            enabled=True,
        )
        assert config.plugin == "timer"
        assert config.interval_seconds == 300
        assert config.enabled is True

    def test_http_with_port(self):
        config = InterfaceConfig(plugin="http", port=8008)
        assert config.port == 8008


class TestMCPServerConfig:
    def test_requires_name_and_url(self):
        with pytest.raises(ValidationError):
            MCPServerConfig()

    def test_requires_name(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(url="http://localhost:3001")

    def test_requires_url(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="test-server")

    def test_defaults(self):
        config = MCPServerConfig(name="fs", url="npx server-fs /tmp")
        assert config.transport == "stdio"
        assert config.tools is None
        assert config.env is None
        assert config.args is None

    def test_full_config(self):
        config = MCPServerConfig(
            name="search",
            url="http://localhost:3001/mcp",
            transport="http",
            tools=["web_search"],
            env={"API_KEY": "secret"},
            args=["--verbose"],
        )
        assert config.transport == "http"
        assert config.tools == ["web_search"]
        assert config.env == {"API_KEY": "secret"}


class TestAgentYAMLConfig:
    def test_minimal(self):
        """Minimal config with no fields should use all defaults."""
        config = AgentYAMLConfig()
        assert config.agent_name is None
        assert config.system_prompt == ""
        assert config.runtime.llm.model.name == "claude-sonnet-4-20250514"
        assert config.interfaces == {}
        assert config.sessions == {}
        assert config.mcp_servers == []

    def test_parses_minimal_dict(self):
        data = {
            "pipeline": {"token_budget": 8000},
            "runtime": {
                "llm": {"model": {"name": "test-model"}},
            },
        }
        config = AgentYAMLConfig(**data)
        assert config.runtime.llm.model.name == "test-model"
        assert config.pipeline["token_budget"] == 8000

    def test_parses_full_edi_config(self):
        """Parse the real EDI agent.yaml from the repo."""
        edi_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "configs",
            "agents",
            "edi",
            "agent.yaml",
        )
        if not os.path.exists(edi_path):
            pytest.skip("EDI config not found")

        with open(edi_path) as f:
            data = yaml.safe_load(f)

        config = AgentYAMLConfig(**data)
        assert config.agent_name == "EDI"
        assert config.system_prompt != ""
        assert config.runtime.llm.model.name is not None
        assert config.runtime.loop.max_iterations > 0
        assert isinstance(config.interfaces, dict)
        assert isinstance(config.sessions, dict)
        assert isinstance(config.pipeline, dict)
        assert config.pipeline.get("token_budget", 0) > 0

    def test_allows_pipeline_fields(self):
        """Pipeline fields (token_budget, compaction, etc.) should be stored under pipeline."""
        data = {
            "pipeline": {
                "token_budget": 48000,
                "compaction": {"enabled": True},
                "layers": [{"name": "core"}],
            },
        }
        config = AgentYAMLConfig(**data)
        assert config.pipeline["token_budget"] == 48000
        assert config.pipeline["compaction"] == {"enabled": True}

    def test_interfaces_with_sessions(self):
        data = {
            "interfaces": {
                "tg": {
                    "plugin": "telegram",
                    "session": {"name": "main", "lifecycle": "persistent"},
                    "pipeline": "interfaces/user_message.yaml",
                },
                "timer": {
                    "plugin": "timer",
                    "interval_seconds": 300,
                    "session": {"name": "hb", "lifecycle": "ephemeral"},
                },
            }
        }
        config = AgentYAMLConfig(**data)
        assert config.interfaces["tg"].session.name == "main"
        assert config.interfaces["timer"].interval_seconds == 300

    def test_mcp_servers(self):
        data = {
            "mcp_servers": [
                {"name": "fs", "url": "npx server-fs /tmp", "transport": "stdio"},
                {"name": "web", "url": "http://localhost:3001", "transport": "http"},
            ]
        }
        config = AgentYAMLConfig(**data)
        assert len(config.mcp_servers) == 2
        assert config.mcp_servers[0].name == "fs"
        assert config.mcp_servers[1].transport == "http"


class TestInterfaceSessionConfig:
    def test_requires_name(self):
        with pytest.raises(ValidationError):
            InterfaceSessionConfig()

    def test_default_lifecycle(self):
        config = InterfaceSessionConfig(name="main")
        assert config.lifecycle == "persistent"

    def test_invalid_lifecycle(self):
        with pytest.raises(ValidationError):
            InterfaceSessionConfig(name="main", lifecycle="invalid")
