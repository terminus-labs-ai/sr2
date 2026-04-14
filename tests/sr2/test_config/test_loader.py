import os

import pytest
import yaml

from sr2.config.loader import ConfigLoader, _expand_env
from sr2.config.models import PipelineConfig


class TestLoadFromDict:
    """Test 1: Load a simple config dict -> returns valid PipelineConfig."""

    def test_load_from_dict_returns_pipeline_config(self):
        loader = ConfigLoader()
        config = loader.load_from_dict({"token_budget": 16000})

        assert isinstance(config, PipelineConfig)
        assert config.token_budget == 16000
        # Defaults should be filled in
        assert config.pre_rot_threshold == 0.25
        assert config.compaction.enabled is True


class TestMergeScalarOverride:
    """Test 2: Deep merge: scalar override works."""

    def test_scalar_override(self):
        loader = ConfigLoader()
        base = {"token_budget": 32000, "pre_rot_threshold": 0.25}
        override = {"token_budget": 16000}

        merged = loader.merge(base, override)

        assert merged["token_budget"] == 16000
        assert merged["pre_rot_threshold"] == 0.25


class TestMergeNestedDict:
    """Test 3: Deep merge: nested dict override merges correctly."""

    def test_nested_dict_merge(self):
        loader = ConfigLoader()
        base = {
            "compaction": {"enabled": True, "raw_window": 5, "min_content_size": 100},
        }
        override = {
            "compaction": {"raw_window": 10},
        }

        merged = loader.merge(base, override)

        assert merged["compaction"]["enabled"] is True
        assert merged["compaction"]["raw_window"] == 10
        assert merged["compaction"]["min_content_size"] == 100


class TestMergeListReplacement:
    """Test 4: Deep merge: list replacement works."""

    def test_list_replacement(self):
        loader = ConfigLoader()
        base = {"layers": [{"name": "core"}, {"name": "session"}]}
        override = {"layers": [{"name": "override_only"}]}

        merged = loader.merge(base, override)

        assert merged["layers"] == [{"name": "override_only"}]


class TestMergeNoneIgnored:
    """Test 5: Deep merge: None values in override don't erase base."""

    def test_none_values_ignored(self):
        loader = ConfigLoader()
        base = {"token_budget": 32000, "pre_rot_threshold": 0.25}
        override = {"token_budget": None, "pre_rot_threshold": 0.5}

        merged = loader.merge(base, override)

        assert merged["token_budget"] == 32000
        assert merged["pre_rot_threshold"] == 0.5


class TestTwoLevelInheritance:
    """Test 6: Two-level inheritance: defaults -> override -> final."""

    def test_two_level_inheritance(self, tmp_path):
        # Write base.yaml
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"pipeline": {"token_budget": 32000, "pre_rot_threshold": 0.25}}))

        # Write mid.yaml that extends base
        mid = tmp_path / "mid.yaml"
        mid.write_text(yaml.dump({"extends": "base.yaml", "pipeline": {"token_budget": 24000}}))

        # Write final.yaml that extends mid
        final = tmp_path / "final.yaml"
        final.write_text(yaml.dump({"extends": "mid.yaml", "pipeline": {"token_budget": 16000}}))

        loader = ConfigLoader()
        config = loader.load(str(final))

        assert config.token_budget == 16000
        assert config.pre_rot_threshold == 0.25


class TestMissingExtendsTarget:
    """Test 7: Missing extends target raises FileNotFoundError."""

    def test_missing_extends_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"extends": "nonexistent.yaml"}))

        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(str(config_file))

    def test_missing_defaults_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"extends": "defaults"}))

        loader = ConfigLoader()  # No defaults_path
        with pytest.raises(FileNotFoundError):
            loader.load(str(config_file))


class TestCircularExtends:
    """Test 8: Circular extends raises ValueError."""

    def test_circular_inheritance(self, tmp_path):
        # a.yaml extends b.yaml, b.yaml extends a.yaml
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(yaml.dump({"extends": "b.yaml", "token_budget": 16000}))
        b.write_text(yaml.dump({"extends": "a.yaml", "token_budget": 32000}))

        loader = ConfigLoader()
        with pytest.raises(ValueError, match="Circular config inheritance detected"):
            loader.load(str(a))


class TestRootLevelPipelineFields:
    """Test 9: Root-level PipelineConfig fields are extracted (not just from pipeline: dict)."""

    def test_system_prompt_at_root_level(self, tmp_path):
        """system_prompt defined at YAML root level should be extracted."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"system_prompt": "You are running autonomously..."}))

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.system_prompt == "You are running autonomously..."

    def test_pipeline_dict_overrides_root_level(self, tmp_path):
        """pipeline: dict values should override root-level values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "system_prompt": "Root prompt",
                    "pipeline": {"system_prompt": "Nested prompt"},
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.system_prompt == "Nested prompt"

    def test_mixed_root_and_pipeline_fields(self, tmp_path):
        """Root-level and pipeline: dict fields should merge correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "system_prompt": "Root prompt",
                    "pipeline": {"token_budget": 65536},
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.system_prompt == "Root prompt"
        assert config.token_budget == 65536

    def test_inheritance_with_root_level_fields(self, tmp_path):
        """Root-level fields should work with inheritance chain."""
        # Base config with system_prompt at root level
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"system_prompt": "Base prompt"}))

        # Child config that extends and adds pipeline: dict
        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({"extends": "base.yaml", "pipeline": {"token_budget": 16000}}))

        loader = ConfigLoader()
        config = loader.load(str(child))

        assert config.system_prompt == "Base prompt"
        assert config.token_budget == 16000


class TestRuntimeLLMExtraction:
    """Test runtime.llm extraction into PipelineConfig.llm."""

    def test_runtime_llm_extracted_as_pipeline_llm(self, tmp_path):
        """runtime.llm.model should populate PipelineConfig.llm.model."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "runtime": {
                        "llm": {
                            "model": {
                                "name": "openai/qwen3.5:40b-a3b",
                                "api_base": "http://localhost:8080/v1",
                            }
                        }
                    }
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.llm is not None
        assert config.llm.model is not None
        assert config.llm.model.name == "openai/qwen3.5:40b-a3b"
        assert config.llm.model.api_base == "http://localhost:8080/v1"

    def test_stream_field_stripped(self, tmp_path):
        """runtime.llm fields not in LLMModelOverride (like stream) should be stripped."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "runtime": {
                        "llm": {
                            "model": {
                                "name": "openai/qwen3.5:40b-a3b",
                                "stream": True,
                                "timeout": 30,
                            }
                        }
                    }
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.llm is not None
        assert config.llm.model.name == "openai/qwen3.5:40b-a3b"
        # stream and timeout are not LLMModelOverride fields, should not be present
        model_dict = config.llm.model.model_dump()
        assert "stream" not in model_dict
        assert "timeout" not in model_dict

    def test_pipeline_llm_takes_priority_over_runtime(self, tmp_path):
        """pipeline.llm should completely override runtime.llm."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "runtime": {
                        "llm": {
                            "model": {"name": "openai/runtime-model"},
                        }
                    },
                    "pipeline": {
                        "llm": {
                            "model": {"name": "openai/pipeline-model"},
                        }
                    },
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.llm.model.name == "openai/pipeline-model"

    def test_root_level_llm_takes_priority_over_runtime(self, tmp_path):
        """Root-level llm: should override runtime.llm."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "runtime": {
                        "llm": {
                            "model": {"name": "openai/runtime-model", "max_tokens": 4096},
                        }
                    },
                    "llm": {
                        "model": {"name": "openai/root-model"},
                    },
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        # Root-level wins for name
        assert config.llm.model.name == "openai/root-model"
        # But runtime max_tokens should merge through as fallback
        assert config.llm.model.max_tokens == 4096

    def test_inheritance_with_runtime_llm_in_child(self, tmp_path):
        """Child config with runtime.llm should override parent's model."""
        # Parent (agent.yaml style)
        parent = tmp_path / "agent.yaml"
        parent.write_text(
            yaml.dump(
                {
                    "runtime": {
                        "llm": {
                            "model": {"name": "openai/base-model"},
                        }
                    },
                    "pipeline": {"token_budget": 65536},
                }
            )
        )

        # Child interface
        interfaces = tmp_path / "interfaces"
        interfaces.mkdir()
        child = interfaces / "user_message.yaml"
        child.write_text(
            yaml.dump(
                {
                    "extends": "agent",
                    "runtime": {
                        "llm": {
                            "model": {"name": "openai/override-model"},
                        }
                    },
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(child))

        assert config.llm.model.name == "openai/override-model"
        assert config.token_budget == 65536

    def test_fast_model_and_embedding_extracted(self, tmp_path):
        """runtime.llm.fast_model and embedding should also be extracted."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "runtime": {
                        "llm": {
                            "model": {"name": "openai/main-model"},
                            "fast_model": {"name": "openai/fast-model"},
                            "embedding": {"name": "openai/embed-model"},
                        }
                    }
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.llm.model.name == "openai/main-model"
        assert config.llm.fast_model.name == "openai/fast-model"
        assert config.llm.embedding.name == "openai/embed-model"

    def test_no_runtime_llm_no_effect(self, tmp_path):
        """Config without runtime.llm should work as before (backward compat)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"pipeline": {"token_budget": 16000}}))

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.llm.model is None
        assert config.token_budget == 16000

    def test_runtime_without_llm_key(self, tmp_path):
        """runtime: dict without llm key should not affect anything."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"runtime": {"stream": True}, "pipeline": {"token_budget": 16000}})
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))

        assert config.llm.model is None


class TestPathTraversalPrevention:
    """Test: extends paths cannot escape the config directory."""

    def test_parent_directory_traversal_blocked(self, tmp_path):
        """extends: '../../../etc/passwd' should raise ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"extends": "../../../etc/passwd"}))

        loader = ConfigLoader()
        with pytest.raises(ValueError, match="extends path escapes config directory"):
            loader.load(str(config_file))

    def test_absolute_path_blocked(self, tmp_path):
        """extends with an absolute-looking relative path that escapes should be blocked."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"extends": "../../outside.yaml"}))

        loader = ConfigLoader()
        with pytest.raises(ValueError, match="extends path escapes config directory"):
            loader.load(str(config_file))

    def test_sibling_file_allowed(self, tmp_path):
        """extends: 'sibling.yaml' in the same directory should work."""
        sibling = tmp_path / "sibling.yaml"
        sibling.write_text(yaml.dump({"pipeline": {"token_budget": 32000}}))

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"extends": "sibling.yaml", "pipeline": {"token_budget": 16000}})
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.token_budget == 16000

    def test_subdirectory_allowed(self, tmp_path):
        """extends: 'subdir/child.yaml' within the config directory should work."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        child = subdir / "child.yaml"
        child.write_text(yaml.dump({"pipeline": {"token_budget": 32000}}))

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"extends": "subdir/child.yaml", "pipeline": {"token_budget": 16000}})
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.token_budget == 16000

    def test_dot_dot_in_middle_that_stays_within_dir(self, tmp_path):
        """extends: 'subdir/../sibling.yaml' resolves within dir — should work."""
        sibling = tmp_path / "sibling.yaml"
        sibling.write_text(yaml.dump({"pipeline": {"token_budget": 32000}}))
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"extends": "subdir/../sibling.yaml", "pipeline": {"token_budget": 16000}})
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.token_budget == 16000


class TestExpandEnvFunction:
    """Unit tests for _expand_env string expansion."""

    def test_simple_var(self, monkeypatch):
        monkeypatch.setenv("SR2_TEST_VAR", "hello")
        assert _expand_env("${SR2_TEST_VAR}") == "hello"

    def test_var_with_surrounding_text(self, monkeypatch):
        monkeypatch.setenv("SR2_TEST_AGENT", "nightshift")
        assert _expand_env("claude-${SR2_TEST_AGENT}") == "claude-nightshift"

    def test_default_value_when_unset(self, monkeypatch):
        monkeypatch.delenv("SR2_UNSET_VAR", raising=False)
        assert _expand_env("${SR2_UNSET_VAR:-fallback}") == "fallback"

    def test_default_ignored_when_var_set(self, monkeypatch):
        monkeypatch.setenv("SR2_SET_VAR", "real")
        assert _expand_env("${SR2_SET_VAR:-fallback}") == "real"

    def test_empty_default(self, monkeypatch):
        monkeypatch.delenv("SR2_UNSET_VAR", raising=False)
        assert _expand_env("${SR2_UNSET_VAR:-}") == ""

    def test_unresolved_var_left_as_is(self, monkeypatch):
        monkeypatch.delenv("SR2_MISSING", raising=False)
        assert _expand_env("${SR2_MISSING}") == "${SR2_MISSING}"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("SR2_A", "one")
        monkeypatch.setenv("SR2_B", "two")
        assert _expand_env("${SR2_A}-${SR2_B}") == "one-two"

    def test_no_vars_passthrough(self):
        assert _expand_env("plain string") == "plain string"


class TestEnvVarInterpolation:
    """Integration tests for env var expansion in ConfigLoader."""

    def test_env_vars_expanded_in_loaded_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDE_AGENT_NAME", "nightshift")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "pipeline": {
                        "memory": {
                            "scope": {
                                "agent_name": "claude-${CLAUDE_AGENT_NAME}",
                            }
                        }
                    }
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.memory.scope.agent_name == "claude-nightshift"

    def test_env_vars_in_lists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SR2_SCOPE", "project")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "pipeline": {
                        "memory": {
                            "scope": {
                                "allowed_read": ["${SR2_SCOPE}"],
                            }
                        }
                    }
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.memory.scope.allowed_read == ["project"]

    def test_default_value_in_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLAUDE_AGENT_NAME", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump({"system_prompt": "Agent: ${CLAUDE_AGENT_NAME:-default-agent}"})
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.system_prompt == "Agent: default-agent"

    def test_env_vars_expanded_in_parent_config(self, tmp_path, monkeypatch):
        """Env vars in parent configs are expanded before merge."""
        monkeypatch.setenv("SR2_BASE_BUDGET", "32000")
        parent = tmp_path / "base.yaml"
        parent.write_text(yaml.dump({"system_prompt": "budget=${SR2_BASE_BUDGET}"}))

        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({"extends": "base.yaml", "pipeline": {"token_budget": 16000}}))

        loader = ConfigLoader()
        config = loader.load(str(child))
        assert config.system_prompt == "budget=32000"
        assert config.token_budget == 16000

    def test_non_string_values_untouched(self, tmp_path):
        """Integers, booleans, etc. pass through without modification."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "pipeline": {
                        "token_budget": 65536,
                        "compaction": {"enabled": True, "raw_window": 10},
                    }
                }
            )
        )

        loader = ConfigLoader()
        config = loader.load(str(config_file))
        assert config.token_budget == 65536
        assert config.compaction.enabled is True
        assert config.compaction.raw_window == 10
