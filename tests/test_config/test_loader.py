import pytest
import yaml

from sr2.config.loader import ConfigLoader
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
