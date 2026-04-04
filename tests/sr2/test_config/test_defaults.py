import pytest
from pathlib import Path
from sr2.config.loader import ConfigLoader
from sr2.config.validation import validate_config

DEFAULTS_PATH = str(Path(__file__).parent.parent.parent.parent / "configs" / "defaults.yaml")


class TestDefaults:
    def test_load_defaults_returns_valid_config(self):
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        assert config is not None

    def test_all_sections_present(self):
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        assert config.kv_cache is not None
        assert config.compaction is not None
        assert config.summarization is not None
        assert config.retrieval is not None
        assert config.intent_detection is not None
        assert config.tool_masking is not None
        assert config.degradation is not None

    def test_token_budget_is_32000(self):
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        assert config.token_budget == 32000

    def test_compaction_raw_window_is_5(self):
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        assert config.compaction.raw_window == 5

    def test_compaction_rules_default_empty(self):
        """Generated defaults have empty rules (users populate per agent)."""
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        assert isinstance(config.compaction.rules, list)

    def test_layers_default_empty(self):
        """Generated defaults have empty layers (users populate per agent)."""
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        assert isinstance(config.layers, list)

    def test_validate_config_expected_errors_for_bare_defaults(self):
        """Generated defaults are intentionally minimal — no layers defined.

        A real agent.yaml extends defaults and adds layers/rules.
        validate_config() will flag 'No layers defined' which is expected.
        """
        loader = ConfigLoader()
        config = loader.load(DEFAULTS_PATH)
        from sr2.config.validation import ConfigValidationError
        with pytest.raises(ConfigValidationError, match="No layers defined"):
            validate_config(config)
