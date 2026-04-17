import pytest
from sr2.config.models import (
    CompactionConfig,
    ContentItemConfig,
    CostGateConfig,
    LayerConfig,
    PipelineConfig,
    RetrievalConfig,
    SummarizationConfig,
)
from sr2.config.validation import ConfigValidationError, validate_config


class TestValidateConfig:
    def test_valid_config_returns_empty_warnings(self):
        """A fully valid config with no issues returns an empty warnings list."""
        config = PipelineConfig(
            token_budget=32000,
            compaction=CompactionConfig(enabled=False, cost_gate=CostGateConfig(enabled=False)),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            layers=[
                LayerConfig(
                    name="core",
                    contents=[
                        ContentItemConfig(key="system", source="config", max_tokens=1000),
                    ],
                )
            ],
        )
        warnings = validate_config(config)
        assert warnings == []

    def test_token_budget_exceeded_raises_error(self):
        """When sum of content max_tokens exceeds token_budget, raise ConfigValidationError."""
        config = PipelineConfig(
            token_budget=2000,
            compaction=CompactionConfig(enabled=False, cost_gate=CostGateConfig(enabled=False)),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            layers=[
                LayerConfig(
                    name="core",
                    contents=[
                        ContentItemConfig(key="a", source="config", max_tokens=1500),
                        ContentItemConfig(key="b", source="config", max_tokens=1500),
                    ],
                )
            ],
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "exceeds token_budget" in str(exc_info.value)
        assert "3000" in str(exc_info.value)
        assert "2000" in str(exc_info.value)

    def test_always_new_before_append_only_raises_error(self):
        """Cache-killing layout: always_new layer before append_only layer raises error."""
        config = PipelineConfig(
            token_budget=32000,
            compaction=CompactionConfig(enabled=False, cost_gate=CostGateConfig(enabled=False)),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            layers=[
                LayerConfig(
                    name="dynamic",
                    cache_policy="always_new",
                    contents=[
                        ContentItemConfig(key="x", source="config", max_tokens=1000),
                    ],
                ),
                LayerConfig(
                    name="stable",
                    cache_policy="append_only",
                    contents=[
                        ContentItemConfig(key="y", source="config", max_tokens=1000),
                    ],
                ),
            ],
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "cache-killing layout" in str(exc_info.value)
        assert "dynamic" in str(exc_info.value)
        assert "stable" in str(exc_info.value)

    def test_no_layers_raises_error(self):
        """A config with no layers raises ConfigValidationError."""
        config = PipelineConfig(
            token_budget=32000,
            compaction=CompactionConfig(enabled=False, cost_gate=CostGateConfig(enabled=False)),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            layers=[],
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config(config)
        assert "No layers defined" in str(exc_info.value)

    def test_compaction_enabled_without_rules_returns_warning(self):
        """Compaction enabled but no rules defined produces a warning."""
        config = PipelineConfig(
            token_budget=32000,
            compaction=CompactionConfig(enabled=True, rules=[], cost_gate=CostGateConfig(enabled=False)),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
            layers=[
                LayerConfig(
                    name="core",
                    contents=[
                        ContentItemConfig(key="system", source="config", max_tokens=1000),
                    ],
                )
            ],
        )
        warnings = validate_config(config)
        assert any("Compaction is enabled but no compaction rules" in w for w in warnings)

    def test_summarization_without_compaction_returns_warning(self):
        """Summarization enabled but compaction disabled produces a warning."""
        config = PipelineConfig(
            token_budget=32000,
            compaction=CompactionConfig(enabled=False, cost_gate=CostGateConfig(enabled=False)),
            summarization=SummarizationConfig(enabled=True),
            retrieval=RetrievalConfig(enabled=False),
            layers=[
                LayerConfig(
                    name="core",
                    contents=[
                        ContentItemConfig(key="system", source="config", max_tokens=1000),
                    ],
                )
            ],
        )
        warnings = validate_config(config)
        assert any("Summarization is enabled but compaction is disabled" in w for w in warnings)
