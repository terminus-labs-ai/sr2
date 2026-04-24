import pytest
from pydantic import ValidationError

from sr2.config import (
    ContentItemConfig,
    LayerConfig,
    PipelineConfig,
)


class TestPipelineConfigDefaults:
    """Test 1: PipelineConfig() with zero args produces valid defaults."""

    def test_default_construction(self):
        config = PipelineConfig()

        assert config.token_budget == 32000
        assert config.pre_rot_threshold == 0.25
        assert config.extends is None
        assert config.kv_cache.strategy == "append_only"
        assert config.compaction.enabled is True
        assert config.summarization.enabled is False
        assert config.memory.extract is False
        assert config.retrieval.enabled is True
        assert config.retrieval.strategy == "keyword"
        assert config.intent_detection.enabled is True
        assert config.tool_masking.strategy == "allowed_list"
        assert config.degradation.circuit_breaker_threshold == 3
        assert config.layers == []


class TestPipelineConfigFromFixture:
    """Test 2: PipelineConfig(**sample_interface_config) parses correctly."""

    def test_parse_sample_config(self, sample_interface_config):
        config = PipelineConfig(**sample_interface_config)

        assert config.token_budget == 32000
        assert config.pre_rot_threshold == 0.25
        assert config.compaction.enabled is False
        assert config.summarization.enabled is False
        assert config.retrieval.enabled is False
        assert config.intent_detection.enabled is False
        assert len(config.layers) == 1
        assert config.layers[0].name == "core"
        assert config.layers[0].contents[0].key == "system_prompt"
        assert config.layers[0].contents[0].source == "config"


class TestTokenBudgetValidation:
    """Test 3: Invalid token_budget (< 1000) raises ValidationError."""

    def test_token_budget_too_low(self):
        with pytest.raises(ValidationError):
            PipelineConfig(token_budget=999)


class TestPreRotThresholdValidation:
    """Test 4: Invalid pre_rot_threshold (> 1.0) raises ValidationError."""

    def test_pre_rot_threshold_too_high(self):
        with pytest.raises(ValidationError):
            PipelineConfig(pre_rot_threshold=1.5)


class TestContentItemExtraFields:
    """Test 5: ContentItemConfig with extra fields (resolver-specific) is accepted."""

    def test_extra_fields_allowed(self):
        item = ContentItemConfig(
            key="knowledge_base",
            source="retrieval",
            custom_resolver_param="some_value",
            another_param=42,
        )

        assert item.key == "knowledge_base"
        assert item.source == "retrieval"
        assert item.custom_resolver_param == "some_value"
        assert item.another_param == 42


class TestLayerConfigEmptyContents:
    """Test 6: LayerConfig with empty contents list is valid."""

    def test_empty_contents(self):
        layer = LayerConfig(name="empty_layer", contents=[])

        assert layer.name == "empty_layer"
        assert layer.contents == []
        assert layer.cache_policy == "immutable"


class TestSerializationRoundTrip:
    """Test 7: Nested config serialization round-trip."""

    def test_round_trip(self, sample_interface_config):
        original = PipelineConfig(**sample_interface_config)
        dump = original.model_dump()
        restored = PipelineConfig(**dump)

        assert original == restored
