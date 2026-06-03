"""Tests for FR2: per-layer degradation metadata (sr2-82).

Validates that LayerConfig carries degradation_category and priority,
and that both are plumbed through to the Layer runtime object.
"""

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# 1. LayerConfig — new fields
# ---------------------------------------------------------------------------


class TestLayerConfigDegradationFields:
    def test_degradation_category_defaults_to_none(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="system",
            target="system",
            resolvers=[ResolverConfig(type="static_template")],
        )
        assert layer.degradation_category is None

    def test_degradation_category_accepts_string(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="memory",
            target="messages",
            resolvers=[ResolverConfig(type="retrieval")],
            degradation_category="memory",
        )
        assert layer.degradation_category == "memory"

    def test_priority_defaults_to_zero(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="system",
            target="system",
            resolvers=[ResolverConfig(type="static_template")],
        )
        assert layer.priority == 0

    def test_priority_accepts_int(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="memory",
            target="messages",
            resolvers=[ResolverConfig(type="retrieval")],
            priority=5,
        )
        assert layer.priority == 5

    def test_priority_accepts_negative(self):
        """Negative priority means 'shed first' (lower = shed first)."""
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="tools",
            target="tools",
            resolvers=[ResolverConfig(type="mcp")],
            priority=-10,
        )
        assert layer.priority == -10

    def test_both_fields_together(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="memory",
            target="messages",
            resolvers=[ResolverConfig(type="retrieval")],
            degradation_category="memory",
            priority=3,
        )
        assert layer.degradation_category == "memory"
        assert layer.priority == 3

    def test_no_category_means_never_dropped(self):
        """A layer without degradation_category is structural and always kept."""
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="system",
            target="system",
            resolvers=[ResolverConfig(type="static_template")],
        )
        assert layer.degradation_category is None
        # No category → the layer is never considered for degradation

    def test_dict_construction_yaml_compat(self):
        """Config dicts representing YAML input should construct with new fields."""
        from sr2.config.models import PipelineConfig

        raw = {
            "layers": [
                {
                    "name": "system",
                    "target": "system",
                    "resolvers": [{"type": "static_template"}],
                    # No degradation fields — should default correctly
                },
                {
                    "name": "memory",
                    "target": "messages",
                    "resolvers": [{"type": "retrieval"}],
                    "degradation_category": "memory",
                    "priority": 3,
                },
            ]
        }
        cfg = PipelineConfig(**raw)

        sys_layer = cfg.layers[0]
        assert sys_layer.degradation_category is None
        assert sys_layer.priority == 0

        mem_layer = cfg.layers[1]
        assert mem_layer.degradation_category == "memory"
        assert mem_layer.priority == 3

    def test_model_dump_round_trip(self):
        """model_dump() output reconstructs with new fields preserved."""
        from sr2.config.models import LayerConfig, ResolverConfig

        layer1 = LayerConfig(
            name="memory",
            target="messages",
            resolvers=[ResolverConfig(type="retrieval")],
            degradation_category="memory",
            priority=5,
        )
        dumped = layer1.model_dump()
        layer2 = LayerConfig(**dumped)

        assert layer2.degradation_category == "memory"
        assert layer2.priority == 5

    def test_backward_compat_without_fields(self):
        """Existing configs without degradation fields still parse correctly."""
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="conversation",
            target="messages",
            token_budget=10000,
            position="append",
            resolvers=[ResolverConfig(type="session")],
        )
        assert layer.name == "conversation"
        assert layer.token_budget == 10000
        assert layer.position == "append"
        assert layer.degradation_category is None
        assert layer.priority == 0


# ---------------------------------------------------------------------------
# 2. Layer — fields plumbed from config
# ---------------------------------------------------------------------------


class TestLayerDegradationFields:
    def test_layer_has_degradation_category_attribute(self):
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.token_counting import CharacterTokenCounter

        layer = Layer(
            name="memory",
            target="messages",
            position="append",
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            degradation_category="memory",
        )
        assert layer.degradation_category == "memory"

    def test_layer_has_priority_attribute(self):
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.token_counting import CharacterTokenCounter

        layer = Layer(
            name="memory",
            target="messages",
            position="append",
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            priority=5,
        )
        assert layer.priority == 5

    def test_layer_defaults_degradation_category_to_none(self):
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.token_counting import CharacterTokenCounter

        layer = Layer(
            name="system",
            target="system",
            position="append",
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        assert layer.degradation_category is None

    def test_layer_defaults_priority_to_zero(self):
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.token_counting import CharacterTokenCounter

        layer = Layer(
            name="system",
            target="system",
            position="append",
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        assert layer.priority == 0

    def test_layer_both_fields(self):
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.token_counting import CharacterTokenCounter

        layer = Layer(
            name="memory",
            target="messages",
            position="append",
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            degradation_category="memory",
            priority=3,
        )
        assert layer.degradation_category == "memory"
        assert layer.priority == 3

    def test_layer_backward_compat(self):
        """Layer construction without new fields still works (backward compat)."""
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.token_counting import CharacterTokenCounter

        layer = Layer(
            name="system",
            target="system",
            position="append",
            token_budget=4000,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        # Existing fields still work
        assert layer.name == "system"
        assert layer.token_budget == 4000
        # New fields have sensible defaults
        assert layer.degradation_category is None
        assert layer.priority == 0
