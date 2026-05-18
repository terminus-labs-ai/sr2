"""Tests for SR2 config models — Step 2 (config model updates).

Validates the updated Pydantic models match the spec:
- EventSubscriptionConfig (new)
- ResolverConfig gains subscriptions + max_executions
- TransformerConfig replaces triggers with subscriptions, gains max_executions
- LayerConfig gains target + position
- PipelineConfig has layers directly (no ContextConfig wrapper)
- TransformTriggers enum removed
"""

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# 1. EventSubscriptionConfig
# ---------------------------------------------------------------------------


class TestEventSubscriptionConfig:
    def test_create_with_event_only(self):
        from sr2.config.models import EventSubscriptionConfig

        sub = EventSubscriptionConfig(event="turn_start")
        assert sub.event == "turn_start"

    def test_phase_defaults_to_none(self):
        from sr2.config.models import EventSubscriptionConfig

        sub = EventSubscriptionConfig(event="overflow")
        assert sub.phase is None

    def test_explicit_phase(self):
        from sr2.config.models import EventSubscriptionConfig

        sub = EventSubscriptionConfig(event="turn_end", phase="post_resolve")
        assert sub.event == "turn_end"
        assert sub.phase == "post_resolve"

    def test_event_is_required(self):
        from sr2.config.models import EventSubscriptionConfig

        with pytest.raises(ValidationError):
            EventSubscriptionConfig()


# ---------------------------------------------------------------------------
# 2. ResolverConfig
# ---------------------------------------------------------------------------


class TestResolverConfig:
    def test_type_and_config_preserved(self):
        from sr2.config.models import ResolverConfig

        r = ResolverConfig(type="static_template")
        assert r.type == "static_template"
        assert r.config == {}

    def test_config_accepts_dict(self):
        from sr2.config.models import ResolverConfig

        r = ResolverConfig(type="retrieval", config={"top_k": 5})
        assert r.config == {"top_k": 5}

    def test_subscriptions_defaults_to_empty_list(self):
        from sr2.config.models import ResolverConfig

        r = ResolverConfig(type="input")
        assert r.subscriptions == []

    def test_subscriptions_accepts_event_subscriptions(self):
        from sr2.config.models import EventSubscriptionConfig, ResolverConfig

        subs = [
            EventSubscriptionConfig(event="turn_start"),
            EventSubscriptionConfig(event="overflow", phase="pre_resolve"),
        ]
        r = ResolverConfig(type="retrieval", subscriptions=subs)
        assert len(r.subscriptions) == 2
        assert r.subscriptions[0].event == "turn_start"
        assert r.subscriptions[1].phase == "pre_resolve"

    def test_max_executions_defaults_to_one(self):
        from sr2.config.models import ResolverConfig

        r = ResolverConfig(type="input")
        assert r.max_executions == 1

    def test_max_executions_explicit(self):
        from sr2.config.models import ResolverConfig

        r = ResolverConfig(type="input", max_executions=3)
        assert r.max_executions == 3


# ---------------------------------------------------------------------------
# 3. TransformerConfig
# ---------------------------------------------------------------------------


class TestTransformerConfig:
    def test_type_and_config_preserved(self):
        from sr2.config.models import TransformerConfig

        t = TransformerConfig(type="compaction")
        assert t.type == "compaction"
        assert t.config == {}

    def test_subscriptions_replaces_triggers(self):
        """TransformerConfig uses subscriptions, not triggers."""
        from sr2.config.models import EventSubscriptionConfig, TransformerConfig

        subs = [EventSubscriptionConfig(event="turn_end")]
        t = TransformerConfig(type="compaction", subscriptions=subs)
        assert len(t.subscriptions) == 1
        assert t.subscriptions[0].event == "turn_end"

    def test_subscriptions_defaults_to_empty_list(self):
        from sr2.config.models import TransformerConfig

        t = TransformerConfig(type="compaction")
        assert t.subscriptions == []

    def test_no_triggers_field(self):
        """The old triggers field must not exist."""
        from sr2.config.models import TransformerConfig

        t = TransformerConfig(type="compaction")
        assert not hasattr(t, "triggers")

    def test_max_executions_defaults_to_one(self):
        from sr2.config.models import TransformerConfig

        t = TransformerConfig(type="compaction")
        assert t.max_executions == 1

    def test_max_executions_explicit(self):
        from sr2.config.models import TransformerConfig

        t = TransformerConfig(type="summarization", max_executions=5)
        assert t.max_executions == 5


# ---------------------------------------------------------------------------
# 4. LayerConfig
# ---------------------------------------------------------------------------


class TestLayerConfig:
    def test_existing_fields_preserved(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="system",
            cache="static",
            token_budget=4000,
            resolvers=[ResolverConfig(type="static_template")],
        )
        assert layer.name == "system"
        assert layer.cache == "static"
        assert layer.token_budget == 4000
        assert len(layer.resolvers) == 1

    def test_transformers_still_optional(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(name="core", resolvers=[ResolverConfig(type="input")])
        assert layer.transformers is None

    def test_target_defaults_to_none(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(name="system", resolvers=[ResolverConfig(type="input")])
        assert layer.target is None

    def test_target_explicit(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="system",
            target="system_prompt",
            resolvers=[ResolverConfig(type="input")],
        )
        assert layer.target == "system_prompt"

    def test_position_defaults_to_append(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(name="system", resolvers=[ResolverConfig(type="input")])
        assert layer.position == "append"

    def test_position_explicit(self):
        from sr2.config.models import LayerConfig, ResolverConfig

        layer = LayerConfig(
            name="system",
            position="prepend",
            resolvers=[ResolverConfig(type="input")],
        )
        assert layer.position == "prepend"

    def test_cache_values(self):
        """Cache still accepts the three literal values and None."""
        from sr2.config.models import LayerConfig, ResolverConfig

        for cache_val in ("static", "ephemeral", "append_only", None):
            layer = LayerConfig(
                name="test",
                cache=cache_val,
                resolvers=[ResolverConfig(type="input")],
            )
            assert layer.cache == cache_val


# ---------------------------------------------------------------------------
# 5. PipelineConfig — layers directly, no ContextConfig wrapper
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_layers_directly_on_pipeline_config(self):
        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig

        cfg = PipelineConfig(
            layers=[
                LayerConfig(name="system", resolvers=[ResolverConfig(type="input")]),
            ]
        )
        assert len(cfg.layers) == 1
        assert cfg.layers[0].name == "system"

    def test_no_context_wrapper(self):
        """PipelineConfig should NOT have a 'context' field."""
        from sr2.config.models import PipelineConfig

        fields = PipelineConfig.model_fields
        assert "context" not in fields
        assert "layers" in fields

    def test_multiple_layers(self):
        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig

        cfg = PipelineConfig(
            layers=[
                LayerConfig(name="system", resolvers=[ResolverConfig(type="static_template")]),
                LayerConfig(name="memory", resolvers=[ResolverConfig(type="retrieval")]),
                LayerConfig(name="conversation", resolvers=[ResolverConfig(type="session")]),
            ]
        )
        assert len(cfg.layers) == 3
        names = [layer.name for layer in cfg.layers]
        assert names == ["system", "memory", "conversation"]


# ---------------------------------------------------------------------------
# 6. TransformTriggers enum removed
# ---------------------------------------------------------------------------


class TestTransformTriggersRemoved:
    def test_enum_not_importable(self):
        """TransformTriggers should no longer exist in sr2.config.models."""
        import sr2.config.models as models_mod

        assert not hasattr(models_mod, "TransformTriggers")

    def test_context_config_not_importable(self):
        """ContextConfig should no longer exist (replaced by direct layers)."""
        import sr2.config.models as models_mod

        assert not hasattr(models_mod, "ContextConfig")


# ---------------------------------------------------------------------------
# 7. Round-trip YAML compatibility
# ---------------------------------------------------------------------------


class TestYAMLRoundTrip:
    """Config dicts that represent typical YAML input should construct valid models."""

    def test_minimal_config_from_dict(self):
        from sr2.config.models import PipelineConfig

        raw = {
            "layers": [
                {
                    "name": "system",
                    "resolvers": [{"type": "static_template"}],
                }
            ]
        }
        cfg = PipelineConfig(**raw)
        assert cfg.layers[0].name == "system"
        assert cfg.layers[0].resolvers[0].type == "static_template"

    def test_full_config_from_dict(self):
        from sr2.config.models import PipelineConfig

        raw = {
            "layers": [
                {
                    "name": "system",
                    "cache": "static",
                    "token_budget": 4000,
                    "target": "system_prompt",
                    "position": "prepend",
                    "resolvers": [
                        {
                            "type": "static_template",
                            "config": {"template": "You are an assistant."},
                            "subscriptions": [{"event": "turn_start"}],
                            "max_executions": 1,
                        },
                    ],
                    "transformers": [
                        {
                            "type": "compaction",
                            "config": {"strategy": "rule_based"},
                            "subscriptions": [
                                {"event": "turn_end"},
                                {"event": "overflow", "phase": "post_resolve"},
                            ],
                            "max_executions": 3,
                        },
                    ],
                },
                {
                    "name": "conversation",
                    "cache": "append_only",
                    "resolvers": [
                        {"type": "session"},
                        {"type": "input"},
                    ],
                },
            ]
        }
        cfg = PipelineConfig(**raw)

        # Layer 0 — system
        sys_layer = cfg.layers[0]
        assert sys_layer.name == "system"
        assert sys_layer.cache == "static"
        assert sys_layer.token_budget == 4000
        assert sys_layer.target == "system_prompt"
        assert sys_layer.position == "prepend"

        resolver = sys_layer.resolvers[0]
        assert resolver.type == "static_template"
        assert resolver.config == {"template": "You are an assistant."}
        assert len(resolver.subscriptions) == 1
        assert resolver.subscriptions[0].event == "turn_start"
        assert resolver.max_executions == 1

        transformer = sys_layer.transformers[0]
        assert transformer.type == "compaction"
        assert transformer.config == {"strategy": "rule_based"}
        assert len(transformer.subscriptions) == 2
        assert transformer.subscriptions[0].event == "turn_end"
        assert transformer.subscriptions[0].phase is None
        assert transformer.subscriptions[1].event == "overflow"
        assert transformer.subscriptions[1].phase == "post_resolve"
        assert transformer.max_executions == 3

        # Layer 1 — conversation (minimal, defaults)
        conv_layer = cfg.layers[1]
        assert conv_layer.name == "conversation"
        assert conv_layer.target is None
        assert conv_layer.position == "append"
        assert len(conv_layer.resolvers) == 2
        assert conv_layer.transformers is None

    def test_model_serialization_round_trip(self):
        """model_dump() output can reconstruct the same model."""
        from sr2.config.models import PipelineConfig

        raw = {
            "layers": [
                {
                    "name": "memory",
                    "resolvers": [
                        {
                            "type": "retrieval",
                            "subscriptions": [{"event": "turn_start", "phase": "resolve"}],
                            "max_executions": 2,
                        }
                    ],
                    "target": "memory_context",
                    "position": "append",
                }
            ]
        }
        cfg1 = PipelineConfig(**raw)
        dumped = cfg1.model_dump()
        cfg2 = PipelineConfig(**dumped)

        assert cfg1 == cfg2
