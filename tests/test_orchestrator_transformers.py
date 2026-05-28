"""Tests for transformer execution infrastructure in orchestrator._build_layer().

Acceptance Criteria covered:
  AC8: A config with transformers: [{type: "unknown_type"}] raises PluginNotFoundError
       (NOT ConfigError, NOT ValueError) containing the unknown type name in the message.
  AC9: A config with transformers: [] or no transformers key builds the layer without
       error, producing a Layer with no transformers.
  AC10: If a type is registered in _TRANSFORMER_FACTORIES, building a config with that
        type produces a layer whose .transformers list contains the instance returned by
        the factory.
  AC11: All existing tests continue to pass (no regressions from removing the
        ConfigError block).
"""

from __future__ import annotations

import pytest

from sr2.config.models import (
    ConfigError,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.plugins.errors import PluginNotFoundError
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from sr2.models import TextBlock, TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLM:
    """Minimal LLMCallable for constructing SR2 without real LLM calls."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            id="mock-resp",
            content=[TextBlock(text="ok")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest):
        yield StreamEvent(type="text", text="ok")
        yield StreamEvent(type="end")


def _make_system_resolver() -> ResolverConfig:
    return ResolverConfig(
        type="static",
        config={"text": "You are a helpful assistant."},
    )


def _make_minimal_config(layer_transformers: list[TransformerConfig] | None = None) -> PipelineConfig:
    """One-layer PipelineConfig with optional transformers on the system layer."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[_make_system_resolver()],
                transformers=layer_transformers,
            )
        ]
    )


class _MinimalTransformerBase:
    """Minimal protocol-conforming transformer base for factory tests.

    Layer.subscriptions and is_done() require subscriptions, execution_count,
    and max_executions. Factories must return objects with these attributes.
    """

    subscriptions: list = []
    execution_count: int = 0
    max_executions: int = 1


def _make_sr2(config: PipelineConfig):
    """Construct SR2 with a mock LLM and CharacterTokenCounter."""
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config,
        llm={"default": _MockLLM()},
        token_counter=CharacterTokenCounter(),
    )


# ---------------------------------------------------------------------------
# AC8: Unknown transformer type raises PluginNotFoundError (not ConfigError)
# ---------------------------------------------------------------------------


class TestAC8UnknownTransformerType:
    """AC8: transformers: [{type: "unknown_type"}] raises PluginNotFoundError, not ConfigError."""

    def test_unknown_type_raises_plugin_not_found_error(self):
        """Constructing SR2 with an unknown transformer type raises PluginNotFoundError."""
        config = _make_minimal_config(
            layer_transformers=[TransformerConfig(type="unknown_type")]
        )

        with pytest.raises(PluginNotFoundError):
            _make_sr2(config)

    def test_unknown_type_error_message_contains_type_name(self):
        """The PluginNotFoundError message includes the unknown type name."""
        config = _make_minimal_config(
            layer_transformers=[TransformerConfig(type="no_such_transformer")]
        )

        with pytest.raises(PluginNotFoundError, match="no_such_transformer"):
            _make_sr2(config)

    def test_unknown_type_is_not_config_error(self):
        """The raised exception is PluginNotFoundError, not ConfigError (old behaviour)."""
        config = _make_minimal_config(
            layer_transformers=[TransformerConfig(type="does_not_exist")]
        )

        # Must NOT be a ConfigError
        with pytest.raises(PluginNotFoundError):
            _make_sr2(config)

        # Verify ConfigError is NOT raised
        try:
            _make_sr2(config)
        except PluginNotFoundError:
            pass  # expected
        except ConfigError:
            pytest.fail("ConfigError was raised; expected PluginNotFoundError")

    def test_second_unknown_type_in_list_also_raises(self):
        """Even the second transformer in a list raises PluginNotFoundError if its type is unknown."""
        # "totally_unknown" is not a registered entry point, so the registry raises
        # PluginNotFoundError when it is encountered, regardless of position in the list.
        config = _make_minimal_config(
            layer_transformers=[
                TransformerConfig(type="summarize"),       # known real entry point
                TransformerConfig(type="totally_unknown"),
            ]
        )
        with pytest.raises(PluginNotFoundError, match="totally_unknown"):
            _make_sr2(config)


# ---------------------------------------------------------------------------
# AC9: Empty or absent transformers builds layer without error, no transformers
# ---------------------------------------------------------------------------


class TestAC9EmptyOrAbsentTransformers:
    """AC9: transformers: [] or no transformers key builds successfully with empty list."""

    def test_no_transformers_key_builds_without_error(self):
        """LayerConfig with no transformers field builds SR2 without error."""
        config = _make_minimal_config(layer_transformers=None)

        sr2 = _make_sr2(config)
        assert sr2 is not None

    def test_empty_transformers_list_builds_without_error(self):
        """LayerConfig with transformers=[] builds SR2 without error."""
        config = _make_minimal_config(layer_transformers=[])

        sr2 = _make_sr2(config)
        assert sr2 is not None

    def test_layer_has_no_transformers_when_key_absent(self):
        """The built Layer has an empty transformers list when config omits the key."""
        config = _make_minimal_config(layer_transformers=None)

        sr2 = _make_sr2(config)
        layer = sr2._engine.layers[0]

        assert layer.transformers == []

    def test_layer_has_no_transformers_when_empty_list(self):
        """The built Layer has an empty transformers list when config passes []."""
        config = _make_minimal_config(layer_transformers=[])

        sr2 = _make_sr2(config)
        layer = sr2._engine.layers[0]

        assert layer.transformers == []

    def test_multi_layer_config_with_no_transformers(self):
        """Multi-layer config with no transformers on any layer builds without error."""
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[_make_system_resolver()],
                    transformers=None,
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[
                        ResolverConfig(type="session"),
                        ResolverConfig(type="input"),
                    ],
                    transformers=[],
                ),
            ]
        )

        sr2 = _make_sr2(config)
        assert sr2 is not None

        for layer in sr2._engine.layers:
            assert layer.transformers == []


# ---------------------------------------------------------------------------
# AC10: Programmatic factory injection (REMOVED — spec defers programmatic
# registration; discovery is entry-point only. Tests below replaced by
# test_orchestrator_registry.py which covers the registry-based path.)
# ---------------------------------------------------------------------------
# class TestAC10RegisteredFactory: (deleted)


# ---------------------------------------------------------------------------
# AC11: No regressions — existing resolver-only configs still work
# ---------------------------------------------------------------------------


class TestAC11NoRegressions:
    """AC11: Removing the ConfigError block must not break existing resolver-only configs."""

    def test_static_resolver_layer_still_builds(self):
        """A layer with only a static resolver builds without error after the change."""
        config = _make_minimal_config(layer_transformers=None)

        sr2 = _make_sr2(config)
        assert sr2 is not None

    def test_session_and_input_resolvers_still_build(self):
        """session + input resolvers in a layer are unaffected by transformer changes."""
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[_make_system_resolver()],
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[
                        ResolverConfig(type="session"),
                        ResolverConfig(type="input"),
                    ],
                ),
            ]
        )

        sr2 = _make_sr2(config)
        assert sr2 is not None
        assert len(sr2._engine.layers) == 2

    def test_unknown_resolver_type_raises_plugin_not_found_error(self):
        """Unknown resolver type raises PluginNotFoundError (cutover from ValueError)."""
        config = _make_minimal_config(layer_transformers=None)
        config.layers[0].resolvers.append(
            ResolverConfig(type="nonexistent_resolver")
        )

        with pytest.raises(PluginNotFoundError, match="nonexistent_resolver"):
            _make_sr2(config)

    def test_dict_without_default_key_is_accepted(self):
        """SR2 accepts a dict without a 'default' key (sr2-14: magic string removed)."""
        from sr2.orchestrator import SR2

        config = _make_minimal_config(layer_transformers=None)

        # A dict without "default" is now valid — first value used as driver.
        instance = SR2(
            pipeline_config=config,
            llm={"other": _MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert instance is not None

    def test_config_error_not_raised_for_empty_transformers(self):
        """ConfigError (old behavior) is NOT raised when transformers=[] after the fix."""
        config = _make_minimal_config(layer_transformers=[])

        # Must not raise ConfigError or any exception
        try:
            sr2 = _make_sr2(config)
        except ConfigError:
            pytest.fail(
                "ConfigError was raised for transformers=[]; the old guard block "
                "was not removed."
            )

    def test_config_error_not_raised_for_summarize_transformer(self):
        """ConfigError (old behavior) is NOT raised when 'summarize' transformer is used."""
        config = _make_minimal_config(
            layer_transformers=[TransformerConfig(type="summarize")]
        )
        try:
            _make_sr2(config)
        except ConfigError:
            pytest.fail(
                "ConfigError was raised for 'summarize' transformer; "
                "the old guard block was not removed."
            )
