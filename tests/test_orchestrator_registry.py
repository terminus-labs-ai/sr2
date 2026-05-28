"""Tests for the PluginRegistry-based resolver and transformer lookup in orchestrator.

Covers:
  - _RESOLVERS and _TRANSFORMERS are PluginRegistry instances
  - Unknown resolver type → PluginNotFoundError (with patched entry_points)
  - Unknown transformer type → PluginNotFoundError (with patched entry_points)
  - PluginNotFoundError message contains the unknown type name
"""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest

from sr2.config.models import (
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.plugins.errors import PluginNotFoundError
from sr2.plugins.registry import PluginRegistry
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


def _make_sr2(config: PipelineConfig):
    """Construct SR2 with a mock LLM and CharacterTokenCounter."""
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config,
        llm={"default": _MockLLM()},
        token_counter=CharacterTokenCounter(),
    )


def _make_config_with_resolver(resolver_type: str) -> PipelineConfig:
    """One-layer config using a single resolver of the given type."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[ResolverConfig(type=resolver_type)],
            )
        ]
    )


def _make_config_with_transformer(transformer_type: str) -> PipelineConfig:
    """One-layer config (static resolver) with a single transformer of the given type."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
                transformers=[TransformerConfig(type=transformer_type)],
            )
        ]
    )


def _static_ep_side_effect(group: str):
    """Return a real-enough entry point for 'static' resolver, [] for anything else.

    Used in transformer tests so the resolver lookup succeeds and the
    PluginNotFoundError is raised for the unknown transformer, not the resolver.
    """
    if group == "sr2.resolvers":
        from sr2.pipeline.resolvers.static import StaticResolver

        ep = MagicMock(spec=importlib.metadata.EntryPoint)
        ep.name = "static"
        ep.load.return_value = StaticResolver
        dist = MagicMock()
        dist.name = "sr2"
        ep.dist = dist
        return [ep]
    return []


# ---------------------------------------------------------------------------
# 1. Module-level registries are PluginRegistry instances
# ---------------------------------------------------------------------------


class TestRegistryTypes:
    """_RESOLVERS and _TRANSFORMERS on the orchestrator module are PluginRegistry instances."""

    def test_resolvers_is_plugin_registry(self):
        """orchestrator._RESOLVERS is a PluginRegistry."""
        from sr2.orchestrator import _RESOLVERS

        assert isinstance(_RESOLVERS, PluginRegistry)

    def test_transformers_is_plugin_registry(self):
        """orchestrator._TRANSFORMERS is a PluginRegistry."""
        from sr2.orchestrator import _TRANSFORMERS

        assert isinstance(_TRANSFORMERS, PluginRegistry)


# ---------------------------------------------------------------------------
# 2. Unknown resolver type → PluginNotFoundError
# ---------------------------------------------------------------------------


class TestUnknownResolverType:
    """Unknown resolver type raises PluginNotFoundError, not ValueError."""

    def test_unknown_resolver_raises_plugin_not_found_error(self):
        """Constructing SR2 with an unknown resolver type raises PluginNotFoundError.

        entry_points is patched to return [] so the test is independent of
        the installed package state.
        """
        config = _make_config_with_resolver("totally_made_up_resolver")

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            with pytest.raises(PluginNotFoundError):
                _make_sr2(config)

    def test_unknown_resolver_is_subclass_of_import_error(self):
        """PluginNotFoundError is a subclass of ImportError (backward-compat contract)."""
        config = _make_config_with_resolver("ghost_resolver")

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            with pytest.raises(ImportError):
                _make_sr2(config)

    def test_unknown_resolver_message_contains_type_name(self):
        """The PluginNotFoundError message contains the requested type name."""
        unknown_type = "resolver_xyzzy_unknown"
        config = _make_config_with_resolver(unknown_type)

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            with pytest.raises(PluginNotFoundError, match=unknown_type):
                _make_sr2(config)


# ---------------------------------------------------------------------------
# 3. Unknown transformer type → PluginNotFoundError
# ---------------------------------------------------------------------------


class TestUnknownTransformerType:
    """Unknown transformer type raises PluginNotFoundError, not ValueError."""

    def test_unknown_transformer_raises_plugin_not_found_error(self):
        """Constructing SR2 with an unknown transformer type raises PluginNotFoundError.

        entry_points is patched with a group-aware side_effect so the static
        resolver lookup succeeds and the error is raised for the transformer.
        """
        config = _make_config_with_transformer("totally_made_up_transformer")

        with patch("sr2.plugins.registry.entry_points", side_effect=_static_ep_side_effect):
            with pytest.raises(PluginNotFoundError):
                _make_sr2(config)

    def test_unknown_transformer_is_subclass_of_import_error(self):
        """PluginNotFoundError is a subclass of ImportError (backward-compat contract)."""
        config = _make_config_with_transformer("ghost_transformer")

        with patch("sr2.plugins.registry.entry_points", side_effect=_static_ep_side_effect):
            with pytest.raises(ImportError):
                _make_sr2(config)


# ---------------------------------------------------------------------------
# 4. PluginNotFoundError message contains the unknown type name
# ---------------------------------------------------------------------------


class TestPluginNotFoundErrorMessage:
    """PluginNotFoundError raised for unknown types always names the requested type."""

    def test_resolver_error_message_contains_type_name(self):
        """PluginNotFoundError for an unknown resolver names the type in the message."""
        unknown = "resolver_name_in_error_abc123"
        config = _make_config_with_resolver(unknown)

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            with pytest.raises(PluginNotFoundError) as exc_info:
                _make_sr2(config)

        assert unknown in str(exc_info.value)

    def test_transformer_error_message_contains_type_name(self):
        """PluginNotFoundError for an unknown transformer names the type in the message."""
        unknown = "transformer_name_in_error_xyz789"
        config = _make_config_with_transformer(unknown)

        with patch("sr2.plugins.registry.entry_points", side_effect=_static_ep_side_effect):
            with pytest.raises(PluginNotFoundError) as exc_info:
                _make_sr2(config)

        assert unknown in str(exc_info.value)

    def test_error_exposes_name_attribute(self):
        """PluginNotFoundError has a .name attribute equal to the requested type."""
        unknown = "sentinel_unknown_type"
        config = _make_config_with_resolver(unknown)

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            with pytest.raises(PluginNotFoundError) as exc_info:
                _make_sr2(config)

        assert exc_info.value.name == unknown
