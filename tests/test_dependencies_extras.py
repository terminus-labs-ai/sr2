"""Tests for FR9/FR10: Dependencies.extras field and SR2.__init__ extras threading.

Covers:
  - Dependencies.extras defaults to empty dict (FR9)
  - Dependencies.extras accepts a mapping (FR9)
  - Dependencies is still frozen/immutable after adding extras (FR9)
  - Dependencies.llm field is unchanged (FR9)
  - SR2 accepts extras kwarg without error (FR10)
  - SR2 without extras still works — backward compat (FR10)
  - extras reaches component build() via deps.extras (FR10 acceptance criterion)
  - extras=None behaves the same as extras={} (FR10)
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
from collections.abc import AsyncIterator, Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import sr2.orchestrator as _orch
from sr2.config.models import (
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMCallable,
    StreamEvent,
)


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

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="text", text="ok")
        yield StreamEvent(type="end")


def _make_minimal_config(resolver_type: str = "static") -> PipelineConfig:
    """One-layer config using a single resolver of the given type."""
    config_kwargs: dict[str, Any] = {}
    if resolver_type == "static":
        config_kwargs = {"text": "You are a helpful assistant."}

    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[ResolverConfig(type=resolver_type, config=config_kwargs)],
            )
        ]
    )


def _make_static_ep_side_effect(group: str) -> list:
    """Return a real-enough entry point for 'static' resolver; [] for everything else."""
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


@pytest.fixture(autouse=False)
def reset_plugin_registries():
    """Reset the module-level PluginRegistry caches before and after each test."""

    def _reset():
        _orch._RESOLVERS._discovered = False
        _orch._RESOLVERS._classes = {}
        _orch._RESOLVERS._collisions = {}
        _orch._TRANSFORMERS._discovered = False
        _orch._TRANSFORMERS._classes = {}
        _orch._TRANSFORMERS._collisions = {}

    _reset()
    yield
    _reset()


def _make_sr2(config: PipelineConfig, **kwargs):
    """Construct SR2 with a mock LLM and CharacterTokenCounter."""
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config,
        llm={"default": _MockLLM()},
        token_counter=CharacterTokenCounter(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Dependencies.extras defaults to empty dict
# ---------------------------------------------------------------------------


class TestDependenciesExtrasDefault:
    """Dependencies.extras is an empty dict when not supplied."""

    def test_extras_defaults_to_empty_dict(self):
        """Constructing Dependencies() with no extras arg yields deps.extras == {}."""
        deps = Dependencies()
        assert deps.extras == {}

    def test_extras_is_mapping_type(self):
        """deps.extras satisfies the Mapping protocol (at minimum, it is dict-like)."""
        deps = Dependencies()
        assert isinstance(deps.extras, Mapping)


# ---------------------------------------------------------------------------
# 2. Dependencies.extras accepts a mapping
# ---------------------------------------------------------------------------


class TestDependenciesExtrasAcceptsMapping:
    """Dependencies.extras stores and exposes the supplied mapping."""

    def test_extras_stores_supplied_dict(self):
        """Constructing with extras={'key': 'val'} stores the value correctly."""
        deps = Dependencies(extras={"key": "val"})
        assert deps.extras["key"] == "val"

    def test_extras_stores_multiple_keys(self):
        """extras with multiple keys are all retrievable."""
        deps = Dependencies(extras={"alpha": 1, "beta": "two", "gamma": [3]})
        assert deps.extras["alpha"] == 1
        assert deps.extras["beta"] == "two"
        assert deps.extras["gamma"] == [3]

    def test_extras_value_can_be_any_type(self):
        """extras values are not restricted to primitive types."""
        sentinel = object()
        deps = Dependencies(extras={"obj": sentinel})
        assert deps.extras["obj"] is sentinel

    def test_extras_accepts_custom_mapping(self):
        """extras accepts any Mapping, not just plain dict."""

        class _CustomMapping(Mapping):
            def __init__(self, data: dict) -> None:
                self._data = data

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        mapping = _CustomMapping({"custom": "value"})
        deps = Dependencies(extras=mapping)
        assert deps.extras["custom"] == "value"


# ---------------------------------------------------------------------------
# 3. Dependencies is still frozen/immutable
# ---------------------------------------------------------------------------


class TestDependenciesImmutability:
    """Dependencies remains frozen after the extras field is added."""

    def test_setting_extras_after_construction_raises(self):
        """Assigning to deps.extras after construction raises an immutability error."""
        deps = Dependencies(extras={"k": "v"})
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.extras = {"other": "value"}  # type: ignore[misc]

    def test_setting_llm_after_construction_raises(self):
        """Assigning to deps.llm after construction still raises an immutability error."""
        deps = Dependencies()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.llm = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. Dependencies.llm field unchanged
# ---------------------------------------------------------------------------


class TestDependenciesLlmFieldUnchanged:
    """Existing llm field behaviour is preserved after adding extras."""

    def test_llm_defaults_to_none(self):
        """Dependencies() with no llm arg yields deps.llm is None."""
        deps = Dependencies()
        assert deps.llm is None

    def test_llm_accepts_dict_of_callables(self):
        """Passing llm={'default': mock} stores the callable correctly."""
        mock_llm = _MockLLM()
        deps = Dependencies(llm={"default": mock_llm})
        assert deps.llm is not None
        assert deps.llm["default"] is mock_llm

    def test_llm_and_extras_coexist(self):
        """Both llm and extras can be set simultaneously without conflict."""
        mock_llm = _MockLLM()
        deps = Dependencies(llm={"default": mock_llm}, extras={"x": 42})
        assert deps.llm["default"] is mock_llm
        assert deps.extras["x"] == 42


# ---------------------------------------------------------------------------
# 5. SR2 accepts extras kwarg without error
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_plugin_registries")
class TestSR2AcceptsExtras:
    """SR2.__init__ accepts an extras keyword argument."""

    def test_sr2_accepts_extras_kwarg(self):
        """Constructing SR2(..., extras={'k': 'v'}) does not raise."""
        config = _make_minimal_config("static")
        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_make_static_ep_side_effect,
        ):
            # Should not raise
            sr2 = _make_sr2(config, extras={"k": "v"})
            assert sr2 is not None

    def test_sr2_extras_kwarg_accepts_various_values(self):
        """extras can hold arbitrary objects without SR2 raising."""
        config = _make_minimal_config("static")
        sentinel = object()
        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_make_static_ep_side_effect,
        ):
            sr2 = _make_sr2(config, extras={"sentinel": sentinel, "count": 99})
            assert sr2 is not None


# ---------------------------------------------------------------------------
# 6. SR2 without extras still works (backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_plugin_registries")
class TestSR2BackwardCompat:
    """Omitting extras from SR2.__init__ is still valid (backward compat)."""

    def test_sr2_without_extras_constructs_successfully(self):
        """SR2 constructed without extras kwarg does not raise."""
        config = _make_minimal_config("static")
        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_make_static_ep_side_effect,
        ):
            sr2 = _make_sr2(config)
            assert sr2 is not None

    def test_sr2_without_extras_behaves_normally(self):
        """SR2 without extras still uses the static resolver without error."""
        config = _make_minimal_config("static")
        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_make_static_ep_side_effect,
        ):
            sr2 = _make_sr2(config)
            # Engine is wired — presence of layers confirms normal construction
            assert len(sr2._engine._layers) == 1


# ---------------------------------------------------------------------------
# 7. extras reaches component build() via deps.extras
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_plugin_registries")
class TestExtrasReachesComponentBuild:
    """FR10 acceptance criterion: extras threading into component build()."""

    def test_extras_value_available_in_resolver_build(self):
        """A resolver's build() can read deps.extras['test_key'].

        A spy resolver records what deps.extras['test_key'] contained at build
        time. SR2 is constructed with extras={'test_key': 'sentinel_value'}.
        The spy confirms it received the sentinel.
        """
        received_extras: dict[str, Any] = {}

        class SpyResolver:
            """Minimal resolver that records extras at build time."""

            name: str = "spy"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []

            def __init__(self, config: Any, captured_value: Any) -> None:
                self._captured_value = captured_value

            @classmethod
            def build(cls, config: Any, deps: Dependencies) -> "SpyResolver":
                # Record what was in extras so the test can assert on it
                received_extras["test_key"] = deps.extras.get("test_key")
                return cls(config, deps.extras.get("test_key"))

            async def resolve(self, events: list) -> Any:
                from sr2.pipeline.models import ResolvedContent

                return ResolvedContent(
                    resolver_name=self.name,
                    source_layer="spy",
                    content=[],
                )

        def _spy_ep_side_effect(group: str) -> list:
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "spy"
                ep.load.return_value = SpyResolver
                dist = MagicMock()
                dist.name = "sr2-test"
                ep.dist = dist
                return [ep]
            # sr2.transformers and everything else → empty
            return []

        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="spy")],
                )
            ]
        )

        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_spy_ep_side_effect,
        ):
            _make_sr2(config, extras={"test_key": "sentinel_value"})

        assert received_extras["test_key"] == "sentinel_value"

    def test_extras_without_key_yields_none_not_error(self):
        """A resolver that calls deps.extras.get('missing') gets None, not KeyError."""
        received: dict[str, Any] = {}

        class SafeSpyResolver:
            name: str = "safe_spy"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []

            def __init__(self) -> None:
                pass

            @classmethod
            def build(cls, config: Any, deps: Dependencies) -> "SafeSpyResolver":
                received["val"] = deps.extras.get("nonexistent_key")
                return cls()

            async def resolve(self, events: list) -> Any:
                from sr2.pipeline.models import ResolvedContent

                return ResolvedContent(
                    resolver_name=self.name,
                    source_layer="safe_spy",
                    content=[],
                )

        def _safe_spy_ep_side_effect(group: str) -> list:
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "safe_spy"
                ep.load.return_value = SafeSpyResolver
                dist = MagicMock()
                dist.name = "sr2-test"
                ep.dist = dist
                return [ep]
            return []

        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="safe_spy")],
                )
            ]
        )

        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_safe_spy_ep_side_effect,
        ):
            _make_sr2(config, extras={"other_key": "irrelevant"})

        assert received["val"] is None


# ---------------------------------------------------------------------------
# 8. extras=None behaves the same as extras={}
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_plugin_registries")
class TestExtrasNoneEquivalentToEmpty:
    """SR2 normalises extras=None to an empty dict."""

    def test_dependencies_extras_is_empty_dict_when_sr2_receives_none(self):
        """Passing extras=None to SR2 stores {} in deps.extras, not None.

        We verify indirectly via a spy resolver that reads deps.extras directly.
        """
        received: dict[str, Any] = {}

        class ExtrasSpyResolver:
            name: str = "extras_spy"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []

            def __init__(self) -> None:
                pass

            @classmethod
            def build(cls, config: Any, deps: Dependencies) -> "ExtrasSpyResolver":
                # Store the actual extras object (not a copy)
                received["extras"] = deps.extras
                return cls()

            async def resolve(self, events: list) -> Any:
                from sr2.pipeline.models import ResolvedContent

                return ResolvedContent(
                    resolver_name=self.name,
                    source_layer="extras_spy",
                    content=[],
                )

        def _extras_spy_ep_side_effect(group: str) -> list:
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "extras_spy"
                ep.load.return_value = ExtrasSpyResolver
                dist = MagicMock()
                dist.name = "sr2-test"
                ep.dist = dist
                return [ep]
            return []

        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="extras_spy")],
                )
            ]
        )

        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_extras_spy_ep_side_effect,
        ):
            _make_sr2(config, extras=None)

        # Must not be None — SR2 should have normalised it to {}
        assert received["extras"] is not None
        assert received["extras"] == {}

    def test_sr2_extras_none_does_not_raise(self):
        """SR2(..., extras=None) constructs without raising."""
        config = _make_minimal_config("static")
        with patch(
            "sr2.plugins.registry.entry_points",
            side_effect=_make_static_ep_side_effect,
        ):
            sr2 = _make_sr2(config, extras=None)
            assert sr2 is not None
