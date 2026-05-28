"""Tests for Dependencies dataclass and protocol build() classmethod additions.

Step 1 of the factory-injection refactor. These tests will FAIL until
sr2/pipeline/dependencies.py exists and Resolver/Transformer protocols
include the build() classmethod.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.protocols import Resolver, Transformer


# ---------------------------------------------------------------------------
# Helpers — minimal concrete classes for protocol conformance checks
# ---------------------------------------------------------------------------


class _MinimalResolver:
    """Implements the pre-existing Resolver surface (no build)."""

    subscriptions: list = []
    max_executions: int = 1

    async def resolve(self, events):
        ...


class _FullResolver:
    """Implements the full Resolver surface including build."""

    subscriptions: list = []
    max_executions: int = 1

    async def resolve(self, events):
        ...

    @classmethod
    def build(cls, config, deps):
        return cls()


class _MinimalTransformer:
    """Implements the pre-existing Transformer surface (no build)."""

    subscriptions: list = []
    max_executions: int = 1

    async def transform(self, content, events):
        ...


class _FullTransformer:
    """Implements the full Transformer surface including build."""

    subscriptions: list = []
    max_executions: int = 1

    async def transform(self, content, events):
        ...

    @classmethod
    def build(cls, config, deps):
        return cls()


# ---------------------------------------------------------------------------
# Dependencies — construction
# ---------------------------------------------------------------------------


class TestDependenciesConstruction:
    def test_no_args_llm_is_none(self):
        deps = Dependencies()
        assert deps.llm is None

    def test_explicit_none(self):
        deps = Dependencies(llm=None)
        assert deps.llm is None

    def test_with_llm_dict(self):
        def fake_llm(*args, **kwargs):
            return "response"

        deps = Dependencies(llm={"default": fake_llm})
        assert deps.llm is not None
        assert "default" in deps.llm
        assert deps.llm["default"] is fake_llm

    def test_with_multiple_llm_entries(self):
        def llm_a(*args, **kwargs): ...
        def llm_b(*args, **kwargs): ...

        deps = Dependencies(llm={"a": llm_a, "b": llm_b})
        assert len(deps.llm) == 2
        assert deps.llm["a"] is llm_a
        assert deps.llm["b"] is llm_b


# ---------------------------------------------------------------------------
# Dependencies — immutability
# ---------------------------------------------------------------------------


class TestDependenciesImmutability:
    def test_frozen_set_llm_raises(self):
        deps = Dependencies()
        with pytest.raises(dataclasses.FrozenInstanceError):
            deps.llm = {"x": lambda: None}

    def test_frozen_set_unknown_attr_raises(self):
        deps = Dependencies()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.extra = "anything"


# ---------------------------------------------------------------------------
# Dependencies — equality
# ---------------------------------------------------------------------------


class TestDependenciesEquality:
    def test_two_empty_instances_are_equal(self):
        assert Dependencies() == Dependencies()

    def test_same_llm_dict_are_equal(self):
        def llm(*args, **kwargs): ...
        d1 = Dependencies(llm={"k": llm})
        d2 = Dependencies(llm={"k": llm})
        assert d1 == d2

    def test_different_llm_dicts_are_not_equal(self):
        def llm_a(*args, **kwargs): ...
        def llm_b(*args, **kwargs): ...
        d1 = Dependencies(llm={"k": llm_a})
        d2 = Dependencies(llm={"k": llm_b})
        assert d1 != d2

    def test_none_vs_dict_not_equal(self):
        d1 = Dependencies(llm=None)
        d2 = Dependencies(llm={"k": lambda: None})
        assert d1 != d2


# ---------------------------------------------------------------------------
# Resolver protocol — build classmethod requirement
# ---------------------------------------------------------------------------


class TestResolverProtocolBuild:
    def test_without_build_does_not_satisfy_resolver(self):
        obj = _MinimalResolver()
        assert not isinstance(obj, Resolver), (
            "A class missing build() should not satisfy the Resolver protocol"
        )

    def test_with_build_satisfies_resolver(self):
        obj = _FullResolver()
        assert isinstance(obj, Resolver), (
            "A class implementing all required members including build() "
            "should satisfy the Resolver protocol"
        )

    def test_resolver_class_with_build_has_build_attribute(self):
        assert hasattr(_FullResolver, "build")

    def test_resolver_class_without_build_has_no_build_attribute(self):
        assert not hasattr(_MinimalResolver, "build")

    def test_build_is_callable_on_full_resolver(self):
        assert callable(getattr(_FullResolver, "build", None))


# ---------------------------------------------------------------------------
# Transformer protocol — build classmethod requirement
# ---------------------------------------------------------------------------


class TestTransformerProtocolBuild:
    def test_without_build_does_not_satisfy_transformer(self):
        obj = _MinimalTransformer()
        assert not isinstance(obj, Transformer), (
            "A class missing build() should not satisfy the Transformer protocol"
        )

    def test_with_build_satisfies_transformer(self):
        obj = _FullTransformer()
        assert isinstance(obj, Transformer), (
            "A class implementing all required members including build() "
            "should satisfy the Transformer protocol"
        )

    def test_transformer_class_with_build_has_build_attribute(self):
        assert hasattr(_FullTransformer, "build")

    def test_transformer_class_without_build_has_no_build_attribute(self):
        assert not hasattr(_MinimalTransformer, "build")

    def test_build_is_callable_on_full_transformer(self):
        assert callable(getattr(_FullTransformer, "build", None))


# ---------------------------------------------------------------------------
# Cross-protocol — build receives Dependencies (smoke-level)
# ---------------------------------------------------------------------------


class TestBuildReceivesDependencies:
    """Verify that build() can accept a Dependencies instance without error.

    This does not test implementation behavior — just that the contract
    is honoured at the call site.
    """

    def test_resolver_build_accepts_dependencies(self):
        deps = Dependencies()
        instance = _FullResolver.build(config=None, deps=deps)
        assert instance is not None

    def test_transformer_build_accepts_dependencies(self):
        deps = Dependencies()
        instance = _FullTransformer.build(config=None, deps=deps)
        assert instance is not None

    def test_resolver_build_accepts_dependencies_with_llm(self):
        def fake_llm(*args, **kwargs): ...
        deps = Dependencies(llm={"default": fake_llm})
        instance = _FullResolver.build(config=None, deps=deps)
        assert instance is not None

    def test_transformer_build_accepts_dependencies_with_llm(self):
        def fake_llm(*args, **kwargs): ...
        deps = Dependencies(llm={"default": fake_llm})
        instance = _FullTransformer.build(config=None, deps=deps)
        assert instance is not None


# ===========================================================================
# Tests from test_dependencies_typed_fields.py — SR2-12: typed memory fields
# Dependencies.extras Service Locator antipattern — typed field fix.
# ===========================================================================

from sr2.memory import InMemoryMemoryStore, Memory, MemoryScope
from sr2.memory.protocol import MemoryExtractor, MemoryStore
from sr2.memory.schema import ExtractionResult
from sr2.config.models import ConfigError, ResolverConfig, TransformerConfig as _TransformerConfig


class _StubExtractor:
    """Always returns zero memories; satisfies MemoryExtractor protocol."""

    def extract(self, turn_text: str, turn_id=None) -> ExtractionResult:
        return ExtractionResult(memories=[], source_turn_id=turn_id)


def _make_transformer_config_typed(max_executions: int = 10) -> _TransformerConfig:
    return _TransformerConfig(type="memory_extraction", max_executions=max_executions)


def _make_resolver_config_typed(**kwargs) -> ResolverConfig:
    return ResolverConfig(type="memory", **kwargs)


class TestDependenciesHasTypedMemoryStoreField:
    """Dependencies must expose `memory_store` as a first-class typed field."""

    def test_dependencies_has_memory_store_field(self):
        """When memory_store is passed to Dependencies, MemoryResolver.build() can find it."""
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config_typed()
        result = MemoryResolver.build(config, deps)
        assert result is not None

    def test_memory_store_field_defaults_to_none(self):
        deps = Dependencies()
        assert deps.memory_store is None  # type: ignore[attr-defined]

    def test_memory_store_field_accepts_memory_store_instance(self):
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        assert deps.memory_store is store  # type: ignore[attr-defined]

    def test_memory_store_is_not_fetched_via_extras(self):
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        assert deps.extras == {}  # type: ignore[attr-defined]
        assert deps.memory_store is store  # type: ignore[attr-defined]

    def test_dependencies_remains_frozen_with_new_field(self):
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.memory_store = None  # type: ignore[misc]


class TestDependenciesHasTypedMemoryExtractorField:
    """Dependencies must expose `memory_extractor` as a first-class typed field."""

    def test_dependencies_has_memory_extractor_field(self):
        """When memory_extractor is passed to Dependencies, MemoryExtractionTransformer.build() finds it."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()
        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_memory_extractor_field_defaults_to_none(self):
        deps = Dependencies()
        assert deps.memory_extractor is None  # type: ignore[attr-defined]

    def test_memory_extractor_field_accepts_extractor_instance(self):
        ext = _StubExtractor()
        deps = Dependencies(memory_extractor=ext)  # type: ignore[call-arg]
        assert deps.memory_extractor is ext  # type: ignore[attr-defined]

    def test_both_typed_fields_coexist(self):
        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
        assert deps.memory_store is store  # type: ignore[attr-defined]
        assert deps.memory_extractor is ext  # type: ignore[attr-defined]


class TestExtractionTransformerBuildReadsTypedFields:
    """MemoryExtractionTransformer.build() must use typed deps fields, not extras."""

    def test_build_succeeds_with_typed_memory_store_field(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_build_raises_config_error_when_typed_field_is_none(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        deps = Dependencies()
        config = _make_transformer_config_typed()

        with pytest.raises(ConfigError):
            MemoryExtractionTransformer.build(config, deps)

    def test_build_uses_typed_memory_extractor_when_set(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._extractor is ext

    def test_build_does_not_require_extras_for_memory_store(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store, extras={})  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_build_does_not_require_extras_for_memory_extractor(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext, extras={})  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._extractor is ext


class TestMemoryResolverBuildReadsTypedFields:
    """MemoryResolver.build() must use typed deps fields, not extras."""

    def test_build_succeeds_with_typed_memory_store_field(self):
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config_typed()

        result = MemoryResolver.build(config, deps)
        assert result is not None

    def test_build_raises_config_error_when_typed_field_is_none(self):
        from sr2.memory.memory_resolver import MemoryResolver

        deps = Dependencies()
        config = _make_resolver_config_typed()

        with pytest.raises(ConfigError):
            MemoryResolver.build(config, deps)

    def test_build_does_not_require_extras_for_memory_store(self):
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store, extras={})  # type: ignore[call-arg]
        config = _make_resolver_config_typed()

        result = MemoryResolver.build(config, deps)
        assert result is not None

    def test_resolver_uses_typed_field_store_at_resolve_time(self):
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config_typed()
        resolver = MemoryResolver.build(config, deps)

        assert resolver._store is store


class TestSR2AcceptsTypedMemoryStoreKwarg:
    """SR2.__init__ must accept memory_store as a typed kwarg (not require extras)."""

    def test_sr2_init_signature_has_memory_store_param(self):
        import inspect
        from sr2.orchestrator import SR2

        sig = inspect.signature(SR2.__init__)
        assert "memory_store" in sig.parameters, (
            f"SR2.__init__ should accept `memory_store`. Current params: {list(sig.parameters)}"
        )

    def test_sr2_init_signature_has_memory_extractor_param(self):
        import inspect
        from sr2.orchestrator import SR2

        sig = inspect.signature(SR2.__init__)
        assert "memory_extractor" in sig.parameters, (
            f"SR2.__init__ should accept `memory_extractor`. Current params: {list(sig.parameters)}"
        )

    def test_sr2_constructs_with_memory_store_kwarg(self):
        import importlib.metadata
        from unittest.mock import MagicMock, patch

        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig
        from sr2.pipeline.token_counting import CharacterTokenCounter
        from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent as _SE
        from sr2.models import TextBlock, TokenUsage
        from sr2.orchestrator import SR2
        import sr2.orchestrator as _orch
        from collections.abc import AsyncIterator

        class _MockLLM2:
            async def complete(self, req: CompletionRequest) -> CompletionResponse:
                return CompletionResponse(
                    id="mock", content=[TextBlock(text="ok")],
                    stop_reason="end_turn", usage=TokenUsage(),
                )

            async def stream(self, req: CompletionRequest) -> AsyncIterator[_SE]:
                yield _SE(type="text", text="ok")
                yield _SE(type="end")

        def _static_ep(group: str) -> list:
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

        _orch._RESOLVERS._discovered = False
        _orch._RESOLVERS._classes = {}
        _orch._TRANSFORMERS._discovered = False
        _orch._TRANSFORMERS._classes = {}

        store = InMemoryMemoryStore()
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "Hello"})],
                )
            ]
        )

        with patch("sr2.plugins.registry.entry_points", side_effect=_static_ep):
            sr2 = SR2(
                pipeline_config=config,
                llm={"default": _MockLLM2()},
                token_counter=CharacterTokenCounter(),
                memory_store=store,
            )
        assert sr2 is not None

        _orch._RESOLVERS._discovered = False
        _orch._RESOLVERS._classes = {}
        _orch._TRANSFORMERS._discovered = False
        _orch._TRANSFORMERS._classes = {}


class TestExtrasNotRequiredForMemorySubsystem:
    """After the fix, the memory subsystem must work without touching extras at all."""

    def test_dependencies_with_only_typed_fields_is_sufficient_for_extraction(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        result = MemoryExtractionTransformer.build(config, deps)
        assert result._store is store

    def test_dependencies_with_only_typed_fields_is_sufficient_for_resolver(self):
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config_typed()

        result = MemoryResolver.build(config, deps)
        assert result._store is store

    def test_typed_field_store_identity_preserved_through_build(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._store is store

    def test_typed_extractor_identity_preserved_through_build(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._extractor is ext


class TestTypedFieldsOnly:
    """After the extras-path removal, typed fields are the sole injection path."""

    def test_typed_field_works_for_extraction_transformer(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._store is store

    def test_typed_field_works_for_memory_resolver(self):
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config_typed()

        resolver = MemoryResolver.build(config, deps)
        assert resolver._store is store

    def test_typed_field_identity_preserved_through_build(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(  # type: ignore[call-arg]
            memory_store=store,
        )
        config = _make_transformer_config_typed()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._store is store


# ===========================================================================
# Tests from test_dependencies_extras.py — FR9/FR10: Dependencies.extras field
# and SR2.__init__ extras threading.
# ===========================================================================

import importlib.metadata as _importlib_metadata_extras
from collections.abc import Mapping as _Mapping
from typing import Any as _Any
from unittest.mock import MagicMock as _MagicMock, patch as _patch
import sr2.orchestrator as _orch_extras

from sr2.config.models import (
    LayerConfig as _LayerConfig_extras,
    PipelineConfig as _PipelineConfig_extras,
    ResolverConfig as _ResolverConfig_extras,
)
from sr2.models import TextBlock as _TextBlock_extras, TokenUsage as _TokenUsage_extras
from sr2.pipeline.token_counting import CharacterTokenCounter as _CharTokenCounter_extras
from sr2.protocols.llm import (
    CompletionRequest as _CompletionRequest_extras,
    CompletionResponse as _CompletionResponse_extras,
    StreamEvent as _StreamEvent_extras,
)


class _MockLLMExtras:
    """Minimal LLMCallable for constructing SR2 in extras tests."""

    async def complete(self, request: _CompletionRequest_extras) -> _CompletionResponse_extras:
        return _CompletionResponse_extras(
            id="mock-resp",
            content=[_TextBlock_extras(text="ok")],
            stop_reason="end_turn",
            usage=_TokenUsage_extras(),
        )

    async def stream(self, request: _CompletionRequest_extras):
        from collections.abc import AsyncIterator
        yield _StreamEvent_extras(type="text", text="ok")
        yield _StreamEvent_extras(type="end")


def _make_minimal_config_extras(resolver_type: str = "static") -> _PipelineConfig_extras:
    config_kwargs: dict[str, _Any] = {}
    if resolver_type == "static":
        config_kwargs = {"text": "You are a helpful assistant."}
    return _PipelineConfig_extras(
        layers=[
            _LayerConfig_extras(
                name="system",
                target="system",
                resolvers=[_ResolverConfig_extras(type=resolver_type, config=config_kwargs)],
            )
        ]
    )


def _make_static_ep_side_effect_extras(group: str) -> list:
    if group == "sr2.resolvers":
        from sr2.pipeline.resolvers.static import StaticResolver
        ep = _MagicMock(spec=_importlib_metadata_extras.EntryPoint)
        ep.name = "static"
        ep.load.return_value = StaticResolver
        dist = _MagicMock()
        dist.name = "sr2"
        ep.dist = dist
        return [ep]
    return []


@pytest.fixture(autouse=False)
def reset_plugin_registries_extras():
    def _reset():
        _orch_extras._RESOLVERS._discovered = False
        _orch_extras._RESOLVERS._classes = {}
        _orch_extras._RESOLVERS._collisions = {}
        _orch_extras._TRANSFORMERS._discovered = False
        _orch_extras._TRANSFORMERS._classes = {}
        _orch_extras._TRANSFORMERS._collisions = {}
    _reset()
    yield
    _reset()


def _make_sr2_extras(config: _PipelineConfig_extras, **kwargs):
    from sr2.orchestrator import SR2
    return SR2(
        pipeline_config=config,
        llm={"default": _MockLLMExtras()},
        token_counter=_CharTokenCounter_extras(),
        **kwargs,
    )


class TestDependenciesExtrasDefault:
    """Dependencies.extras is an empty dict when not supplied."""

    def test_extras_defaults_to_empty_dict(self):
        deps = Dependencies()
        assert deps.extras == {}

    def test_extras_is_mapping_type(self):
        deps = Dependencies()
        assert isinstance(deps.extras, _Mapping)


class TestDependenciesExtrasAcceptsMapping:
    """Dependencies.extras stores and exposes the supplied mapping."""

    def test_extras_stores_supplied_dict(self):
        deps = Dependencies(extras={"key": "val"})
        assert deps.extras["key"] == "val"

    def test_extras_stores_multiple_keys(self):
        deps = Dependencies(extras={"alpha": 1, "beta": "two", "gamma": [3]})
        assert deps.extras["alpha"] == 1
        assert deps.extras["beta"] == "two"
        assert deps.extras["gamma"] == [3]

    def test_extras_value_can_be_any_type(self):
        sentinel = object()
        deps = Dependencies(extras={"obj": sentinel})
        assert deps.extras["obj"] is sentinel

    def test_extras_accepts_custom_mapping(self):
        class _CustomMapping(_Mapping):
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


class TestDependenciesExtrasImmutability:
    """Dependencies remains frozen after the extras field is added."""

    def test_setting_extras_after_construction_raises(self):
        deps = Dependencies(extras={"k": "v"})
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.extras = {"other": "value"}  # type: ignore[misc]

    def test_setting_llm_after_construction_raises(self):
        deps = Dependencies()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.llm = {}  # type: ignore[misc]


class TestDependenciesLlmFieldUnchanged:
    """Existing llm field behaviour is preserved after adding extras."""

    def test_llm_defaults_to_none(self):
        deps = Dependencies()
        assert deps.llm is None

    def test_llm_accepts_dict_of_callables(self):
        mock_llm = _MockLLMExtras()
        deps = Dependencies(llm={"default": mock_llm})
        assert deps.llm is not None
        assert deps.llm["default"] is mock_llm

    def test_llm_and_extras_coexist(self):
        mock_llm = _MockLLMExtras()
        deps = Dependencies(llm={"default": mock_llm}, extras={"x": 42})
        assert deps.llm["default"] is mock_llm
        assert deps.extras["x"] == 42


@pytest.mark.usefixtures("reset_plugin_registries_extras")
class TestSR2AcceptsExtras:
    """SR2.__init__ accepts an extras keyword argument."""

    def test_sr2_accepts_extras_kwarg(self):
        config = _make_minimal_config_extras("static")
        with _patch("sr2.plugins.registry.entry_points", side_effect=_make_static_ep_side_effect_extras):
            sr2 = _make_sr2_extras(config, extras={"k": "v"})
            assert sr2 is not None

    def test_sr2_extras_kwarg_accepts_various_values(self):
        config = _make_minimal_config_extras("static")
        sentinel = object()
        with _patch("sr2.plugins.registry.entry_points", side_effect=_make_static_ep_side_effect_extras):
            sr2 = _make_sr2_extras(config, extras={"sentinel": sentinel, "count": 99})
            assert sr2 is not None


@pytest.mark.usefixtures("reset_plugin_registries_extras")
class TestSR2BackwardCompat:
    """Omitting extras from SR2.__init__ is still valid (backward compat)."""

    def test_sr2_without_extras_constructs_successfully(self):
        config = _make_minimal_config_extras("static")
        with _patch("sr2.plugins.registry.entry_points", side_effect=_make_static_ep_side_effect_extras):
            sr2 = _make_sr2_extras(config)
            assert sr2 is not None

    def test_sr2_without_extras_behaves_normally(self):
        config = _make_minimal_config_extras("static")
        with _patch("sr2.plugins.registry.entry_points", side_effect=_make_static_ep_side_effect_extras):
            sr2 = _make_sr2_extras(config)
            assert len(sr2._engine.layers) == 1


@pytest.mark.usefixtures("reset_plugin_registries_extras")
class TestExtrasReachesComponentBuild:
    """FR10 acceptance criterion: extras threading into component build()."""

    def test_extras_value_available_in_resolver_build(self):
        received_extras: dict[str, _Any] = {}

        class SpyResolver:
            name: str = "spy"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []

            def __init__(self, config: _Any, captured_value: _Any) -> None:
                self._captured_value = captured_value

            @classmethod
            def build(cls, config: _Any, deps: Dependencies) -> "SpyResolver":
                received_extras["test_key"] = deps.extras.get("test_key")
                return cls(config, deps.extras.get("test_key"))

            async def resolve(self, events: list) -> _Any:
                from sr2.pipeline.models import ResolvedContent
                return ResolvedContent(resolver_name=self.name, source_layer="spy", content=[])

        def _spy_ep_side_effect(group: str) -> list:
            if group == "sr2.resolvers":
                ep = _MagicMock(spec=_importlib_metadata_extras.EntryPoint)
                ep.name = "spy"
                ep.load.return_value = SpyResolver
                dist = _MagicMock()
                dist.name = "sr2-test"
                ep.dist = dist
                return [ep]
            return []

        config = _PipelineConfig_extras(
            layers=[
                _LayerConfig_extras(
                    name="system",
                    target="system",
                    resolvers=[_ResolverConfig_extras(type="spy")],
                )
            ]
        )

        with _patch("sr2.plugins.registry.entry_points", side_effect=_spy_ep_side_effect):
            _make_sr2_extras(config, extras={"test_key": "sentinel_value"})

        assert received_extras["test_key"] == "sentinel_value"

    def test_extras_without_key_yields_none_not_error(self):
        received: dict[str, _Any] = {}

        class SafeSpyResolver:
            name: str = "safe_spy"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []

            def __init__(self) -> None:
                pass

            @classmethod
            def build(cls, config: _Any, deps: Dependencies) -> "SafeSpyResolver":
                received["val"] = deps.extras.get("nonexistent_key")
                return cls()

            async def resolve(self, events: list) -> _Any:
                from sr2.pipeline.models import ResolvedContent
                return ResolvedContent(resolver_name=self.name, source_layer="safe_spy", content=[])

        def _safe_spy_ep_side_effect(group: str) -> list:
            if group == "sr2.resolvers":
                ep = _MagicMock(spec=_importlib_metadata_extras.EntryPoint)
                ep.name = "safe_spy"
                ep.load.return_value = SafeSpyResolver
                dist = _MagicMock()
                dist.name = "sr2-test"
                ep.dist = dist
                return [ep]
            return []

        config = _PipelineConfig_extras(
            layers=[
                _LayerConfig_extras(
                    name="system",
                    target="system",
                    resolvers=[_ResolverConfig_extras(type="safe_spy")],
                )
            ]
        )

        with _patch("sr2.plugins.registry.entry_points", side_effect=_safe_spy_ep_side_effect):
            _make_sr2_extras(config, extras={"other_key": "irrelevant"})

        assert received["val"] is None


@pytest.mark.usefixtures("reset_plugin_registries_extras")
class TestExtrasNoneEquivalentToEmpty:
    """SR2 normalises extras=None to an empty dict."""

    def test_dependencies_extras_is_empty_dict_when_sr2_receives_none(self):
        received: dict[str, _Any] = {}

        class ExtrasSpyResolver:
            name: str = "extras_spy"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []

            def __init__(self) -> None:
                pass

            @classmethod
            def build(cls, config: _Any, deps: Dependencies) -> "ExtrasSpyResolver":
                received["extras"] = deps.extras
                return cls()

            async def resolve(self, events: list) -> _Any:
                from sr2.pipeline.models import ResolvedContent
                return ResolvedContent(resolver_name=self.name, source_layer="extras_spy", content=[])

        def _extras_spy_ep_side_effect(group: str) -> list:
            if group == "sr2.resolvers":
                ep = _MagicMock(spec=_importlib_metadata_extras.EntryPoint)
                ep.name = "extras_spy"
                ep.load.return_value = ExtrasSpyResolver
                dist = _MagicMock()
                dist.name = "sr2-test"
                ep.dist = dist
                return [ep]
            return []

        config = _PipelineConfig_extras(
            layers=[
                _LayerConfig_extras(
                    name="system",
                    target="system",
                    resolvers=[_ResolverConfig_extras(type="extras_spy")],
                )
            ]
        )

        with _patch("sr2.plugins.registry.entry_points", side_effect=_extras_spy_ep_side_effect):
            _make_sr2_extras(config, extras=None)

        assert received["extras"] is not None
        assert received["extras"] == {}

    def test_sr2_extras_none_does_not_raise(self):
        config = _make_minimal_config_extras("static")
        with _patch("sr2.plugins.registry.entry_points", side_effect=_make_static_ep_side_effect_extras):
            sr2 = _make_sr2_extras(config, extras=None)
            assert sr2 is not None
