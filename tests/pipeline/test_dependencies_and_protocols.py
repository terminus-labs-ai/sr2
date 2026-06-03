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
    execution_count: int = 0

    async def resolve(self, events):
        ...


class _FullResolver:
    """Implements the full Resolver surface including build."""

    subscriptions: list = []
    max_executions: int = 1
    execution_count: int = 0

    async def resolve(self, events):
        ...

    @classmethod
    def build(cls, config, deps):
        return cls()


class _MinimalTransformer:
    """Implements the pre-existing Transformer surface (no build)."""

    subscriptions: list = []
    max_executions: int = 1
    execution_count: int = 0

    async def transform(self, content, events):
        ...


class _FullTransformer:
    """Implements the full Transformer surface including build."""

    subscriptions: list = []
    max_executions: int = 1
    execution_count: int = 0

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
        """memory_store is available as a typed field, no service-locator needed."""
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
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
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config_typed()

        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_build_does_not_require_extras_for_memory_extractor(self):
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
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
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
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



