"""Failing tests for sr2-12: Dependencies.extras Service Locator antipattern.

These tests pin the DESIRED behavior after the fix:
  - Dependencies has typed fields `memory_store` and `memory_extractor`
    instead of routing them through the untyped `extras: Mapping[str, Any]` bag.
  - MemoryExtractionTransformer.build() reads from typed fields on Dependencies.
  - MemoryResolver.build() reads from typed fields on Dependencies.
  - SR2.__init__ accepts `memory_store` and `memory_extractor` as typed kwargs.
  - Callers using the old string-key extras pattern get a clear error or the
    typed path works in isolation (no silent breakage).

All tests here MUST FAIL against the current implementation (before the fix)
and PASS after the fix is applied.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

import pytest

from sr2.config.models import ConfigError, ResolverConfig, TransformerConfig
from sr2.memory import InMemoryMemoryStore, Memory, MemoryScope
from sr2.memory.protocol import MemoryExtractor, MemoryStore
from sr2.memory.schema import ExtractionResult
from sr2.pipeline.dependencies import Dependencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubExtractor:
    """Always returns zero memories; satisfies MemoryExtractor protocol."""

    def extract(
        self,
        turn_text: str,
        turn_id: str | None = None,
    ) -> ExtractionResult:
        return ExtractionResult(memories=[], source_turn_id=turn_id)


def _make_transformer_config(max_executions: int = 10) -> TransformerConfig:
    return TransformerConfig(type="memory_extraction", max_executions=max_executions)


def _make_resolver_config(**kwargs) -> ResolverConfig:
    return ResolverConfig(type="memory", **kwargs)


# ---------------------------------------------------------------------------
# 1. Dependencies has a typed `memory_store` field
# ---------------------------------------------------------------------------


class TestDependenciesHasTypedMemoryStoreField:
    """Dependencies must expose `memory_store` as a first-class typed field."""

    def test_dependencies_has_memory_store_field(self):
        """dataclasses.fields(Dependencies) includes a field named 'memory_store'."""
        field_names = {f.name for f in dataclasses.fields(Dependencies)}
        assert "memory_store" in field_names, (
            "Dependencies should have a typed 'memory_store' field, "
            "not rely on extras['memory_store']. "
            f"Current fields: {field_names}"
        )

    def test_memory_store_field_defaults_to_none(self):
        """Dependencies() with no args gives deps.memory_store is None."""
        deps = Dependencies()
        assert deps.memory_store is None  # type: ignore[attr-defined]

    def test_memory_store_field_accepts_memory_store_instance(self):
        """Dependencies(memory_store=store) stores the instance on the typed field."""
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        assert deps.memory_store is store  # type: ignore[attr-defined]

    def test_memory_store_is_not_fetched_via_extras(self):
        """Typed field path: deps.memory_store is accessible without touching extras."""
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        # This should work even when extras is empty (the default)
        assert deps.extras == {}  # type: ignore[attr-defined]
        assert deps.memory_store is store  # type: ignore[attr-defined]

    def test_dependencies_remains_frozen_with_new_field(self):
        """Dependencies is still a frozen dataclass after adding memory_store field."""
        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            deps.memory_store = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Dependencies has a typed `memory_extractor` field
# ---------------------------------------------------------------------------


class TestDependenciesHasTypedMemoryExtractorField:
    """Dependencies must expose `memory_extractor` as a first-class typed field."""

    def test_dependencies_has_memory_extractor_field(self):
        """dataclasses.fields(Dependencies) includes a field named 'memory_extractor'."""
        field_names = {f.name for f in dataclasses.fields(Dependencies)}
        assert "memory_extractor" in field_names, (
            "Dependencies should have a typed 'memory_extractor' field, "
            "not rely on extras['memory_extractor']. "
            f"Current fields: {field_names}"
        )

    def test_memory_extractor_field_defaults_to_none(self):
        """Dependencies() with no args gives deps.memory_extractor is None."""
        deps = Dependencies()
        assert deps.memory_extractor is None  # type: ignore[attr-defined]

    def test_memory_extractor_field_accepts_extractor_instance(self):
        """Dependencies(memory_extractor=ext) stores the instance on the typed field."""
        ext = _StubExtractor()
        deps = Dependencies(memory_extractor=ext)  # type: ignore[call-arg]
        assert deps.memory_extractor is ext  # type: ignore[attr-defined]

    def test_both_typed_fields_coexist(self):
        """memory_store and memory_extractor can both be set simultaneously."""
        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(  # type: ignore[call-arg]
            memory_store=store,
            memory_extractor=ext,
        )
        assert deps.memory_store is store  # type: ignore[attr-defined]
        assert deps.memory_extractor is ext  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 3. MemoryExtractionTransformer.build() reads from typed fields
# ---------------------------------------------------------------------------


class TestExtractionTransformerBuildReadsTypedFields:
    """MemoryExtractionTransformer.build() must use typed deps fields, not extras."""

    def test_build_succeeds_with_typed_memory_store_field(self):
        """build() works when memory_store is on deps as a typed field (not extras)."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        # Typed field path — extras is empty
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config()

        # Must not raise — store is present on the typed field
        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_build_raises_config_error_when_typed_field_is_none(self):
        """build() raises ConfigError when deps.memory_store is None (typed field absent)."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        deps = Dependencies()  # memory_store=None (default)
        config = _make_transformer_config()

        with pytest.raises(ConfigError):
            MemoryExtractionTransformer.build(config, deps)

    def test_build_uses_typed_memory_extractor_when_set(self):
        """build() uses deps.memory_extractor typed field when present."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
        config = _make_transformer_config()

        transformer = MemoryExtractionTransformer.build(config, deps)
        # Must use the typed extractor, not fall back to the registry
        assert transformer._extractor is ext

    def test_build_does_not_require_extras_for_memory_store(self):
        """build() must NOT require memory_store to be in deps.extras.

        This is the core anti-pattern: if extras is empty but the typed field
        is set, build() should succeed.
        """
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        # Explicitly set extras={} (empty) and memory_store on typed field
        deps = Dependencies(memory_store=store, extras={})  # type: ignore[call-arg]
        config = _make_transformer_config()

        # Should succeed — memory_store is on the typed field, not in extras
        result = MemoryExtractionTransformer.build(config, deps)
        assert result is not None

    def test_build_does_not_require_extras_for_memory_extractor(self):
        """build() must NOT require memory_extractor to be in deps.extras."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        # Typed fields set, extras empty
        deps = Dependencies(memory_store=store, memory_extractor=ext, extras={})  # type: ignore[call-arg]
        config = _make_transformer_config()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._extractor is ext


# ---------------------------------------------------------------------------
# 4. MemoryResolver.build() reads from typed fields
# ---------------------------------------------------------------------------


class TestMemoryResolverBuildReadsTypedFields:
    """MemoryResolver.build() must use typed deps fields, not extras."""

    def test_build_succeeds_with_typed_memory_store_field(self):
        """build() works when memory_store is on deps as a typed field (not extras)."""
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        # Typed field — extras is empty
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config()

        result = MemoryResolver.build(config, deps)
        assert result is not None

    def test_build_raises_config_error_when_typed_field_is_none(self):
        """build() raises ConfigError when deps.memory_store typed field is None."""
        from sr2.memory.memory_resolver import MemoryResolver

        deps = Dependencies()  # memory_store=None (default)
        config = _make_resolver_config()

        with pytest.raises(ConfigError):
            MemoryResolver.build(config, deps)

    def test_build_does_not_require_extras_for_memory_store(self):
        """build() must NOT require memory_store to be in deps.extras.

        If the typed field is set and extras is empty, build() should succeed.
        """
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        # extras intentionally empty; store on typed field only
        deps = Dependencies(memory_store=store, extras={})  # type: ignore[call-arg]
        config = _make_resolver_config()

        result = MemoryResolver.build(config, deps)
        assert result is not None

    def test_resolver_uses_typed_field_store_at_resolve_time(self):
        """Resolver built from typed field holds the exact same store object.

        Tests object identity — the injected store is stored on the resolver
        instance, not a copy. End-to-end search behavior is covered separately
        by the memory_resolver test suite (InMemoryMemoryStore search semantics).
        """
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config()
        resolver = MemoryResolver.build(config, deps)

        # Identity check: same store object, not a copy
        assert resolver._store is store


# ---------------------------------------------------------------------------
# 5. SR2 accepts memory_store as a typed kwarg
# ---------------------------------------------------------------------------


class TestSR2AcceptsTypedMemoryStoreKwarg:
    """SR2.__init__ must accept memory_store as a typed kwarg (not require extras)."""

    def test_sr2_init_signature_has_memory_store_param(self):
        """SR2.__init__ must have a `memory_store` parameter in its signature."""
        from sr2.orchestrator import SR2

        sig = inspect.signature(SR2.__init__)
        assert "memory_store" in sig.parameters, (
            "SR2.__init__ should accept `memory_store` as a named parameter. "
            "Currently it requires callers to pass it via the untyped `extras` dict, "
            "which is the Service Locator antipattern described in sr2-12. "
            f"Current params: {list(sig.parameters)}"
        )

    def test_sr2_init_signature_has_memory_extractor_param(self):
        """SR2.__init__ must have a `memory_extractor` parameter in its signature."""
        from sr2.orchestrator import SR2

        sig = inspect.signature(SR2.__init__)
        assert "memory_extractor" in sig.parameters, (
            "SR2.__init__ should accept `memory_extractor` as a named parameter. "
            f"Current params: {list(sig.parameters)}"
        )

    def test_sr2_constructs_with_memory_store_kwarg(self):
        """SR2 can be constructed with memory_store= without going through extras."""
        import importlib.metadata
        from unittest.mock import MagicMock, patch

        from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig
        from sr2.pipeline.token_counting import CharacterTokenCounter
        from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
        from sr2.models import TextBlock, TokenUsage
        from sr2.orchestrator import SR2
        import sr2.orchestrator as _orch
        from collections.abc import AsyncIterator

        class _MockLLM:
            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                return CompletionResponse(
                    id="mock", content=[TextBlock(text="ok")],
                    stop_reason="end_turn", usage=TokenUsage(),
                )

            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                yield StreamEvent(type="text", text="ok")
                yield StreamEvent(type="end")

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
            # Must not raise — memory_store passed as typed kwarg, not extras
            sr2 = SR2(
                pipeline_config=config,
                llm={"default": _MockLLM()},
                token_counter=CharacterTokenCounter(),
                memory_store=store,  # typed kwarg — not extras={"memory_store": store}
            )
        assert sr2 is not None

        _orch._RESOLVERS._discovered = False
        _orch._RESOLVERS._classes = {}
        _orch._TRANSFORMERS._discovered = False
        _orch._TRANSFORMERS._classes = {}


# ---------------------------------------------------------------------------
# 6. Dependencies.extras is NOT required for memory subsystem
# ---------------------------------------------------------------------------


class TestExtrasNotRequiredForMemorySubsystem:
    """After the fix, the memory subsystem must work without touching extras at all.

    These tests document the broken contract: right now, memory_store MUST go
    through extras. After the fix, it should work via typed fields, and extras
    should be an optional escape hatch only.
    """

    def test_dependencies_with_only_typed_fields_is_sufficient_for_extraction(self):
        """Dependencies with typed memory_store (no extras) works for extraction build().

        Currently FAILS because build() checks extras, not the typed field.
        After the fix: build() should check the typed field first.
        """
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        # No extras at all — only the typed field
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config()

        # This must succeed after the fix. Before the fix, it raises ConfigError
        # because build() checks `if "memory_store" not in deps.extras`.
        result = MemoryExtractionTransformer.build(config, deps)
        assert result._store is store

    def test_dependencies_with_only_typed_fields_is_sufficient_for_resolver(self):
        """Dependencies with typed memory_store (no extras) works for resolver build().

        Currently FAILS because build() reads from extras, not the typed field.
        After the fix: build() should use the typed field.
        """
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        # No extras at all — only the typed field
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_resolver_config()

        # Must succeed after the fix.
        result = MemoryResolver.build(config, deps)
        assert result._store is store

    def test_typed_field_store_identity_preserved_through_build(self):
        """The exact store object passed to Dependencies ends up in the transformer."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        deps = Dependencies(memory_store=store)  # type: ignore[call-arg]
        config = _make_transformer_config()

        transformer = MemoryExtractionTransformer.build(config, deps)
        # Same object identity — not a copy, not a different instance
        assert transformer._store is store

    def test_typed_extractor_identity_preserved_through_build(self):
        """The exact extractor passed to Dependencies ends up in the transformer."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        ext = _StubExtractor()
        deps = Dependencies(memory_store=store, memory_extractor=ext)  # type: ignore[call-arg]
        config = _make_transformer_config()

        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._extractor is ext


# ---------------------------------------------------------------------------
# 7. Backward compatibility: extras path still works as a fallback
# ---------------------------------------------------------------------------


class TestExtrasPathBackwardCompat:
    """The extras dict path must continue to work after the typed-field fix.

    Callers passing extras={"memory_store": store} must not break silently.
    The typed field takes precedence; extras is an accepted fallback.
    This ensures existing code that uses the old API continues to function
    while the migration to typed fields is in progress.
    """

    def test_extras_path_still_works_for_extraction_transformer(self):
        """MemoryExtractionTransformer.build() still works when store is in extras."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = InMemoryMemoryStore()
        # Old-style invocation: store in extras, not on typed field
        deps = Dependencies(extras={"memory_store": store})
        config = _make_transformer_config()

        # Must not raise — backward compat
        transformer = MemoryExtractionTransformer.build(config, deps)
        assert transformer._store is store

    def test_extras_path_still_works_for_memory_resolver(self):
        """MemoryResolver.build() still works when store is in extras."""
        from sr2.memory.memory_resolver import MemoryResolver

        store = InMemoryMemoryStore()
        # Old-style invocation: store in extras, not on typed field
        deps = Dependencies(extras={"memory_store": store})
        config = _make_resolver_config()

        # Must not raise — backward compat
        resolver = MemoryResolver.build(config, deps)
        assert resolver._store is store

    def test_typed_field_takes_precedence_over_extras(self):
        """When both typed field and extras carry a store, typed field wins."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        typed_store = InMemoryMemoryStore()
        extras_store = InMemoryMemoryStore()
        deps = Dependencies(  # type: ignore[call-arg]
            memory_store=typed_store,
            extras={"memory_store": extras_store},
        )
        config = _make_transformer_config()

        transformer = MemoryExtractionTransformer.build(config, deps)
        # typed field wins
        assert transformer._store is typed_store
