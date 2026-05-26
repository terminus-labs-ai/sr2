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
