"""Tests for Seedable protocol — replace isinstance(SessionResolver) with protocol check.

Covers:
  1. Seedable protocol is runtime_checkable
  2. SessionResolver satisfies Seedable
  3. StaticResolver does not satisfy Seedable (no seed method)
  4. Custom resolver with seed() method satisfies Seedable (OCP)
  5. Layer.seed() calls seed() on Seedable resolvers, skips non-Seedable
  6. Layer.seed() does not import SessionResolver (verify no hardcoded dependency)
"""

from __future__ import annotations

from typing import Any

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import Message, TextBlock
from sr2.pipeline.protocols import Seedable
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.pipeline.resolvers.static import StaticResolver


def make_message(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


# ---------------------------------------------------------------------------
# 1. Seedable protocol basics
# ---------------------------------------------------------------------------


class TestSeedableProtocol:
    def test_seedable_is_runtime_checkable(self):
        """Seedable must be @runtime_checkable so isinstance() works."""
        from typing import runtime_checkable

        # This will fail if Seedable isn't decorated properly
        try:
            isinstance(object(), Seedable)
        except TypeError:
            pytest.fail(
                "Seedable is not runtime_checkable — isinstance() raised TypeError"
            )

    def test_session_resolver_satisfies_seedable(self):
        """SessionResolver must satisfy the Seedable protocol."""
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        assert isinstance(resolver, Seedable), (
            "SessionResolver does not satisfy Seedable protocol"
        )

    def test_static_resolver_does_not_satisfy_seedable(self):
        """StaticResolver does not have seed() — should NOT satisfy Seedable."""
        config = ResolverConfig(type="static", config={"text": "hello"})
        resolver = StaticResolver(config)

        assert not isinstance(resolver, Seedable), (
            "StaticResolver unexpectedly satisfies Seedable — it has no seed() method"
        )

    def test_custom_class_with_seed_satisfies_seedable(self):
        """Any class with a seed(messages: list[Message]) method satisfies Seedable,
        even if it doesn't inherit from SessionResolver (OCP compliance)."""

        class CustomSeedableResolver:
            name: str = "custom"
            max_executions: int = 1
            execution_count: int = 0
            _seeded: list[Message] = []

            def seed(self, messages: list[Message]) -> None:
                self._seeded = messages

        resolver = CustomSeedableResolver()

        assert isinstance(resolver, Seedable), (
            "Custom class with seed() method does not satisfy Seedable protocol"
        )

    def test_class_without_seed_does_not_satisfy_seedable(self):
        """A class with no seed() method should not satisfy Seedable."""

        class NoSeedResolver:
            name: str = "no_seed"
            max_executions: int = 1
            execution_count: int = 0

        resolver = NoSeedResolver()

        assert not isinstance(resolver, Seedable), (
            "Class without seed() unexpectedly satisfies Seedable"
        )


# ---------------------------------------------------------------------------
# 2. Layer.seed() uses protocol, not hardcoded isinstance
# ---------------------------------------------------------------------------


class TestLayerSeedUsesProtocol:
    def _make_layer(self, resolvers):
        """Build a minimal Layer for testing."""
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.models import CompilationTarget
        from sr2.pipeline.event_bus import EventBus
        from sr2.pipeline.token_counting import CharacterTokenCounter

        return Layer(
            name="test",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=resolvers,
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

    def test_layer_seed_calls_seed_on_custom_seedable_resolver(self):
        """Layer.seed() must call seed() on any Seedable resolver,
        not just SessionResolver. This proves the OCP: adding a new
        seedable resolver type works without editing Layer.seed()."""

        class CustomSeedableResolver:
            name: str = "custom"
            max_executions: int = 1
            execution_count: int = 0
            subscriptions: list = []
            _seeded: list[Message] | None = None

            def seed(self, messages: list[Message]) -> None:
                self._seeded = list(messages)

        resolver = CustomSeedableResolver()
        layer = self._make_layer([resolver])

        messages = [make_message("user", "seeded")]
        layer.seed(messages)

        assert resolver._seeded is not None, (
            "Layer.seed() did not call seed() on custom Seedable resolver"
        )
        assert len(resolver._seeded) == 1
        assert resolver._seeded[0].content[0].text == "seeded"

    def test_layer_seed_skips_non_seedable_resolver(self):
        """Layer.seed() silently skips resolvers that don't satisfy Seedable."""
        config = ResolverConfig(type="static", config={"text": "system prompt"})
        resolver = StaticResolver(config)
        layer = self._make_layer([resolver])

        # Must not raise — StaticResolver has no seed() method
        layer.seed([make_message("user", "should be ignored")])

    def test_layer_seed_mixed_resolvers(self):
        """Layer.seed() with mixed Seedable and non-Seedable resolvers:
        only Seedable ones receive the call."""
        session_config = ResolverConfig(type="session")
        session_resolver = SessionResolver(session_config)

        static_config = ResolverConfig(type="static", config={"text": "static"})
        static_resolver = StaticResolver(static_config)

        layer = self._make_layer([session_resolver, static_resolver])

        messages = [make_message("user", "both resolvers")]
        layer.seed(messages)

        # SessionResolver should have received the seed
        assert len(session_resolver._history) == 1
        assert session_resolver._history[0].content[0].text == "both resolvers"

        # StaticResolver has no seed — verify it wasn't affected
        assert not hasattr(static_resolver, "_seeded") or static_resolver._seeded is None

    def test_layer_seed_no_hardcoded_session_resolver_import(self):
        """Layer.seed() must not import SessionResolver internally.

        This test verifies the method body doesn't contain the hardcoded
        import that was the original problem (sr2-76)."""
        import inspect
        from sr2.pipeline.layer import Layer

        source = inspect.getsource(Layer.seed)
        assert "SessionResolver" not in source, (
            "Layer.seed() still imports SessionResolver — should use Seedable protocol instead"
        )
        assert "Seedable" in source, (
            "Layer.seed() should reference Seedable protocol"
        )


# ---------------------------------------------------------------------------
# 3. Protocol method signature verification
# ---------------------------------------------------------------------------


class TestSeedableMethodSignature:
    def test_seed_method_accepts_list_of_messages(self):
        """Seedable.seed() must accept a list[Message] argument."""
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        messages = [
            make_message("user", "hello"),
            make_message("assistant", "hi there"),
        ]
        # Should not raise
        resolver.seed(messages)

    def test_seed_method_returns_none(self):
        """Seedable.seed() must return None (side-effect only)."""
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        result = resolver.seed([make_message("user", "test")])
        assert result is None, (
            f"Seedable.seed() returned {result!r} — should return None"
        )

    def test_seed_method_is_not_async(self):
        """Seedable.seed() must be synchronous."""
        import inspect
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        assert not inspect.iscoroutinefunction(resolver.seed), (
            "Seedable.seed() must be a sync method"
        )
