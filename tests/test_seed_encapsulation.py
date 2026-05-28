"""Tests for sr2-16: encapsulation of seed_session() — public API chain.

These tests assert the desired public API exists at each level:
    SR2.seed_session()          → no private accessors
    PipelineEngine.seed()       → delegates to Layer.seed()
    Layer.seed()                → delegates to resolver.seed()
    SessionResolver.seed()      → sets history without _history mutation
    Resolver protocol           → seed() is a valid no-op method

All tests FAIL until the implementation adds:
    - SessionResolver.seed(messages)
    - Layer.seed(messages)
    - PipelineEngine.seed(messages)
    - SR2.seed_session() reimplemented via engine.seed()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import Message, TextBlock, TokenUsage
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    StreamEvent,
)


# ---------------------------------------------------------------------------
# Helpers shared with test_seed_session.py — duplicated here to keep this
# file self-contained and independent of private-accessor helpers.
# ---------------------------------------------------------------------------


def make_message(role: str, text: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])


def make_user_message(text: str = "Hello") -> Message:
    return make_message("user", text)


def make_assistant_message(text: str = "Hi there") -> Message:
    return make_message("assistant", text)


def make_user_input(text: str = "current turn") -> list:
    return [TextBlock(text=text)]


class MockLLM:
    def __init__(self) -> None:
        self.stream_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            id="mock-resp",
            content=[TextBlock(text="mock response")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        yield StreamEvent(type="text", text="mock response")
        yield StreamEvent(type="end")


def make_minimal_config() -> PipelineConfig:
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
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ]
    )


def make_config_no_session() -> PipelineConfig:
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
            ),
            LayerConfig(
                name="input_only",
                target="messages",
                resolvers=[
                    ResolverConfig(
                        type="input",
                        subscriptions=[
                            EventSubscriptionConfig(event="user_input", phase="completed")
                        ],
                    ),
                ],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 1. SessionResolver.seed() public method exists and works
# ---------------------------------------------------------------------------


class TestSessionResolverSeedMethod:
    def test_session_resolver_has_seed_method(self):
        """SessionResolver must expose a public seed() method.

        Fails until SessionResolver.seed() is added.
        """
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        assert hasattr(resolver, "seed"), (
            "SessionResolver has no 'seed' method — add seed(messages: list[Message]) -> None"
        )
        assert callable(resolver.seed)

    def test_session_resolver_seed_sets_history_via_public_method(self):
        """SessionResolver.seed() sets history without touching _history directly.

        The test asserts the *effect* (history appears in resolve output),
        not the internal attribute. Fails until seed() is implemented.
        """
        from sr2.pipeline.events import Event

        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        messages = [make_user_message("seeded"), make_assistant_message("seeded reply")]
        resolver.seed(messages)

        result = asyncio.run(resolver.resolve([]))
        resolved_texts = [
            block.text
            for msg in result.content
            for block in msg.content
            if hasattr(block, "text")
        ]
        assert "seeded" in resolved_texts, (
            "seed() did not inject messages into resolve() output"
        )
        assert "seeded reply" in resolved_texts

    def test_session_resolver_seed_stores_independent_copies(self):
        """SessionResolver.seed() must copy messages, not store references.

        Mutating the list after seed() must not change what resolve() returns.
        """
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        messages = [make_user_message("original")]
        resolver.seed(messages)
        messages.clear()

        result = asyncio.run(resolver.resolve([]))
        texts = [
            block.text
            for msg in result.content
            for block in msg.content
            if hasattr(block, "text")
        ]
        assert "original" in texts, "seed() stored a reference; clearing input erased seeded history"

    def test_session_resolver_seed_overwrites_existing_history(self):
        """seed() replaces any prior history — does not append.

        Fails until seed() is implemented (or until _history is replaced
        through the public method rather than appended).
        """
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        resolver.seed([make_user_message("first"), make_assistant_message("first reply")])
        resolver.seed([make_user_message("second seed")])

        result = asyncio.run(resolver.resolve([]))
        texts = [
            block.text
            for msg in result.content
            for block in msg.content
            if hasattr(block, "text")
        ]
        assert len(result.content) == 1, (
            f"seed() appended instead of replacing — got {len(result.content)} messages"
        )
        assert "second seed" in texts

    def test_session_resolver_seed_with_empty_list_clears_history(self):
        """seed([]) after a non-empty seed clears all history."""
        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        resolver.seed([make_user_message("something")])
        resolver.seed([])

        result = asyncio.run(resolver.resolve([]))
        assert result.content == [], (
            "seed([]) did not clear history"
        )


# ---------------------------------------------------------------------------
# 2. Layer.seed() public method exists and delegates
# ---------------------------------------------------------------------------


class TestLayerSeedMethod:
    def _make_layer_with_session_resolver(self):
        """Build a Layer that has a SessionResolver, using the public SR2 factory."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        # Access via the public engine.seed path once it exists, but for now
        # grab the layer to test it independently.
        # We test via Layer directly — build one ourselves.
        return sr2  # returned so tests can call engine.seed()

    def test_layer_has_seed_method(self):
        """Layer must expose a public seed() method.

        Fails until Layer.seed() is added.
        """
        from sr2.pipeline.layer import Layer

        assert hasattr(Layer, "seed"), (
            "Layer class has no 'seed' method — add seed(messages: list[Message]) -> None"
        )

    def test_layer_seed_propagates_to_session_resolver(self):
        """Layer.seed(messages) must call seed() on any SessionResolver it holds.

        Builds a Layer directly using the real resolver factory so the test
        exercises the concrete implementation path.
        """
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.models import CompilationTarget
        from sr2.pipeline.event_bus import EventBus

        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        layer = Layer(
            name="conv",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        messages = [make_user_message("layer seeded")]
        layer.seed(messages)

        result = asyncio.run(resolver.resolve([]))
        texts = [
            block.text
            for msg in result.content
            for block in msg.content
            if hasattr(block, "text")
        ]
        assert "layer seeded" in texts, (
            "Layer.seed() did not propagate to SessionResolver.seed()"
        )

    def test_layer_seed_noop_for_non_session_resolvers(self):
        """Layer.seed() must not raise when no resolver is a SessionResolver.

        Other resolvers have no seed() method — Layer must handle this gracefully.
        Fails if Layer.seed() calls .seed() unconditionally without checking.
        """
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.models import CompilationTarget
        from sr2.pipeline.event_bus import EventBus
        from sr2.pipeline.resolvers.static import StaticResolver

        static_config = ResolverConfig(type="static", config={"text": "hello"})
        resolver = StaticResolver(static_config)

        layer = Layer(
            name="sys",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            token_counter=CharacterTokenCounter(),
            event_bus=EventBus(),
        )

        # Must not raise — no SessionResolver in this layer
        layer.seed([make_user_message("ignored")])


# ---------------------------------------------------------------------------
# 3. PipelineEngine.seed() public method exists and delegates to layers
# ---------------------------------------------------------------------------


class TestPipelineEngineSeedMethod:
    def test_engine_has_seed_method(self):
        """PipelineEngine must expose a public seed() method.

        Fails until PipelineEngine.seed() is added.
        """
        from sr2.pipeline.engine import PipelineEngine

        assert hasattr(PipelineEngine, "seed"), (
            "PipelineEngine class has no 'seed' method — add seed(messages: list[Message]) -> None"
        )

    def test_engine_seed_propagates_through_layers(self):
        """PipelineEngine.seed(messages) must call seed() on all layers.

        Accesses _engine via SR2 (unavoidable at this level for construction),
        then calls the public engine.seed() method — not _layers directly.
        This one-level access to _engine is intentional, not an oversight.
        """
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Call engine.seed() directly (the new public method being required)
        messages = [make_user_message("engine seed"), make_assistant_message("engine reply")]
        sr2._engine.seed(messages)  # _engine access is one level, not three

        # Verify via behavior: run a turn and check the messages appear
        asyncio.run(_exhaust_turn(sr2, make_user_input("live input")))

        assert len(mock_llm.stream_calls) == 1
        request = mock_llm.stream_calls[0]
        all_text = _all_message_text(request)
        assert "engine seed" in all_text, "engine.seed() did not reach SessionResolver"


# ---------------------------------------------------------------------------
# 4. SR2.seed_session() uses engine.seed() — no private traversal
# ---------------------------------------------------------------------------


class TestSR2SeedSessionNoPrivateAccess:
    def test_seed_session_does_not_require_private_layers_attribute(self):
        """SR2.seed_session() must NOT access engine._layers directly.

        This test patches _engine._layers to raise AttributeError, then
        confirms seed_session() still works via the public engine.seed() path.

        Fails under the current implementation (which uses _engine._layers).
        Passes once seed_session() delegates to engine.seed().
        """
        from sr2.orchestrator import SR2
        from unittest.mock import patch, MagicMock

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        # Block access to _layers — the private attribute
        original_layers = sr2._engine._layers
        del sr2._engine._layers

        try:
            # Should work via engine.seed() even with _layers removed
            sr2.seed_session([make_user_message("public path")])
        except AttributeError as exc:
            raise AssertionError(
                "seed_session() still accesses engine._layers directly; "
                f"it must delegate to engine.seed() instead. Original error: {exc}"
            ) from exc
        finally:
            sr2._engine._layers = original_layers

    def test_seed_session_does_not_directly_mutate_resolver_history(self):
        """SR2.seed_session() must route through the public seed() API.

        Verifies that seed_session() calls resolver.seed() rather than
        directly assigning resolver._history, by spying on resolver.seed()
        via monkey-patching. If seed_session() never calls seed(), the spy
        records nothing and the assertion fails.

        Note: one level of _engine access is acceptable in test setup.
        """
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        # Collect session resolvers — one level of private access is acceptable for setup.
        resolvers: list[SessionResolver] = []
        for layer in sr2._engine._layers:
            for r in layer.resolvers:
                if isinstance(r, SessionResolver):
                    resolvers.append(r)

        assert resolvers, "No SessionResolver found — config may be wrong"

        # Spy on resolver.seed() to confirm it gets called.
        seed_called = []
        original_seed = getattr(resolvers[0], "seed", None)

        if original_seed is None:
            pytest.fail(
                "SessionResolver.seed() does not exist — "
                "seed_session() cannot delegate to it"
            )

        def spy_seed(messages):
            seed_called.append(messages)
            original_seed(messages)

        resolvers[0].seed = spy_seed  # monkey-patch the instance

        sr2.seed_session([make_user_message("via public api")])

        assert seed_called, (
            "SR2.seed_session() did not call resolver.seed() — "
            "it must delegate through engine.seed() → layer.seed() → resolver.seed()"
        )

    @pytest.mark.asyncio
    async def test_seed_session_behavior_preserved_after_refactor(self):
        """Seeded messages still appear in turn context after the encapsulation fix.

        Regression test: the refactor must not break seed_session() behavior.
        """
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        seed = [make_user_message("refactored seed")]
        sr2.seed_session(seed)

        async for _ in sr2.turn(make_user_input("live input")):
            pass

        assert len(mock_llm.stream_calls) == 1
        all_text = _all_message_text(mock_llm.stream_calls[0])
        assert "refactored seed" in all_text


# ---------------------------------------------------------------------------
# 5. Resolver protocol compatibility — seed() is a valid no-op on others
# ---------------------------------------------------------------------------


class TestResolverSeedProtocolCompatibility:
    def test_non_session_resolvers_can_have_noop_seed(self):
        """Resolvers that aren't SessionResolver must not crash when seed() is
        called on them.

        Layer.seed() will call seed() on all resolvers that expose it.
        Resolvers that don't need it should have a no-op default.
        This test confirms a StaticResolver (or any non-session resolver)
        survives seed() being called without raising AttributeError.
        """
        from sr2.pipeline.resolvers.static import StaticResolver

        config = ResolverConfig(type="static", config={"text": "hello"})
        resolver = StaticResolver(config)

        # If the Resolver protocol adds seed() with a default no-op, this works.
        # If it doesn't, Layer.seed() must guard with hasattr(resolver, 'seed').
        # Either way, the call must not raise.
        try:
            if hasattr(resolver, "seed"):
                resolver.seed([make_user_message("noop")])
        except Exception as exc:
            pytest.fail(
                f"Calling seed() on StaticResolver raised {type(exc).__name__}: {exc}"
            )

    def test_session_resolver_satisfies_resolver_protocol(self):
        """SessionResolver with seed() still satisfies the Resolver protocol.

        Adding seed() must not break isinstance(resolver, Resolver).
        """
        from sr2.pipeline.protocols import Resolver

        config = ResolverConfig(type="session")
        resolver = SessionResolver(config)

        assert isinstance(resolver, Resolver), (
            "SessionResolver no longer satisfies Resolver protocol after adding seed()"
        )


# ---------------------------------------------------------------------------
# 6. turn() execution_count reset: same private-traversal issue
# ---------------------------------------------------------------------------


class TestTurnExecutionCountReset:
    def test_turn_has_public_reset_path_without_engine_layers(self):
        """SR2.turn() resets execution_count by walking engine._layers directly.

        This is the same pattern as seed_session(). After the fix, turn()
        should delegate to a public method (e.g. engine.reset_execution_counts()).

        This test confirms PipelineEngine has a public reset method for counts.
        Fails until PipelineEngine.reset_execution_counts() (or equivalent) is added.
        Note: this test only checks presence; see test below for behavioral check.
        """
        from sr2.pipeline.engine import PipelineEngine

        # Either reset_execution_counts() or a more general prepare_turn() etc.
        has_public_reset = any(
            hasattr(PipelineEngine, name)
            for name in ("reset_execution_counts", "prepare_turn", "reset_for_turn")
        )
        assert has_public_reset, (
            "PipelineEngine has no public method for resetting execution counts "
            "(reset_execution_counts / prepare_turn / reset_for_turn). "
            "SR2.turn() currently traverses _engine._layers to do this — "
            "add a public delegation method."
        )

    def test_reset_execution_counts_actually_resets(self):
        """PipelineEngine.reset_execution_counts() must zero all resolver execution counts.

        Behavioral companion to the presence check above. Runs a turn to
        increment execution counts, then calls the reset method and verifies
        all resolver execution_count values are zero.
        """
        from sr2.pipeline.engine import PipelineEngine
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Run a turn to increment execution counts
        asyncio.run(_exhaust_turn(sr2, make_user_input("trigger")))

        # At least some resolvers should have non-zero execution_count after a turn
        engine: PipelineEngine = sr2._engine
        all_before = [
            r.execution_count
            for layer in engine._layers
            for r in layer.resolvers
        ]
        assert any(c > 0 for c in all_before), (
            "No resolver incremented execution_count — test setup is wrong"
        )

        # Call whichever public reset method exists
        if hasattr(engine, "reset_execution_counts"):
            engine.reset_execution_counts()
        elif hasattr(engine, "prepare_turn"):
            engine.prepare_turn()
        elif hasattr(engine, "reset_for_turn"):
            engine.reset_for_turn()
        else:
            pytest.fail("No public reset method found on PipelineEngine")

        all_after = [
            r.execution_count
            for layer in engine._layers
            for r in layer.resolvers
        ]
        assert all(c == 0 for c in all_after), (
            f"reset method did not zero all execution counts; got {all_after}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _all_message_text(request: CompletionRequest) -> str:
    return " ".join(
        block.text
        for msg in request.messages
        for block in msg.content
        if hasattr(block, "text")
    )


async def _exhaust_turn(sr2, user_input: list) -> None:
    async for _ in sr2.turn(user_input):
        pass
