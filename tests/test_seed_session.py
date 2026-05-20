"""Tests for SR2.seed_session() — pre-populate conversation history.

Covers:
  1. After seed_session(messages), turn() includes those messages in context
  2. seed_session([]) results in empty history
  3. seed_session() overwrites existing history (does not append)
  4. Input list mutation after seed_session() does not affect internal state
  5. Works when multiple layers have SessionResolver instances
  6. Works when no layers have SessionResolver (no-op, no error)
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
# Helpers
# ---------------------------------------------------------------------------


def make_message(role: str, text: str) -> Message:
    """Build a minimal Message for use as seed history."""
    return Message(role=role, content=[TextBlock(text=text)])


def make_user_message(text: str = "Hello") -> Message:
    return make_message("user", text)


def make_assistant_message(text: str = "Hi there") -> Message:
    return make_message("assistant", text)


def make_user_input(text: str = "current turn") -> list:
    return [TextBlock(text=text)]


class MockLLM:
    """Minimal LLMCallable for testing — records stream calls."""

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
    """Single conversation layer with one SessionResolver."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
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
    """Config with no SessionResolver in any layer."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
            ),
            LayerConfig(
                name="input_only",
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


def make_config_multiple_session_resolvers() -> PipelineConfig:
    """Config with SessionResolver in two separate layers."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="context_layer",
                resolvers=[
                    ResolverConfig(type="session"),
                ],
            ),
            LayerConfig(
                name="conversation_layer",
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


def _all_message_text(request: CompletionRequest) -> str:
    """Concatenate all text from all message content blocks in a request."""
    return " ".join(
        block.text
        for msg in request.messages
        for block in msg.content
        if hasattr(block, "text")
    )


def _find_session_resolvers(sr2_instance) -> list[SessionResolver]:
    """Walk engine layers and collect all SessionResolver instances."""
    results = []
    for layer in sr2_instance._engine._layers:
        for resolver in layer.resolvers:
            if isinstance(resolver, SessionResolver):
                results.append(resolver)
    return results


# ---------------------------------------------------------------------------
# 1. seed_session sets history that appears in the next turn's context
# ---------------------------------------------------------------------------


class TestSeedSessionAppearsInTurn:
    @pytest.mark.asyncio
    async def test_seeded_user_message_appears_in_turn_request(self):
        """Messages passed to seed_session() appear in the CompletionRequest
        during the subsequent turn() call."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        seed = [make_user_message("seeded prior message")]
        sr2.seed_session(seed)

        async for _ in sr2.turn(make_user_input("current input")):
            pass

        assert len(mock_llm.stream_calls) == 1
        request = mock_llm.stream_calls[0]
        all_text = _all_message_text(request)
        assert "seeded prior message" in all_text

    @pytest.mark.asyncio
    async def test_seeded_conversation_appears_in_order(self):
        """Multi-message seed appears in the request in seeded order."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        seed = [
            make_user_message("first user"),
            make_assistant_message("first assistant"),
            make_user_message("second user"),
            make_assistant_message("second assistant"),
        ]
        sr2.seed_session(seed)

        async for _ in sr2.turn(make_user_input("now what?")):
            pass

        assert len(mock_llm.stream_calls) == 1
        request = mock_llm.stream_calls[0]
        all_texts = [
            block.text
            for msg in request.messages
            for block in msg.content
            if hasattr(block, "text")
        ]
        expected_order = ["first user", "first assistant", "second user", "second assistant"]
        positions = [
            next(i for i, t in enumerate(all_texts) if expected in t)
            for expected in expected_order
        ]
        assert positions == sorted(positions), (
            f"Seeded messages not in order. Positions: {positions}"
        )

    @pytest.mark.asyncio
    async def test_seeded_messages_appear_before_current_input(self):
        """Seeded messages appear before the current turn's user input
        in the CompletionRequest messages list."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        seed = [make_user_message("history msg")]
        sr2.seed_session(seed)

        async for _ in sr2.turn(make_user_input("live input")):
            pass

        request = mock_llm.stream_calls[0]
        # Find the position of seeded text vs live input text
        all_texts = [
            block.text
            for msg in request.messages
            for block in msg.content
            if hasattr(block, "text")
        ]
        history_idx = next(
            (i for i, t in enumerate(all_texts) if "history msg" in t), None
        )
        live_idx = next(
            (i for i, t in enumerate(all_texts) if "live input" in t), None
        )
        assert history_idx is not None, "seeded message not found in request"
        assert live_idx is not None, "live input not found in request"
        assert history_idx < live_idx, "seeded history must precede current input"


    @pytest.mark.asyncio
    async def test_seeded_messages_not_duplicated_after_second_turn(self):
        """Seeded messages appear exactly once across two turns without re-seeding.

        sr2-relay calls seed_session() once when a session is created, then
        calls turn() for each subsequent request. The seeded history must
        appear in the second turn's context but not be duplicated.
        """
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        seed = [
            make_user_message("seeded prior turn"),
            make_assistant_message("seeded prior response"),
        ]
        sr2.seed_session(seed)

        # First turn
        async for _ in sr2.turn(make_user_input("turn one input")):
            pass

        # Second turn — no re-seeding
        async for _ in sr2.turn(make_user_input("turn two input")):
            pass

        assert len(mock_llm.stream_calls) == 2
        second_request = mock_llm.stream_calls[1]
        all_texts = [
            block.text
            for msg in second_request.messages
            for block in msg.content
            if hasattr(block, "text")
        ]
        seeded_occurrences = sum(1 for t in all_texts if "seeded prior turn" in t)
        assert seeded_occurrences == 1, (
            f"Seeded message appeared {seeded_occurrences} times in second turn "
            f"(expected exactly 1 — seed must not duplicate across turns)"
        )


# ---------------------------------------------------------------------------
# 2. seed_session([]) results in empty history
# ---------------------------------------------------------------------------


class TestSeedSessionEmpty:
    def test_empty_seed_sets_resolver_history_to_empty(self):
        """seed_session([]) leaves all SessionResolver._history as empty list."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        sr2.seed_session([])

        resolvers = _find_session_resolvers(sr2)
        assert len(resolvers) > 0
        for resolver in resolvers:
            assert resolver._history == []

    @pytest.mark.asyncio
    async def test_empty_seed_does_not_inject_phantom_messages(self):
        """seed_session([]) does not inject any extra messages into turn()."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        # Baseline: turn without seeding
        async for _ in sr2.turn(make_user_input("baseline")):
            pass
        baseline_request = mock_llm.stream_calls[0]
        baseline_msg_count = len(baseline_request.messages)

        # Now test with empty seed on a fresh instance
        mock_llm2 = MockLLM()
        sr2b = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm2},
            token_counter=CharacterTokenCounter(),
        )
        sr2b.seed_session([])
        async for _ in sr2b.turn(make_user_input("baseline")):
            pass
        seeded_request = mock_llm2.stream_calls[0]
        seeded_msg_count = len(seeded_request.messages)

        assert seeded_msg_count == baseline_msg_count


# ---------------------------------------------------------------------------
# 3. seed_session() overwrites existing history
# ---------------------------------------------------------------------------


class TestSeedSessionOverwrites:
    def test_second_seed_replaces_first(self):
        """Calling seed_session() twice replaces history with the second call's
        messages — does not append."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        sr2.seed_session([make_user_message("first seed")])
        sr2.seed_session([make_user_message("second seed")])

        resolvers = _find_session_resolvers(sr2)
        assert len(resolvers) > 0
        for resolver in resolvers:
            assert len(resolver._history) == 1
            assert resolver._history[0].content[0].text == "second seed"

    def test_seed_after_real_turn_overwrites_history(self):
        """seed_session() after a live turn replaces the accumulated
        conversation history, not appends to it."""
        from sr2.orchestrator import SR2

        # We can test this by inspecting _history directly after
        # manually appending to a resolver and then calling seed_session.
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        # Manually prime a resolver to simulate post-turn state
        resolvers = _find_session_resolvers(sr2)
        assert len(resolvers) > 0
        for resolver in resolvers:
            resolver._history = [
                make_user_message("pre-existing turn 1"),
                make_assistant_message("pre-existing response 1"),
            ]

        # Now seed — should overwrite
        new_seed = [make_user_message("fresh seed")]
        sr2.seed_session(new_seed)

        for resolver in resolvers:
            assert len(resolver._history) == 1
            assert resolver._history[0].content[0].text == "fresh seed"

    def test_seed_with_empty_list_clears_existing_history(self):
        """seed_session([]) clears any history that was there previously."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        resolvers = _find_session_resolvers(sr2)
        for resolver in resolvers:
            resolver._history = [make_user_message("existing history")]

        sr2.seed_session([])

        for resolver in resolvers:
            assert resolver._history == []


# ---------------------------------------------------------------------------
# 4. Copy semantics — mutating input after seed_session() has no effect
# ---------------------------------------------------------------------------


class TestSeedSessionCopySemantics:
    def test_mutating_input_list_does_not_affect_history(self):
        """Mutating the list passed to seed_session() after the call
        does not change the internal history."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        seed = [make_user_message("original message")]
        sr2.seed_session(seed)

        # Mutate the original list
        seed.append(make_user_message("injected after seed"))
        seed.clear()

        resolvers = _find_session_resolvers(sr2)
        for resolver in resolvers:
            assert len(resolver._history) == 1
            assert resolver._history[0].content[0].text == "original message"

    def test_mutating_message_object_does_not_affect_history(self):
        """The resolver stores copies of Message objects, not references.

        Verified by identity check: the objects in _history must not be
        the same objects as those passed to seed_session().
        """
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        seed = [make_user_message("msg one"), make_user_message("msg two")]
        sr2.seed_session(seed)

        resolvers = _find_session_resolvers(sr2)
        for resolver in resolvers:
            assert len(resolver._history) == 2
            assert resolver._history[0] is not seed[0], (
                "seed_session() stored a reference, not a copy of seed[0]"
            )
            assert resolver._history[1] is not seed[1], (
                "seed_session() stored a reference, not a copy of seed[1]"
            )


# ---------------------------------------------------------------------------
# 5. Multiple SessionResolver instances (multiple layers)
# ---------------------------------------------------------------------------


class TestSeedSessionMultipleResolvers:
    def test_all_session_resolvers_receive_seed(self):
        """When multiple layers each have a SessionResolver, seed_session()
        sets _history on all of them."""
        from sr2.orchestrator import SR2

        config = make_config_multiple_session_resolvers()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        seed = [
            make_user_message("shared history user"),
            make_assistant_message("shared history assistant"),
        ]
        sr2.seed_session(seed)

        resolvers = _find_session_resolvers(sr2)
        assert len(resolvers) == 2, "Expected two SessionResolvers in this config"

        for resolver in resolvers:
            assert len(resolver._history) == 2
            assert resolver._history[0].content[0].text == "shared history user"
            assert resolver._history[1].content[0].text == "shared history assistant"

    def test_multiple_resolvers_each_have_independent_copy(self):
        """Each SessionResolver gets its own copy of the seeded history.
        Mutating one resolver's _history does not affect the other's."""
        from sr2.orchestrator import SR2

        config = make_config_multiple_session_resolvers()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        seed = [make_user_message("shared")]
        sr2.seed_session(seed)

        resolvers = _find_session_resolvers(sr2)
        assert len(resolvers) == 2

        # Corrupt one resolver's history directly
        resolvers[0]._history.clear()

        # The other resolver's history must still be intact
        assert len(resolvers[1]._history) == 1
        assert resolvers[1]._history[0].content[0].text == "shared"


# ---------------------------------------------------------------------------
# 6. No SessionResolver present — no-op, no error
# ---------------------------------------------------------------------------


class TestSeedSessionNoResolver:
    def test_seed_session_with_no_session_resolver_does_not_raise(self):
        """seed_session() completes without error when no layer contains
        a SessionResolver."""
        from sr2.orchestrator import SR2

        config = make_config_no_session()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        # Must not raise
        sr2.seed_session([make_user_message("ignored")])

    def test_seed_session_empty_with_no_session_resolver_does_not_raise(self):
        """seed_session([]) with no SessionResolver present also does not raise."""
        from sr2.orchestrator import SR2

        config = make_config_no_session()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        sr2.seed_session([])

    def test_config_with_no_session_resolver_has_zero_resolvers(self):
        """Sanity check: our no-session config genuinely has no SessionResolvers."""
        from sr2.orchestrator import SR2

        config = make_config_no_session()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        resolvers = _find_session_resolvers(sr2)
        assert len(resolvers) == 0


# ---------------------------------------------------------------------------
# 7. seed_session is synchronous
# ---------------------------------------------------------------------------


class TestSeedSessionIsSynchronous:
    def test_seed_session_is_not_a_coroutine(self):
        """seed_session() must be a plain synchronous method, not a coroutine.
        Callers should not need to await it."""
        import inspect
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        result = sr2.seed_session([make_user_message("test")])

        # If seed_session were async, calling it without await would return
        # a coroutine object rather than executing. The history would be empty.
        assert not inspect.iscoroutine(result), (
            "seed_session() returned a coroutine — it must be synchronous"
        )

    def test_seed_session_returns_none(self):
        """seed_session() returns None (no return value needed)."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        result = sr2.seed_session([make_user_message("test")])
        assert result is None
