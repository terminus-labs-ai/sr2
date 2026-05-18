"""Tests for sr2.orchestrator.SR2 — the top-level orchestrator.

Covers:
  - Construction (valid config, missing "default" key)
  - turn() as async iterator yielding StreamEvents
  - turn() calls PipelineEngine.run() with user_input
  - turn() calls llm["default"].stream() with the compiled request
  - turn() emits assistant_response on the engine bus after stream ends
  - turn() kicks off post_process() fire-and-forget
  - multi-turn: resolvers reset between turns
  - post_process() is a no-op
  - LLM errors during stream propagate
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    StreamEvent,
)

# ---------------------------------------------------------------------------
# Import under test — expected to fail with ImportError until implementation
# ---------------------------------------------------------------------------

# We import lazily inside each test class so that a single ImportError
# doesn't prevent all tests from being collected.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_user_input(text: str = "Hello") -> list:
    """Return a minimal list[ContentBlock] representing user input."""
    return [TextBlock(text=text)]


def make_completion_response(text: str = "I am the assistant.") -> CompletionResponse:
    return CompletionResponse(
        id="test-resp-001",
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


class MockLLM:
    """Minimal LLMCallable implementation for testing."""

    def __init__(self, events: list[StreamEvent] | None = None):
        self._events: list[StreamEvent] = events or [
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="world"),
            StreamEvent(type="end"),
        ]
        self.stream_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return make_completion_response()

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        for event in self._events:
            yield event


class ErrorLLM:
    """LLMCallable that raises during stream."""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return make_completion_response()

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="text", text="partial")
        raise RuntimeError("LLM backend error")


def make_minimal_config() -> PipelineConfig:
    """Minimal two-layer PipelineConfig sufficient for testing."""
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                        # turn_start subscription is default for StaticResolver
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
                resolvers=[
                    ResolverConfig(
                        type="session",
                        # default subscriptions: user_input + assistant_response
                    ),
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
# 1. Construction
# ---------------------------------------------------------------------------


class TestSR2Construction:
    def test_constructs_with_valid_args(self):
        """SR2 constructs without error given valid config, llm dict, token_counter."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        llm = {"default": MockLLM()}
        counter = CharacterTokenCounter()

        sr2 = SR2(pipeline_config=config, llm=llm, token_counter=counter)

        assert sr2 is not None

    def test_raises_if_default_key_missing(self):
        """SR2 raises ValueError when 'default' key is absent from llm dict."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        llm = {"other": MockLLM()}
        counter = CharacterTokenCounter()

        with pytest.raises(ValueError, match="default"):
            SR2(pipeline_config=config, llm=llm, token_counter=counter)

    def test_raises_if_llm_dict_empty(self):
        """SR2 raises ValueError when llm dict is empty (no 'default' key)."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        counter = CharacterTokenCounter()

        with pytest.raises(ValueError):
            SR2(pipeline_config=config, llm={}, token_counter=counter)


# ---------------------------------------------------------------------------
# 2. turn() — async iterator contract
# ---------------------------------------------------------------------------


class TestTurnAsyncIterator:
    @pytest.mark.asyncio
    async def test_turn_is_async_iterator(self):
        """turn() returns an object usable with 'async for'."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": MockLLM()}, token_counter=CharacterTokenCounter())
        user_input = make_user_input()

        result = sr2.turn(user_input)
        # Must have __aiter__
        assert hasattr(result, "__aiter__")

    @pytest.mark.asyncio
    async def test_turn_yields_stream_events(self):
        """turn() yields StreamEvent objects."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": MockLLM()}, token_counter=CharacterTokenCounter())
        user_input = make_user_input()

        events = [e async for e in sr2.turn(user_input)]

        assert len(events) > 0
        for event in events:
            assert isinstance(event, StreamEvent)

    @pytest.mark.asyncio
    async def test_turn_yields_text_events_from_llm(self):
        """turn() yields text StreamEvents matching what the LLM stream emits."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="foo"),
            StreamEvent(type="text", text="bar"),
            StreamEvent(type="end"),
        ])
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())
        user_input = make_user_input()

        events = [e async for e in sr2.turn(user_input)]
        text_events = [e for e in events if e.type == "text"]

        assert len(text_events) == 2
        assert text_events[0].text == "foo"
        assert text_events[1].text == "bar"

    @pytest.mark.asyncio
    async def test_turn_yields_end_event(self):
        """turn() yields an 'end' StreamEvent as its final item."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="response"),
            StreamEvent(type="end"),
        ])
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())
        user_input = make_user_input()

        events = [e async for e in sr2.turn(user_input)]
        end_events = [e for e in events if e.type == "end"]

        assert len(end_events) >= 1


# ---------------------------------------------------------------------------
# 3. turn() — engine and LLM integration
# ---------------------------------------------------------------------------


class TestTurnEngineIntegration:
    @pytest.mark.asyncio
    async def test_turn_calls_pipeline_engine_run(self):
        """turn() calls PipelineEngine.run() with the provided user_input."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        user_input = make_user_input("test message")

        # Patch the engine's run method after construction
        original_run = sr2._engine.run
        captured_inputs = []

        async def capturing_run(ui):
            captured_inputs.append(ui)
            return await original_run(ui)

        sr2._engine.run = capturing_run

        async for _ in sr2.turn(user_input):
            pass

        assert len(captured_inputs) == 1
        assert captured_inputs[0] == user_input

    @pytest.mark.asyncio
    async def test_turn_calls_llm_stream_with_compiled_request(self):
        """turn() calls llm['default'].stream() with the CompletionRequest from PipelineEngine."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        user_input = make_user_input("stream test")

        async for _ in sr2.turn(user_input):
            pass

        assert len(mock_llm.stream_calls) == 1
        assert isinstance(mock_llm.stream_calls[0], CompletionRequest)


# ---------------------------------------------------------------------------
# 4. turn() — assistant_response event
# ---------------------------------------------------------------------------


class TestTurnAssistantResponseEvent:
    @pytest.mark.asyncio
    async def test_assistant_response_event_emitted_after_stream(self):
        """After stream completes, an 'assistant_response' event is queued on the bus."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="Hello"),
            StreamEvent(type="end"),
        ])
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        emitted_events: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: emitted_events.append(e))

        async for _ in sr2.turn(make_user_input()):
            pass

        # Allow any async tasks to settle
        await asyncio.sleep(0)

        assert any(e.name == "assistant_response" for e in emitted_events)

    @pytest.mark.asyncio
    async def test_assistant_response_event_data_is_completion_response(self):
        """The assistant_response event carries a CompletionResponse as its data."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="My response"),
            StreamEvent(type="end"),
        ])
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        captured: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: captured.append(e))

        async for _ in sr2.turn(make_user_input()):
            pass

        await asyncio.sleep(0)

        assert len(captured) >= 1
        response_data = captured[0].data
        assert isinstance(response_data, CompletionResponse)

    @pytest.mark.asyncio
    async def test_assistant_response_contains_accumulated_text(self):
        """CompletionResponse built from stream contains all streamed text chunks."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="world"),
            StreamEvent(type="end"),
        ])
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        captured: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: captured.append(e))

        async for _ in sr2.turn(make_user_input()):
            pass

        await asyncio.sleep(0)

        assert len(captured) >= 1
        response: CompletionResponse = captured[0].data
        # The full text should be the concatenation of streamed chunks
        full_text = "".join(
            block.text
            for block in response.content
            if hasattr(block, "text")
        )
        assert "Hello " in full_text
        assert "world" in full_text


# ---------------------------------------------------------------------------
# 5. turn() — multi-turn (resolver reset)
# ---------------------------------------------------------------------------


class TestTurnMultiTurn:
    @pytest.mark.asyncio
    async def test_second_turn_works(self):
        """Second call to turn() completes without error."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        async for _ in sr2.turn(make_user_input("first")):
            pass

        # Second turn must not raise
        second_events = [e async for e in sr2.turn(make_user_input("second"))]
        assert len(second_events) > 0

    @pytest.mark.asyncio
    async def test_second_turn_calls_llm_again(self):
        """LLM stream is called once per turn, not only on the first turn."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        async for _ in sr2.turn(make_user_input("first")):
            pass
        async for _ in sr2.turn(make_user_input("second")):
            pass

        assert len(mock_llm.stream_calls) == 2

    @pytest.mark.asyncio
    async def test_resolvers_produce_output_on_second_turn(self):
        """Resolvers re-fire on turn 2, producing the second turn's user input.

        This specifically tests the execution_count reset fix: if resolvers
        are not reset between turns, they silently skip on turn 2 and the
        second request would be missing the user message.
        """
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        async for _ in sr2.turn(make_user_input("first message")):
            pass
        async for _ in sr2.turn(make_user_input("second message")):
            pass

        assert len(mock_llm.stream_calls) == 2
        second_request = mock_llm.stream_calls[1]
        assert isinstance(second_request, CompletionRequest)

        # The second request must contain the second turn's user input text.
        # If resolvers weren't reset, this would be empty or contain "first message".
        all_message_text = " ".join(
            block.text
            for msg in second_request.messages
            for block in msg.content
            if hasattr(block, "text")
        )
        assert "second message" in all_message_text


# ---------------------------------------------------------------------------
# 6. post_process()
# ---------------------------------------------------------------------------


class TestPostProcess:
    @pytest.mark.asyncio
    async def test_post_process_called_by_turn(self):
        """turn() must invoke post_process() as part of its lifecycle.

        If the implementer forgets the fire-and-forget call, this test catches it.
        """
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": MockLLM()}, token_counter=CharacterTokenCounter())

        calls: list[CompletionResponse] = []
        original = sr2.post_process

        async def spy(response: CompletionResponse) -> None:
            calls.append(response)
            return await original(response)

        sr2.post_process = spy

        async for _ in sr2.turn(make_user_input()):
            pass

        # Allow fire-and-forget task to complete
        await asyncio.sleep(0)

        assert len(calls) == 1
        assert isinstance(calls[0], CompletionResponse)

    @pytest.mark.asyncio
    async def test_post_process_is_noop(self):
        """post_process() returns None without raising."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": MockLLM()}, token_counter=CharacterTokenCounter())
        response = make_completion_response()

        result = await sr2.post_process(response)

        assert result is None

    @pytest.mark.asyncio
    async def test_post_process_accepts_completion_response(self):
        """post_process() signature accepts a CompletionResponse without type error."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": MockLLM()}, token_counter=CharacterTokenCounter())

        # Should not raise regardless of response content
        await sr2.post_process(make_completion_response("Some long response text."))


# ---------------------------------------------------------------------------
# 7. Error handling
# ---------------------------------------------------------------------------


class TestTurnErrorHandling:
    @pytest.mark.asyncio
    async def test_llm_error_during_stream_propagates(self):
        """Errors raised by the LLM during streaming are not silently swallowed."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": ErrorLLM()}, token_counter=CharacterTokenCounter())

        with pytest.raises(RuntimeError, match="LLM backend error"):
            async for _ in sr2.turn(make_user_input()):
                pass
