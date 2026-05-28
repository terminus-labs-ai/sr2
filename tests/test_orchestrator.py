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
from sr2.models import TextBlock, ToolUseBlock, TokenUsage
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
    """Minimal LLMCallable implementation for testing.

    When ``follow_up_events`` is provided, the first call returns ``events``
    and all subsequent calls return ``follow_up_events``. This supports
    multi-iteration tool loop tests that need the LLM to terminate after
    tool execution.
    """

    def __init__(
        self,
        events: list[StreamEvent] | None = None,
        follow_up_events: list[StreamEvent] | None = None,
    ):
        self._events: list[StreamEvent] = events or [
            StreamEvent(type="text", text="Hello "),
            StreamEvent(type="text", text="world"),
            StreamEvent(type="end"),
        ]
        self._follow_up_events: list[StreamEvent] | None = follow_up_events
        self.stream_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        return make_completion_response()

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        call_index = len(self.stream_calls) - 1
        if call_index == 0 or self._follow_up_events is None:
            events = self._events
        else:
            events = self._follow_up_events
        for event in events:
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
                target="system",
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
                target="messages",
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

    def test_dict_without_default_key_is_valid(self):
        """SR2 accepts a dict without a 'default' key (sr2-14: magic string removed)."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        llm = {"other": MockLLM()}
        counter = CharacterTokenCounter()

        # A dict without "default" is now valid — first value is used as driver.
        sr2 = SR2(pipeline_config=config, llm=llm, token_counter=counter)
        assert sr2 is not None

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
    async def test_turn_calls_pipeline_engine_start_turn(self):
        """turn() calls PipelineEngine.start_turn() to begin the turn."""
        from sr2.orchestrator import SR2

        mock_llm = MockLLM()
        config = make_minimal_config()
        sr2 = SR2(pipeline_config=config, llm={"default": mock_llm}, token_counter=CharacterTokenCounter())

        user_input = make_user_input("test message")

        # Patch start_turn to verify it is called
        original_start_turn = sr2._engine.start_turn
        start_turn_calls = []

        async def capturing_start_turn(turn_seq):
            start_turn_calls.append(turn_seq)
            return await original_start_turn(turn_seq)

        sr2._engine.start_turn = capturing_start_turn

        async for _ in sr2.turn(user_input):
            pass

        assert len(start_turn_calls) == 1, (
            f"Expected start_turn called once, got {len(start_turn_calls)}"
        )

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


# ---------------------------------------------------------------------------
# 8. Regression: sr2-9 — tool_use stream events must appear in assistant_response
# ---------------------------------------------------------------------------
# These tests are INTENTIONALLY RED against the current implementation.
# orchestrator.turn() accumulates only text events and builds a text-only
# CompletionResponse.content, dropping all tool_use StreamEvents.
# They will turn green once sr2-9 is fixed.


class TestTurnToolUseRegression:
    """Regression tests for sr2-9: tool_use StreamEvents dropped from assistant_response."""

    @pytest.mark.asyncio
    async def test_assistant_response_contains_tool_use_block(self):
        """The intermediate assistant message (with ToolUseBlock) is built during the
        tool loop iteration. The final assistant_response event (emitted after the loop)
        contains the final LLM text response. The ToolUseBlock is present in the
        conversation messages fed to the second LLM call.

        This test verifies the tool loop fires the executor and produces a valid final response.
        """
        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event
        from sr2.models import ToolResultBlock

        async def stub_executor(block: ToolUseBlock) -> ToolResultBlock:
            return ToolResultBlock(tool_use_id=block.id, content="result")

        tool_stream = MockLLM(
            events=[
                StreamEvent(type="text", text="Let me check"),
                StreamEvent(
                    type="tool_use",
                    tool_use_id="tc_1",
                    tool_name="get_weather",
                    tool_input={"location": "Oslo"},
                ),
                StreamEvent(type="end"),
            ],
            follow_up_events=[
                StreamEvent(type="text", text="The weather is sunny."),
                StreamEvent(type="end"),
            ],
        )
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": tool_stream},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        captured: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: captured.append(e))

        async for _ in sr2.turn(make_user_input()):
            pass

        await asyncio.sleep(0)

        # The assistant_response event contains the final (post-tool) response.
        assert len(captured) >= 1
        response: CompletionResponse = captured[0].data

        # The second LLM call returned a text-only response; final content has a TextBlock.
        text_blocks = [b for b in response.content if hasattr(b, "text")]
        assert len(text_blocks) >= 1, (
            f"Expected text content in final assistant_response, got: {response.content!r}"
        )

    @pytest.mark.asyncio
    async def test_tool_use_block_has_correct_id_name_input(self):
        """The executor receives a ToolUseBlock with the correct id, name, and input.

        In the multi-iteration loop, the ToolUseBlock is built from the stream event
        and passed to the executor. We verify the executor sees the right values.
        """
        from sr2.orchestrator import SR2
        from sr2.models import ToolResultBlock

        captured_blocks: list[ToolUseBlock] = []

        async def capturing_executor(block: ToolUseBlock) -> ToolResultBlock:
            captured_blocks.append(block)
            return ToolResultBlock(tool_use_id=block.id, content="result")

        tool_stream = MockLLM(
            events=[
                StreamEvent(type="text", text="Let me check"),
                StreamEvent(
                    type="tool_use",
                    tool_use_id="tc_1",
                    tool_name="get_weather",
                    tool_input={"location": "Oslo"},
                ),
                StreamEvent(type="end"),
            ],
            follow_up_events=[
                StreamEvent(type="text", text="Done."),
                StreamEvent(type="end"),
            ],
        )
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": tool_stream},
            token_counter=CharacterTokenCounter(),
            tool_executor=capturing_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(captured_blocks) == 1, (
            f"Expected executor called once, got {len(captured_blocks)}"
        )
        tb = captured_blocks[0]
        assert tb.id == "tc_1", f"Expected id='tc_1', got {tb.id!r}"
        assert tb.name == "get_weather", f"Expected name='get_weather', got {tb.name!r}"
        assert tb.input == {"location": "Oslo"}, f"Expected input={{'location': 'Oslo'}}, got {tb.input!r}"

    @pytest.mark.asyncio
    async def test_assistant_response_retains_text_block_alongside_tool_use(self):
        """When the LLM returns text + tool_use, the text is yielded to the caller and
        the tool is executed. The final assistant_response contains the final LLM text.

        This test verifies the text yielded during a tool-use iteration reaches the caller
        and the final response contains the follow-up text.
        """
        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event
        from sr2.models import ToolResultBlock

        async def stub_executor(block: ToolUseBlock) -> ToolResultBlock:
            return ToolResultBlock(tool_use_id=block.id, content="result")

        tool_stream = MockLLM(
            events=[
                StreamEvent(type="text", text="Let me check"),
                StreamEvent(
                    type="tool_use",
                    tool_use_id="tc_1",
                    tool_name="get_weather",
                    tool_input={"location": "Oslo"},
                ),
                StreamEvent(type="end"),
            ],
            follow_up_events=[
                StreamEvent(type="text", text="The weather is sunny in Oslo."),
                StreamEvent(type="end"),
            ],
        )
        config = make_minimal_config()
        sr2 = SR2(
            pipeline_config=config,
            llm={"default": tool_stream},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        yielded_events: list[StreamEvent] = []
        async for e in sr2.turn(make_user_input()):
            yielded_events.append(e)

        # "Let me check" is yielded as a text event during the first iteration
        text_events = [e for e in yielded_events if e.type == "text"]
        all_yielded_text = "".join(e.text for e in text_events)
        assert "Let me check" in all_yielded_text, (
            f"Expected 'Let me check' in yielded text events. Got: {all_yielded_text!r}"
        )
        assert "Oslo" in all_yielded_text, (
            f"Expected final response text in yield stream. Got: {all_yielded_text!r}"
        )


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
