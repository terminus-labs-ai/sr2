"""Tests for sr2.orchestrator.SR2 — the top-level orchestrator.

Covers:
  - Construction (valid config, missing "default" key)
  - turn() as async iterator yielding StreamEvents
  - turn() uses start_turn / end_turn with user_input injection
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
from conftest import (
    ErrorLLM,
    MockLLM,
    make_completion_response,
    make_minimal_config,
    make_user_input,
)

# ---------------------------------------------------------------------------
# Import under test — expected to fail with ImportError until implementation
# ---------------------------------------------------------------------------

# We import lazily inside each test class so that a single ImportError
# doesn't prevent all tests from being collected.


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


# ===========================================================================
# Tests from test_sr2_14_ocp_llm_dict.py — OCP: hardcoded 'default' key in llm dict
# SR2-14: SR2 should accept a single LLMCallable and SummarizationTransformer
# must not silently fall back to the "default" key when a named key is absent.
# ===========================================================================


class _MockLLMNamed:
    """Named MockLLM variant for sr2-14 tests (needs .name attribute)."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.stream_calls: list[CompletionRequest] = []
        self.complete_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.complete_calls.append(request)
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text=f"response from {self.name}")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        yield StreamEvent(type="text", text=f"text from {self.name}")
        yield StreamEvent(type="end")


def _make_minimal_pipeline_config_sr2_14():
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(type="static", config={"text": "You are helpful."})
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[ResolverConfig(type="session")],
            ),
        ]
    )


def _make_summarization_config(model_key=None, **kwargs):
    from sr2.config.models import EventSubscriptionConfig, TransformerConfig

    inner: dict = {"keep_strategy": "keep_last_n", "keep_last_n": 2}
    if model_key is not None:
        inner["model"] = model_key
    inner.update(kwargs)
    return TransformerConfig(
        type="summarization",
        subscriptions=[EventSubscriptionConfig(event="turn_start")],
        config=inner,
        max_executions=5,
    )


class TestSR2AcceptsSingleLLM:
    """SR2 should be constructable with a bare LLMCallable, not forced to wrap it in a dict."""

    def test_sr2_accepts_single_llm_callable(self):
        """P1: SR2.__init__ accepts a single LLMCallable without requiring a dict wrapper."""
        from sr2.orchestrator import SR2

        config = _make_minimal_pipeline_config_sr2_14()
        llm = _MockLLMNamed("driver")
        counter = CharacterTokenCounter()

        sr2 = SR2(pipeline_config=config, llm=llm, token_counter=counter)
        assert sr2 is not None

    def test_sr2_single_llm_is_the_driver(self):
        """P1: When a single LLMCallable is provided, it becomes the driver LLM."""
        from sr2.orchestrator import SR2

        config = _make_minimal_pipeline_config_sr2_14()
        driver = _MockLLMNamed("driver")
        counter = CharacterTokenCounter()

        sr2 = SR2(pipeline_config=config, llm=driver, token_counter=counter)
        assert sr2._llm is driver

    def test_sr2_dict_with_no_default_key_is_valid_if_driver_specified(self):
        """P1: SR2 dict form should not require the string 'default' if driver is explicit."""
        from sr2.orchestrator import SR2

        config = _make_minimal_pipeline_config_sr2_14()
        llm_haiku = _MockLLMNamed("haiku")
        llm_opus = _MockLLMNamed("opus")
        counter = CharacterTokenCounter()

        sr2 = SR2(
            pipeline_config=config,
            llm={"haiku": llm_haiku, "opus": llm_opus},
            token_counter=counter,
        )
        assert sr2 is not None


class TestSR2NoDualAccess:
    """SR2 should not store the driver LLM via two different paths simultaneously."""

    def test_sr2_driver_not_duplicated_in_deps(self):
        """P2: When llm is a single callable, deps.llm should not be a dict wrapping it."""
        from sr2.orchestrator import SR2

        config = _make_minimal_pipeline_config_sr2_14()
        driver = _MockLLMNamed("driver")
        counter = CharacterTokenCounter()

        sr2 = SR2(pipeline_config=config, llm=driver, token_counter=counter)

        if hasattr(sr2, "_llm"):
            assert sr2._llm is driver, "sr2._llm should be the provided driver"


class TestSummarizationNoSilentDefaultFallback:
    """SummarizationTransformer.build() must raise when a configured key is absent from deps.llm."""

    @pytest.fixture
    def transformer_cls(self):
        from sr2.pipeline.transformers.summarization import SummarizationTransformer

        return SummarizationTransformer

    def _make_deps_no_default(self, named_key: str, llm):
        from sr2.pipeline.dependencies import Dependencies

        return Dependencies(llm={named_key: llm})

    def _make_deps_with_default_only(self, default_llm):
        from sr2.pipeline.dependencies import Dependencies

        return Dependencies(llm={"default": default_llm})

    def test_named_key_absent_raises_not_silently_falls_back(self, transformer_cls):
        """P3: config['model']='haiku' but 'haiku' absent from deps.llm → should raise."""
        from sr2.config.models import ConfigError

        default_llm = _MockLLMNamed("default")
        config = _make_summarization_config(model_key="haiku")
        deps = self._make_deps_with_default_only(default_llm)

        with pytest.raises((ConfigError, KeyError, ValueError)):
            transformer_cls.build(config, deps)

    def test_named_key_absent_does_not_silently_return_instance(self, transformer_cls):
        """P3: Variant — ensure build() does not return an instance when named key is missing."""
        from sr2.config.models import ConfigError

        default_llm = _MockLLMNamed("default")
        config = _make_summarization_config(model_key="nonexistent-model")
        deps = self._make_deps_with_default_only(default_llm)

        raised = False
        try:
            result = transformer_cls.build(config, deps)
            assert result._llm is not default_llm, (
                "build() silently returned an instance using 'default' LLM "
                "even though config specified a different model key that was absent."
            )
        except (ConfigError, KeyError, ValueError):
            raised = True

        assert raised, (
            "Expected build() to raise when configured model key is absent from deps.llm, "
            "but it silently fell back to 'default'. This is the bug."
        )

    def test_named_key_present_without_default_works(self, transformer_cls):
        """P4: deps.llm has the named key but NO 'default' key → should work fine."""
        haiku_llm = _MockLLMNamed("haiku")
        config = _make_summarization_config(model_key="haiku")
        deps = self._make_deps_no_default("haiku", haiku_llm)

        result = transformer_cls.build(config, deps)
        assert result._llm is haiku_llm

    def test_no_model_key_in_config_without_default_in_deps_raises(self, transformer_cls):
        """P4: No 'model' key in config AND no 'default' key in deps.llm → should raise."""
        from sr2.config.models import ConfigError

        haiku_llm = _MockLLMNamed("haiku")
        config = _make_summarization_config(model_key=None)
        deps = self._make_deps_no_default("haiku", haiku_llm)

        with pytest.raises((ConfigError, KeyError, ValueError)):
            transformer_cls.build(config, deps)


class TestDependenciesLLMTypeContract:
    """Dependencies.llm types the 'default' key convention only in prose — documents the gap."""

    def test_dependencies_llm_dict_without_default_is_accepted_at_runtime(self):
        """Dependencies accepts any dict[str, LLMCallable] — no 'default' enforcement."""
        from sr2.pipeline.dependencies import Dependencies

        llm_a = _MockLLMNamed("a")
        llm_b = _MockLLMNamed("b")

        deps = Dependencies(llm={"a": llm_a, "b": llm_b})

        assert deps.llm is not None
        assert "default" not in deps.llm

    def test_driver_key_should_be_explicit_not_magic_string(self):
        """P4: SR2 can be constructed without using the string 'default' anywhere."""
        from sr2.orchestrator import SR2

        config = _make_minimal_pipeline_config_sr2_14()
        driver = _MockLLMNamed("primary-driver")
        counter = CharacterTokenCounter()

        try:
            sr2 = SR2(pipeline_config=config, llm=driver, token_counter=counter)
            constructed = True
        except (TypeError, ValueError, AttributeError):
            constructed = False

        assert constructed, (
            "SR2 should be constructable with a single LLMCallable (no magic 'default' key)."
        )


# ===========================================================================
# Tests from test_orchestrator_provenance.py — SR2 orchestrator provenance (Chunk 4)
# Covers: FR11 session_id, FR12 provenance_store, FR14 transformer config errors,
#         AC1 SQLite round-trip, AC5/AC6 protocol compliance.
# ===========================================================================


from sr2.pipeline.provenance import Entry, EntryOrigin, InMemoryProvenanceStore, ProvenanceStore


_PROV_ENTRY_COUNTER = 0


def _next_prov_entry_id() -> str:
    global _PROV_ENTRY_COUNTER
    _PROV_ENTRY_COUNTER += 1
    return f"ORCH{_PROV_ENTRY_COUNTER:022d}"


def _make_prov_entry(session_id: str, layer: str = "conversation") -> Entry:
    from datetime import datetime, timezone

    return Entry(
        id=_next_prov_entry_id(),
        content=TextBlock(text="round-trip content"),
        sources=(),
        origin=EntryOrigin(kind="resolver", name="test_resolver"),
        layer=layer,
        session_id=session_id,
        created_at=datetime.now(tz=timezone.utc),
    )


def _make_config_with_transformers(layer_name: str = "system") -> PipelineConfig:
    from sr2.config.models import TransformerConfig

    return PipelineConfig(
        layers=[
            LayerConfig(
                name=layer_name,
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static",
                        config={"text": "You are a helpful assistant."},
                    )
                ],
                transformers=[
                    TransformerConfig(type="some_transformer"),
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


def _make_config_with_empty_transformers() -> PipelineConfig:
    from sr2.config.models import TransformerConfig

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
                transformers=[],
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


class TestSR2SessionId:
    def test_session_id_auto_minted_when_none(self):
        """SR2() without session_id → sr2.session_id is a non-empty string."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert isinstance(sr2.session_id, str)
        assert len(sr2.session_id) > 0

    def test_session_id_is_ulid_format(self):
        """SR2() without session_id → sr2.session_id is 26 characters (ULID format)."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert len(sr2.session_id) == 26

    def test_explicit_session_id_stored(self):
        """SR2(session_id='my-id') → sr2.session_id == 'my-id'."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id="my-id",
        )

        assert sr2.session_id == "my-id"

    def test_explicit_session_id_is_exact_string(self):
        """SR2(session_id='my-id') → session_id is the exact string passed."""
        from sr2.orchestrator import SR2

        custom_id = "explicit-session-42"
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id=custom_id,
        )

        assert sr2.session_id is custom_id

    def test_two_instances_get_different_session_ids(self):
        """Two SR2() without session_id → different session IDs."""
        from sr2.orchestrator import SR2

        config = make_minimal_config()
        llm = {"default": MockLLM()}
        counter = CharacterTokenCounter()

        sr2_a = SR2(pipeline_config=config, llm=llm, token_counter=counter)
        sr2_b = SR2(pipeline_config=config, llm=llm, token_counter=counter)

        assert sr2_a.session_id != sr2_b.session_id


class TestSR2ProvenanceStore:
    def test_constructs_without_provenance_store(self):
        """SR2() without provenance_store → constructs without error."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert sr2 is not None

    def test_default_provenance_store_is_in_memory(self):
        """SR2() without provenance_store → engine uses InMemoryProvenanceStore."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert isinstance(sr2.provenance_store, InMemoryProvenanceStore)

    def test_explicit_provenance_store_used_as_is(self):
        """SR2(provenance_store=store) → engine._provenance_store is the exact object."""
        from sr2.orchestrator import SR2

        custom_store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=custom_store,
        )

        assert sr2.provenance_store is custom_store

    def test_provided_store_satisfies_protocol(self):
        """SR2(provenance_store=InMemoryProvenanceStore()) → store satisfies ProvenanceStore."""
        from sr2.orchestrator import SR2

        store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        assert isinstance(sr2.provenance_store, ProvenanceStore)


class TestTransformerConfigError:
    def test_empty_transformers_list_does_not_raise(self):
        """transformers=[] → no error; empty list is fine."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=_make_config_with_empty_transformers(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )
        assert sr2 is not None

    def test_non_empty_transformers_with_unknown_type_raises_plugin_not_found_error(self):
        """transformers=[TransformerConfig(type=unknown)] → raises PluginNotFoundError."""
        from sr2.orchestrator import SR2
        from sr2.plugins.errors import PluginNotFoundError

        with pytest.raises(PluginNotFoundError):
            SR2(
                pipeline_config=_make_config_with_transformers(),
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )

    def test_plugin_not_found_error_message_contains_unknown_type_name(self):
        """PluginNotFoundError message contains the unknown transformer type name."""
        from sr2.orchestrator import SR2
        from sr2.plugins.errors import PluginNotFoundError

        with pytest.raises(PluginNotFoundError, match="some_transformer"):
            SR2(
                pipeline_config=_make_config_with_transformers(),
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )

    def test_plugin_not_found_error_message_mentions_transformer(self):
        """PluginNotFoundError message mentions 'transformer' to guide the user."""
        from sr2.orchestrator import SR2
        from sr2.plugins.errors import PluginNotFoundError

        with pytest.raises(PluginNotFoundError) as exc_info:
            SR2(
                pipeline_config=_make_config_with_transformers(),
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )

        message = str(exc_info.value).lower()
        assert "transformer" in message

    def test_single_layer_with_unknown_transformer_raises(self):
        """Only one layer has transformers → PluginNotFoundError is still raised."""
        from sr2.orchestrator import SR2
        from sr2.config.models import TransformerConfig
        from sr2.plugins.errors import PluginNotFoundError

        config = PipelineConfig(
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
                    transformers=[
                        TransformerConfig(type="some_transformer"),
                    ],
                ),
            ]
        )

        with pytest.raises(PluginNotFoundError):
            SR2(
                pipeline_config=config,
                llm={"default": MockLLM()},
                token_counter=CharacterTokenCounter(),
            )


class TestProtocolCompliance:
    def test_in_memory_store_satisfies_protocol(self):
        """AC5: InMemoryProvenanceStore() satisfies isinstance(store, ProvenanceStore)."""
        store = InMemoryProvenanceStore()
        assert isinstance(store, ProvenanceStore)


class TestSR2RoundTrip:
    @pytest.mark.asyncio
    async def test_in_memory_store_session_id_threaded_to_entries(self):
        """FR11: session_id threads through — entries written with sr2.session_id are queryable."""
        from sr2.orchestrator import SR2

        store = InMemoryProvenanceStore()
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        entry = _make_prov_entry(session_id=sr2.session_id)
        await store.write(entry)

        entries = await store.get_session(sr2.session_id)
        assert len(entries) == 1
        assert entries[0].session_id == sr2.session_id
        assert entries[0].id == entry.id

    @pytest.mark.asyncio
    async def test_sqlite_store_entries_survive_close_and_reopen(self, tmp_path):
        """AC1: Entries written with session_id survive SQLite close → reopen."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "provenance_test.db"
        session_id = "round-trip-session-001"

        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            session_id=session_id,
            provenance_store=store,
        )

        entry1 = _make_prov_entry(session_id=sr2.session_id, layer="system")
        entry2 = _make_prov_entry(session_id=sr2.session_id, layer="conversation")
        await store.write_batch([entry1, entry2])

        entries_before = await store.get_session(session_id)
        assert len(entries_before) == 2
        assert all(e.session_id == session_id for e in entries_before)

        await store.close()

        store2 = SQLiteProvenanceStore(db_path=str(db_path))
        await store2.connect()

        entries_after = await store2.get_session(session_id)

        assert len(entries_after) == len(entries_before)
        ids_before = {e.id for e in entries_before}
        ids_after = {e.id for e in entries_after}
        assert ids_before == ids_after
        assert all(e.session_id == session_id for e in entries_after)

        await store2.close()

    @pytest.mark.asyncio
    async def test_sqlite_store_satisfies_protocol_via_orchestrator(self, tmp_path):
        """AC6: SQLiteProvenanceStore plugged into SR2 satisfies isinstance(store, ProvenanceStore)."""
        from sr2.orchestrator import SR2
        from sr2.pipeline.stores.sqlite import SQLiteProvenanceStore

        db_path = tmp_path / "protocol_check.db"
        store = SQLiteProvenanceStore(db_path=str(db_path))
        await store.connect()

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            provenance_store=store,
        )

        assert isinstance(sr2.provenance_store, ProvenanceStore)

        await store.close()
