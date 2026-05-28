"""Tests for sr2-26: FR3+FR11+FR14 — rewrite SR2.turn() as multi-iteration loop.

Requirements:
  FR3  — Tool loop: start_turn → stream LLM → detect tool_use → execute tools → queue events
         → continue_turn → repeat until no more tool_use → end_turn.
  FR11 — post_process is called exactly ONCE after the full loop completes (not per iteration).
  FR14 — A single "end" StreamEvent is emitted to the caller once the entire loop finishes
         (not after each LLM call).

Tests will FAIL until the feature is implemented (red phase).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Helpers & fakes
# ---------------------------------------------------------------------------


def make_user_input(text: str = "Hello") -> list:
    return [TextBlock(text=text)]


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


class SequentialMockLLM:
    """LLM that returns a different event list on each successive stream() call.

    Constructed with a list of call-sequences. Each element in `call_sequences`
    is the list of StreamEvents returned on that call (0-indexed).

    Once all sequences are exhausted, subsequent calls return the last sequence.
    This makes writing multi-iteration tests declarative.
    """

    def __init__(self, call_sequences: list[list[StreamEvent]]) -> None:
        assert call_sequences, "Must provide at least one call sequence"
        self._sequences = call_sequences
        self.stream_calls: list[CompletionRequest] = []

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        idx = min(len(self.stream_calls), len(self._sequences) - 1)
        self.stream_calls.append(request)
        for event in self._sequences[idx]:
            yield event


def tool_use_event(
    tool_use_id: str = "call_001",
    tool_name: str = "get_weather",
    tool_input: dict[str, Any] | None = None,
) -> StreamEvent:
    return StreamEvent(
        type="tool_use",
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_input=tool_input or {"location": "Oslo"},
    )


async def stub_executor(block: ToolUseBlock) -> ToolResultBlock:
    """Minimal executor that always returns a synthetic result."""
    return ToolResultBlock(
        tool_use_id=block.id,
        content=f"result_for_{block.name}",
    )


async def recording_executor_factory() -> tuple[list[ToolUseBlock], Any]:
    """Return (call_log, executor_fn) so tests can inspect invocations."""
    call_log: list[ToolUseBlock] = []

    async def executor(block: ToolUseBlock) -> ToolResultBlock:
        call_log.append(block)
        return ToolResultBlock(tool_use_id=block.id, content=f"result_for_{block.name}")

    return call_log, executor


# ---------------------------------------------------------------------------
# FR3 — Tool loop executes multiple iterations
# ---------------------------------------------------------------------------


class TestToolLoopIterations:
    """FR3: The turn loop must iterate: tool_use → execute → continue → no tool_use → end."""

    @pytest.mark.asyncio
    async def test_single_tool_use_triggers_two_llm_calls(self):
        """Single tool_use: LLM is called twice — first returns tool_use, second returns text.

        The current implementation calls the LLM only once per turn, so this test
        will FAIL until the multi-iteration loop is implemented.
        """
        from sr2.orchestrator import SR2

        # First call: returns tool_use. Second call: returns text (no tool_use).
        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Let me check."),
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="The weather is sunny."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(llm.stream_calls) == 2, (
            f"Expected 2 LLM stream calls (one per iteration), got {len(llm.stream_calls)}"
        )

    @pytest.mark.asyncio
    async def test_two_sequential_tool_calls_trigger_three_llm_calls(self):
        """Two tool_use blocks across two iterations: loop runs 3 LLM calls total.

        Iteration 1 → tool_use (tool1)
        Iteration 2 → tool_use (tool2)
        Iteration 3 → text only → stop

        Will FAIL until multi-iteration loop is implemented.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "search_web"),
                    StreamEvent(type="end"),
                ],
                [
                    tool_use_event("call_002", "read_file"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Here is the final answer."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(llm.stream_calls) == 3, (
            f"Expected 3 LLM calls (2 tool iterations + 1 final), got {len(llm.stream_calls)}"
        )

    @pytest.mark.asyncio
    async def test_no_tool_use_runs_single_iteration(self):
        """No tool_use: loop completes in a single LLM call (same as current behaviour).

        This ensures the multi-iteration path doesn't break the no-tool case.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Hello there."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(llm.stream_calls) == 1, (
            f"Expected 1 LLM call for no-tool turn, got {len(llm.stream_calls)}"
        )

    @pytest.mark.asyncio
    async def test_executor_called_once_per_tool_use_block(self):
        """The tool executor is called exactly once for each tool_use block encountered.

        Turn with one tool_use → executor called once.
        """
        from sr2.orchestrator import SR2

        call_log, executor = await recording_executor_factory()

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather", {"location": "Oslo"}),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(call_log) == 1, f"Expected executor called once, got {len(call_log)}"
        assert call_log[0].id == "call_001"
        assert call_log[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_executor_called_twice_for_two_tool_use_blocks(self):
        """Executor is called once per iteration — two tool_use blocks → two executor calls."""
        from sr2.orchestrator import SR2

        call_log, executor = await recording_executor_factory()

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "search_web"),
                    StreamEvent(type="end"),
                ],
                [
                    tool_use_event("call_002", "read_file"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="All done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(call_log) == 2, (
            f"Expected executor called twice (once per tool_use), got {len(call_log)}"
        )
        names = [b.name for b in call_log]
        assert "search_web" in names
        assert "read_file" in names


# ---------------------------------------------------------------------------
# FR14 — Single end event
# ---------------------------------------------------------------------------


class TestSingleEndEvent:
    """FR14: Exactly one StreamEvent(type='end') must appear in the turn output stream."""

    @pytest.mark.asyncio
    async def test_single_tool_use_emits_exactly_one_end_event(self):
        """With a tool_use loop, only ONE end event must be emitted to the caller.

        The current implementation yields the LLM's raw stream, so the
        intermediate 'end' from the first LLM call would leak through.
        This test will FAIL until the multi-iteration loop gates events.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        events = [e async for e in sr2.turn(make_user_input())]
        end_events = [e for e in events if e.type == "end"]

        assert len(end_events) == 1, (
            f"Expected exactly 1 end event in caller's stream, got {len(end_events)}. "
            f"Intermediate 'end' events from tool iterations must NOT be forwarded."
        )

    @pytest.mark.asyncio
    async def test_two_tool_iterations_emit_exactly_one_end_event(self):
        """Two-iteration tool loop must still emit only one end event to the caller."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "search_web"),
                    StreamEvent(type="end"),
                ],
                [
                    tool_use_event("call_002", "read_file"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Final answer."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        events = [e async for e in sr2.turn(make_user_input())]
        end_events = [e for e in events if e.type == "end"]

        assert len(end_events) == 1, (
            f"Expected exactly 1 end event for 2-iteration turn, got {len(end_events)}"
        )

    @pytest.mark.asyncio
    async def test_text_only_turn_emits_exactly_one_end_event(self):
        """Text-only turn (no tool_use): still exactly one end event."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="A simple response."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        events = [e async for e in sr2.turn(make_user_input())]
        end_events = [e for e in events if e.type == "end"]

        assert len(end_events) == 1, (
            f"Expected exactly 1 end event for text-only turn, got {len(end_events)}"
        )

    @pytest.mark.asyncio
    async def test_end_event_is_the_last_yielded_event(self):
        """The end event must be the final event in the caller's stream.

        Intermediate events (text, tool_use) must all precede it.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="The weather is fine."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        events = [e async for e in sr2.turn(make_user_input())]

        assert len(events) > 0, "Expected at least one event in the stream"
        assert events[-1].type == "end", (
            f"Last event must be 'end', got '{events[-1].type}'"
        )


# ---------------------------------------------------------------------------
# FR11 — post_process called exactly once
# ---------------------------------------------------------------------------


class TestPostProcessCalledOnce:
    """FR11: post_process must be called exactly once after the full loop completes."""

    @pytest.mark.asyncio
    async def test_post_process_called_once_for_single_tool_use(self):
        """After a 2-iteration turn (1 tool_use), post_process is called exactly once.

        The current implementation already calls post_process once per turn.
        This test will pass now but guards against a regression where the
        multi-iteration refactor calls post_process once per LLM call.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="The weather is fine."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        post_process_calls: list[CompletionResponse] = []
        original_post_process = sr2.post_process

        async def spy_post_process(response: CompletionResponse) -> None:
            post_process_calls.append(response)
            return await original_post_process(response)

        sr2.post_process = spy_post_process

        async for _ in sr2.turn(make_user_input()):
            pass

        # Allow fire-and-forget tasks to settle
        await asyncio.sleep(0)

        assert len(post_process_calls) == 1, (
            f"Expected post_process called exactly once after a 2-iteration turn, "
            f"got {len(post_process_calls)} calls. "
            f"post_process must be called after the FULL loop, not per LLM call."
        )

    @pytest.mark.asyncio
    async def test_post_process_called_once_for_two_tool_iterations(self):
        """After a 3-iteration turn (2 tool_use blocks), post_process is called exactly once.

        Will FAIL if the multi-iteration loop naively calls post_process after each
        LLM call rather than after the full loop finishes.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "search_web"),
                    StreamEvent(type="end"),
                ],
                [
                    tool_use_event("call_002", "read_file"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Final answer."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        post_process_calls: list[CompletionResponse] = []
        original_post_process = sr2.post_process

        async def spy_post_process(response: CompletionResponse) -> None:
            post_process_calls.append(response)
            return await original_post_process(response)

        sr2.post_process = spy_post_process

        async for _ in sr2.turn(make_user_input()):
            pass

        await asyncio.sleep(0)

        assert len(post_process_calls) == 1, (
            f"Expected post_process called exactly once after a 3-iteration turn, "
            f"got {len(post_process_calls)} calls."
        )

    @pytest.mark.asyncio
    async def test_post_process_not_called_before_loop_completes(self):
        """post_process must not be called while the tool loop is still running.

        We track whether post_process is called BEFORE the second LLM call
        completes. If it is, the implementation called it mid-loop.

        Approach: the spy records *when* it was called relative to the stream
        events already yielded. At minimum, the final "text" event from the
        second LLM call must have been yielded before post_process fires.
        """
        from sr2.orchestrator import SR2

        second_llm_text = "The weather is sunny."
        final_events_seen_before_post_process: list[str] = []
        events_yielded: list[str] = []

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text=second_llm_text),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        original_post_process = sr2.post_process

        async def spy_post_process(response: CompletionResponse) -> None:
            # Record what text events had been yielded at the moment we're called
            final_events_seen_before_post_process.extend(events_yielded)
            return await original_post_process(response)

        sr2.post_process = spy_post_process

        async for event in sr2.turn(make_user_input()):
            if event.type == "text":
                events_yielded.append(event.text or "")

        await asyncio.sleep(0)

        # post_process must have been called at some point (fire-and-forget settled)
        # AND it must have seen the second LLM's text event already yielded.
        # (If it fires mid-loop, final_events_seen_before_post_process won't contain
        # the second iteration's text.)
        assert len(final_events_seen_before_post_process) > 0 or len(events_yielded) == 0, (
            "post_process was called before any events were yielded — fires too early"
        )

        # If the second LLM call produced text, that text must appear in the stream
        # before post_process was called. Check that there's at least content from
        # the final iteration.
        # (This assertion only holds if the second call yielded text events at all.)
        if second_llm_text in " ".join(events_yielded):
            assert second_llm_text in " ".join(final_events_seen_before_post_process), (
                "post_process fired before the final LLM iteration's text was yielded. "
                "post_process must only be called after the full loop finishes."
            )

    @pytest.mark.asyncio
    async def test_post_process_called_once_for_text_only_turn(self):
        """Text-only turn: post_process still called exactly once (baseline)."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Simple reply."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        post_process_calls: list[CompletionResponse] = []
        original_post_process = sr2.post_process

        async def spy_post_process(response: CompletionResponse) -> None:
            post_process_calls.append(response)
            return await original_post_process(response)

        sr2.post_process = spy_post_process

        async for _ in sr2.turn(make_user_input()):
            pass

        await asyncio.sleep(0)

        assert len(post_process_calls) == 1, (
            f"Expected post_process called once for text-only turn, got {len(post_process_calls)}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestToolLoopErrorHandling:
    """Executor errors must be wrapped as ToolResultBlock(is_error=True) and fed
    back to the LLM — the iterator must NOT raise on executor errors."""

    @pytest.mark.asyncio
    async def test_executor_exception_does_not_raise(self):
        """When the executor raises, turn() must NOT propagate the exception.

        Instead the error is wrapped as ToolResultBlock(is_error=True) and fed
        to the next LLM call. The turn continues normally.
        """
        from sr2.orchestrator import SR2

        class ToolExecutionError(Exception):
            pass

        async def failing_executor(block: ToolUseBlock) -> ToolResultBlock:
            raise ToolExecutionError("backend tool failed")

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "broken_tool"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Sorry, the tool failed."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=failing_executor,
        )

        # Must NOT raise — executor errors are wrapped and fed back to the LLM
        events = [e async for e in sr2.turn(make_user_input())]
        assert any(e.type == "end" for e in events), "turn() must complete normally"

    @pytest.mark.asyncio
    async def test_executor_exception_produces_error_result_in_next_llm_call(self):
        """When the executor raises, the next LLM call must receive a
        ToolResultBlock(is_error=True) with the error message as content.

        We verify this by inspecting what messages the second LLM call sees.
        """
        from sr2.orchestrator import SR2

        error_message = "backend tool failed"

        async def failing_executor(block: ToolUseBlock) -> ToolResultBlock:
            raise RuntimeError(error_message)

        # Capture the second call's request to inspect tool result content
        captured_requests: list[CompletionRequest] = []

        class CapturingMockLLM:
            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                if len(captured_requests) == 1:
                    # First call: return tool_use
                    yield tool_use_event("call_001", "broken_tool")
                    yield StreamEvent(type="end")
                else:
                    # Second call: return text (done)
                    yield StreamEvent(type="text", text="I encountered an error.")
                    yield StreamEvent(type="end")

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": CapturingMockLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=failing_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(captured_requests) == 2, "LLM must be called twice"

        # The second request must contain a tool_result message with is_error=True
        second_request = captured_requests[1]
        tool_result_found = False
        for msg in second_request.messages:
            for block in msg.content:
                if hasattr(block, "is_error") and block.is_error:
                    tool_result_found = True
                    assert error_message in str(block.content), (
                        f"Error content must include the exception message. "
                        f"Got: {block.content!r}"
                    )

        assert tool_result_found, (
            "Second LLM call must include a ToolResultBlock(is_error=True) "
            "wrapping the executor's exception"
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError bypasses error wrapping and propagates directly."""
        from sr2.orchestrator import SR2

        async def cancelling_executor(block: ToolUseBlock) -> ToolResultBlock:
            raise asyncio.CancelledError()

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "cancel_me"),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=cancelling_executor,
        )

        with pytest.raises(asyncio.CancelledError):
            async for _ in sr2.turn(make_user_input()):
                pass


# ---------------------------------------------------------------------------
# FR7 — ToolLoopLimitError
# ---------------------------------------------------------------------------


class TestToolLoopLimitError:
    """FR7: turn() raises ToolLoopLimitError when tool iterations exceed the limit."""

    @pytest.mark.asyncio
    async def test_tool_loop_limit_raises_after_default_limit(self):
        """When the LLM always returns tool_use and the executor always succeeds,
        turn() must raise ToolLoopLimitError after the default limit (25 iterations).
        """
        from sr2.orchestrator import SR2
        from sr2.config.models import ToolLoopLimitError

        # LLM always returns a tool_use (never terminates)
        always_tool_llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_loop", "infinite_tool"),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": always_tool_llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        with pytest.raises(ToolLoopLimitError):
            async for _ in sr2.turn(make_user_input()):
                pass


# ---------------------------------------------------------------------------
# FR11 — post_process awaited (not fire-and-forget)
# ---------------------------------------------------------------------------


class TestPostProcessAwaited:
    """FR11 (strong): post_process must be awaited before the iterator exits."""

    @pytest.mark.asyncio
    async def test_post_process_awaited_before_turn_exits(self):
        """post_process must be awaited synchronously — flag is set before turn() exits,
        with NO asyncio.sleep between the turn and the assertion.
        """
        from sr2.orchestrator import SR2

        class TrackingSR2(SR2):
            flag: bool = False

            async def post_process(self, response: CompletionResponse) -> None:
                self.flag = True

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Hello."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = TrackingSR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        # No asyncio.sleep — if post_process is fire-and-forget, flag is still False
        assert sr2.flag is True, (
            "post_process must be awaited before turn() exits. "
            "If this fails, post_process is fire-and-forget (create_task) instead of awaited."
        )


# ---------------------------------------------------------------------------
# FR13 — iteration_complete event
# ---------------------------------------------------------------------------


class TestIterationCompleteEvent:
    """FR13: iteration_complete event appears in the stream for multi-iteration turns."""

    @pytest.mark.asyncio
    async def test_iteration_complete_appears_in_multi_iteration_turn(self):
        """After a tool_use iteration, an iteration_complete event must be emitted."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=stub_executor,
        )

        events = [e async for e in sr2.turn(make_user_input())]
        event_types = [e.type for e in events]

        assert "iteration_complete" in event_types, (
            f"Expected 'iteration_complete' event in stream for a tool-use turn. "
            f"Got event types: {event_types}"
        )
