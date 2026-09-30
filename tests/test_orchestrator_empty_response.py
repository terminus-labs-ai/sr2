"""Tests for sr2-37 — orchestrator empty-response guard.

Local models sometimes return a completely empty response (no text, no tool
calls, possibly only thinking events). Today that silently ends the turn with
no output. Required behavior:

  1. An "empty iteration" is a final-candidate iteration with no tool_use
     blocks AND text that is empty or whitespace-only. Thinking-only output
     counts as empty.
  2. On an empty iteration, the orchestrator retries the SAME request exactly
     once (one extra stream() call).
  3. If the retry is non-empty (text and/or tool calls), behavior proceeds as
     if the empty response never happened.
  4. If the retry is also empty, the turn finalizes with the placeholder text
     "[empty model response]" — yielded as a text StreamEvent AND stored in
     session history like normal final text.
  5. Non-empty responses: existing behavior unchanged — one stream() call per
     iteration, no retry.
  6. At most ONE retry per turn, even across multiple empty iterations.
  7. The retry does not consume a tool-loop iteration.

These tests are INTENTIONALLY RED until sr2-37 is implemented.
"""

from __future__ import annotations

import asyncio

import pytest

from sr2.config.models import PipelineConfig
from sr2.models import ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionResponse, StreamEvent
from conftest import (
    MockLLM,
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
    tool_use_event,
)

PLACEHOLDER = "[empty model response]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sr2(llm, *, tool_executor=None, config: PipelineConfig | None = None):
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config or make_minimal_config(),
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=tool_executor,
    )


async def _drain(sr2, text: str = "Hello") -> list[StreamEvent]:
    return [e async for e in sr2.turn(make_user_input(text))]


def _yielded_text(events: list[StreamEvent]) -> str:
    return "".join(e.text for e in events if e.type == "text")


def _empty_sequence() -> list[StreamEvent]:
    """A completely empty LLM response: only the terminating end event."""
    return [StreamEvent(type="end")]


# ---------------------------------------------------------------------------
# 1+2+3. Empty then non-empty retry
# ---------------------------------------------------------------------------


class TestEmptyThenNonEmptyRetry:
    @pytest.mark.asyncio
    async def test_empty_then_text_retry_streams_retry_text(self):
        """Empty first response, retry returns text → the retry text is streamed
        to the caller as if the empty response never happened."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [StreamEvent(type="text", text="Recovered answer."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert "Recovered answer." in _yielded_text(events), (
            f"Expected retry text streamed to caller. Yielded text: {_yielded_text(events)!r}"
        )

    @pytest.mark.asyncio
    async def test_empty_then_text_retry_calls_stream_exactly_twice(self):
        """Empty first response triggers exactly one extra stream() call."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [StreamEvent(type="text", text="Recovered answer."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        await _drain(sr2)

        assert len(llm.stream_calls) == 2, (
            f"Expected 2 stream() calls (original + 1 retry), got {len(llm.stream_calls)}"
        )

    @pytest.mark.asyncio
    async def test_retry_uses_equivalent_request(self):
        """The retry re-sends the SAME request: same messages, same system."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [StreamEvent(type="text", text="ok"), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        await _drain(sr2)

        assert len(llm.stream_calls) == 2
        first, second = llm.stream_calls
        assert second.messages == first.messages, (
            "Retry request messages differ from the original request"
        )
        assert second.system == first.system, (
            "Retry request system differs from the original request"
        )

    @pytest.mark.asyncio
    async def test_retry_text_stored_in_history(self):
        """The retry's text is stored as the assistant response (normal mechanism)."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [StreamEvent(type="text", text="Recovered answer."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        captured = []
        sr2.bus.subscribe("assistant_response", lambda e: captured.append(e))

        await _drain(sr2)
        await asyncio.sleep(0)

        assert len(captured) >= 1
        response: CompletionResponse = captured[0].data
        stored_text = "".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        assert "Recovered answer." in stored_text

    @pytest.mark.asyncio
    async def test_empty_then_tool_call_retry_continues_tool_loop(self):
        """Empty first response, retry returns a tool call → tool executes and
        the loop continues to a final text response."""
        executed: list[ToolUseBlock] = []

        async def capturing_executor(block: ToolUseBlock) -> ToolResultBlock:
            executed.append(block)
            return ToolResultBlock(tool_use_id=block.id, content="tool_result")

        llm = SequentialMockLLM([
            _empty_sequence(),
            [tool_use_event(tool_use_id="tc_retry", tool_name="get_weather"), StreamEvent(type="end")],
            [StreamEvent(type="text", text="Sunny in Oslo."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm, tool_executor=capturing_executor)

        events = await _drain(sr2)

        assert len(executed) == 1, (
            f"Expected the retry's tool call to execute once, got {len(executed)}"
        )
        assert executed[0].id == "tc_retry"
        assert "Sunny in Oslo." in _yielded_text(events), (
            f"Expected final text after tool loop. Yielded: {_yielded_text(events)!r}"
        )
        # empty + retry-with-tool + final = 3 stream calls
        assert len(llm.stream_calls) == 3, (
            f"Expected 3 stream() calls, got {len(llm.stream_calls)}"
        )


# ---------------------------------------------------------------------------
# 4. Retry also empty → placeholder
# ---------------------------------------------------------------------------


class TestRetryAlsoEmpty:
    @pytest.mark.asyncio
    async def test_double_empty_yields_placeholder_text_event(self):
        """Two empty responses → placeholder yielded as a text StreamEvent."""
        llm = SequentialMockLLM([_empty_sequence(), _empty_sequence()])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        text_events = [e for e in events if e.type == "text"]
        assert any(PLACEHOLDER in e.text for e in text_events), (
            f"Expected a text StreamEvent containing {PLACEHOLDER!r}. "
            f"Text events: {[e.text for e in text_events]!r}"
        )

    @pytest.mark.asyncio
    async def test_double_empty_makes_exactly_two_stream_calls(self):
        """Two empty responses → exactly 2 stream() calls (original + 1 retry)."""
        llm = SequentialMockLLM([_empty_sequence(), _empty_sequence()])
        sr2 = _make_sr2(llm)

        await _drain(sr2)

        assert len(llm.stream_calls) == 2, (
            f"Expected exactly 2 stream() calls, got {len(llm.stream_calls)}"
        )

    @pytest.mark.asyncio
    async def test_double_empty_stores_placeholder_in_history(self):
        """The placeholder is stored as the assistant response via the same
        mechanism as normal final text (assistant_response on the bus)."""
        llm = SequentialMockLLM([_empty_sequence(), _empty_sequence()])
        sr2 = _make_sr2(llm)

        captured = []
        sr2.bus.subscribe("assistant_response", lambda e: captured.append(e))

        await _drain(sr2)
        await asyncio.sleep(0)

        assert len(captured) >= 1, "No assistant_response event emitted"
        response: CompletionResponse = captured[0].data
        stored_text = "".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        assert PLACEHOLDER in stored_text, (
            f"Expected {PLACEHOLDER!r} stored in history, got {stored_text!r}"
        )

    @pytest.mark.asyncio
    async def test_placeholder_visible_in_next_turn_conversation(self):
        """Session history from a double-empty turn feeds the placeholder into
        the next turn's compiled request (proves real history storage)."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            _empty_sequence(),
            [StreamEvent(type="text", text="Second turn answer."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        await _drain(sr2, "first")
        await _drain(sr2, "second")

        assert len(llm.stream_calls) == 3
        third_request = llm.stream_calls[2]
        all_message_text = " ".join(
            block.text
            for msg in third_request.messages
            for block in msg.content
            if hasattr(block, "text")
        )
        assert PLACEHOLDER in all_message_text, (
            f"Expected {PLACEHOLDER!r} in turn-2 conversation history. "
            f"Got: {all_message_text!r}"
        )

    @pytest.mark.asyncio
    async def test_double_empty_still_emits_end_event(self):
        """Even the placeholder turn terminates with a single end event."""
        llm = SequentialMockLLM([_empty_sequence(), _empty_sequence()])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        end_events = [e for e in events if e.type == "end"]
        assert len(end_events) == 1


# ---------------------------------------------------------------------------
# 1. Empty-iteration definition: whitespace-only and thinking-only
# ---------------------------------------------------------------------------


class TestEmptyDefinition:
    @pytest.mark.asyncio
    async def test_whitespace_only_text_counts_as_empty(self):
        """A response whose only text is whitespace triggers the retry."""
        llm = SequentialMockLLM([
            [StreamEvent(type="text", text="   \n\t  "), StreamEvent(type="end")],
            [StreamEvent(type="text", text="Real answer."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 2, (
            f"Whitespace-only response must trigger a retry: expected 2 stream() "
            f"calls, got {len(llm.stream_calls)}"
        )
        assert "Real answer." in _yielded_text(events)

    @pytest.mark.asyncio
    async def test_thinking_only_response_counts_as_empty(self):
        """A response with only thinking events (no text, no tools) triggers the retry."""
        llm = SequentialMockLLM([
            [
                StreamEvent(type="thinking", text="Hmm, let me ponder..."),
                StreamEvent(type="end"),
            ],
            [StreamEvent(type="text", text="Pondered answer."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 2, (
            f"Thinking-only response must trigger a retry: expected 2 stream() "
            f"calls, got {len(llm.stream_calls)}"
        )
        assert "Pondered answer." in _yielded_text(events)

    @pytest.mark.asyncio
    async def test_thinking_then_empty_retry_yields_placeholder(self):
        """Thinking-only twice → placeholder finalization (thinking text must
        NOT be treated as response text)."""
        thinking_seq = [
            StreamEvent(type="thinking", text="thinking hard"),
            StreamEvent(type="end"),
        ]
        llm = SequentialMockLLM([thinking_seq, thinking_seq])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 2
        text_events = [e for e in events if e.type == "text"]
        assert any(PLACEHOLDER in e.text for e in text_events), (
            f"Expected {PLACEHOLDER!r} text event, got {[e.text for e in text_events]!r}"
        )


# ---------------------------------------------------------------------------
# 5. Non-empty responses: no behavior change
# ---------------------------------------------------------------------------


class TestNoSpuriousRetry:
    @pytest.mark.asyncio
    async def test_normal_text_response_makes_exactly_one_stream_call(self):
        """A normal non-empty response never triggers a retry."""
        llm = MockLLM(events=[
            StreamEvent(type="text", text="Normal answer."),
            StreamEvent(type="end"),
        ])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 1, (
            f"Non-empty response must not retry: expected 1 stream() call, "
            f"got {len(llm.stream_calls)}"
        )
        assert "Normal answer." in _yielded_text(events)

    @pytest.mark.asyncio
    async def test_tool_use_iteration_with_no_text_is_not_empty(self):
        """An iteration with tool calls but no text is NOT empty — no retry."""

        llm = SequentialMockLLM([
            [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
            [StreamEvent(type="text", text="Done."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm, tool_executor=stub_executor)

        events = await _drain(sr2)

        # Exactly 2 calls: tool iteration + final — no retry inserted anywhere.
        assert len(llm.stream_calls) == 2, (
            f"Expected 2 stream() calls (tool + final), got {len(llm.stream_calls)}"
        )
        assert "Done." in _yielded_text(events)


# ---------------------------------------------------------------------------
# 6. At most one retry per turn
# ---------------------------------------------------------------------------


class TestAtMostOneRetryPerTurn:
    @pytest.mark.asyncio
    async def test_second_empty_iteration_does_not_retry_again(self):
        """empty → retry returns tool call → tool runs → empty again.
        The retry budget is spent: the second empty iteration finalizes with
        the placeholder instead of retrying. Exactly 3 stream() calls."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
            _empty_sequence(),
        ])
        sr2 = _make_sr2(llm, tool_executor=stub_executor)

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 3, (
            f"Retry budget is 1 per turn: expected 3 stream() calls "
            f"(empty, tool retry, second empty — no further retry), "
            f"got {len(llm.stream_calls)}"
        )
        text_events = [e for e in events if e.type == "text"]
        assert any(PLACEHOLDER in e.text for e in text_events), (
            f"Second empty iteration must finalize with {PLACEHOLDER!r}. "
            f"Text events: {[e.text for e in text_events]!r}"
        )

    @pytest.mark.asyncio
    async def test_all_empty_never_exceeds_two_stream_calls(self):
        """A pathologically empty LLM is called at most twice per turn."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            _empty_sequence(),
            _empty_sequence(),
            _empty_sequence(),
        ])
        sr2 = _make_sr2(llm)

        await _drain(sr2)

        assert len(llm.stream_calls) == 2, (
            f"Expected exactly 2 stream() calls, got {len(llm.stream_calls)}"
        )

    @pytest.mark.asyncio
    async def test_retry_budget_resets_between_turns(self):
        """The one-retry cap is per TURN: turn 2 gets a fresh retry."""
        llm = SequentialMockLLM([
            # Turn 1: empty → retry succeeds
            _empty_sequence(),
            [StreamEvent(type="text", text="Turn one."), StreamEvent(type="end")],
            # Turn 2: empty → retry succeeds again
            _empty_sequence(),
            [StreamEvent(type="text", text="Turn two."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        events1 = await _drain(sr2, "first")
        events2 = await _drain(sr2, "second")

        assert len(llm.stream_calls) == 4
        assert "Turn one." in _yielded_text(events1)
        assert "Turn two." in _yielded_text(events2), (
            "Turn 2's empty response must retry — the retry budget resets per turn"
        )


# ---------------------------------------------------------------------------
# 7. Retry does not consume a tool-loop iteration
# ---------------------------------------------------------------------------


def _config_with_iteration_limit(max_tool_iterations: int) -> PipelineConfig:
    base = make_minimal_config()
    return PipelineConfig(
        layers=base.layers,
        max_tool_iterations=max_tool_iterations,
    )


class TestRetryDoesNotConsumeIteration:
    @pytest.mark.asyncio
    async def test_empty_at_iteration_limit_can_still_retry(self):
        """With max_tool_iterations=1: one tool iteration exhausts the budget,
        then an empty response arrives. The retry must still fire (it is not a
        tool-loop iteration) and its text must be streamed."""
        llm = SequentialMockLLM([
            [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
            _empty_sequence(),
            [StreamEvent(type="text", text="Final after retry."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(
            llm,
            tool_executor=stub_executor,
            config=_config_with_iteration_limit(1),
        )

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 3, (
            f"Expected 3 stream() calls (tool, empty, retry), got {len(llm.stream_calls)}"
        )
        assert "Final after retry." in _yielded_text(events), (
            f"Retry at the iteration limit must still stream its text. "
            f"Yielded: {_yielded_text(events)!r}"
        )

    @pytest.mark.asyncio
    async def test_retry_tool_call_belongs_to_same_iteration(self):
        """With max_tool_iterations=1: the first response is empty and the
        retry returns a tool call. The retry's tool call belongs to the SAME
        iteration (seq 0), so it must execute normally. If the retry consumed
        a tool-loop iteration, the tool call would arrive at seq 1 and trip
        ToolLoopLimitError instead."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
            [StreamEvent(type="text", text="Done after retry tool."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(
            llm,
            tool_executor=stub_executor,
            config=_config_with_iteration_limit(1),
        )

        events = await _drain(sr2)

        assert len(llm.stream_calls) == 3, (
            f"Expected 3 stream() calls (empty, retry tool, final), "
            f"got {len(llm.stream_calls)}"
        )
        assert "Done after retry tool." in _yielded_text(events), (
            f"Retry's tool call must execute in the same iteration and the "
            f"loop must finish normally. Yielded: {_yielded_text(events)!r}"
        )


# ---------------------------------------------------------------------------
# sr2-111: explicit empty_response retry event on the public turn() stream
# ---------------------------------------------------------------------------


class _OrderRecordingLLM(SequentialMockLLM):
    """SequentialMockLLM that records, at the start of each stream() call,
    how many turn() events the consumer had already received."""

    def __init__(self, call_sequences, received: list[StreamEvent]) -> None:
        super().__init__(call_sequences)
        self._received = received
        self.received_at_call: list[int] = []

    async def stream(self, request):
        self.received_at_call.append(len(self._received))
        async for event in super().stream(request):
            yield event


async def _drain_recording(sr2, received: list[StreamEvent], text: str = "Hello") -> list[StreamEvent]:
    start = len(received)
    async for event in sr2.turn(make_user_input(text)):
        received.append(event)
    return received[start:]


def _retry_events(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "retry"]


def _assert_empty_response_retry(event: StreamEvent, iteration: int) -> None:
    assert event.type == "retry"
    assert event.iteration == iteration, (
        f"retry event must name the current iteration {iteration}, got {event.iteration}"
    )
    assert event.meta is not None
    assert event.meta.get("reason") == "empty_response"
    assert event.meta.get("attempt") == 1


class TestEmptyResponseRetryEvent:
    @pytest.mark.asyncio
    async def test_retry_event_yielded_immediately_before_repeated_call(self):
        """The retry event is the last event the caller receives before the
        repeated model call starts, and the retry's output follows it."""
        received: list[StreamEvent] = []
        llm = _OrderRecordingLLM(
            [
                _empty_sequence(),
                [StreamEvent(type="text", text="Recovered answer."), StreamEvent(type="end")],
            ],
            received,
        )
        sr2 = _make_sr2(llm)

        events = await _drain_recording(sr2, received)

        retries = _retry_events(events)
        assert len(retries) == 1, f"expected one retry event, got {[e.type for e in events]}"
        retry_index = events.index(retries[0])
        _assert_empty_response_retry(retries[0], iteration=0)

        assert len(llm.received_at_call) == 2
        assert llm.received_at_call[0] <= retry_index, (
            "the original model call must start before the retry event is yielded"
        )
        assert llm.received_at_call[1] == retry_index + 1, (
            "the retry event must be the last event yielded before the repeated "
            f"model call; caller had received {llm.received_at_call[1]} events, "
            f"retry event is at index {retry_index}"
        )

        text_indices = [i for i, e in enumerate(events) if e.type == "text"]
        assert text_indices and all(i > retry_index for i in text_indices)
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_retry_event_names_current_iteration_after_tool_iteration(self):
        """Tool iteration 0 completes, iteration 1 is empty → the retry event
        carries iteration 1 and precedes the repeated call."""
        received: list[StreamEvent] = []
        llm = _OrderRecordingLLM(
            [
                [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
                _empty_sequence(),
                [StreamEvent(type="text", text="Final after retry."), StreamEvent(type="end")],
            ],
            received,
        )
        sr2 = _make_sr2(llm, tool_executor=stub_executor)

        events = await _drain_recording(sr2, received)

        retries = _retry_events(events)
        assert len(retries) == 1
        retry_index = events.index(retries[0])
        completes = [e for e in events if e.type == "iteration_complete"]
        assert [e.iteration for e in completes] == [0]
        assert events.index(completes[0]) < retry_index
        _assert_empty_response_retry(retries[0], iteration=1)

        assert len(llm.received_at_call) == 3
        assert llm.received_at_call[2] == retry_index + 1, (
            "the retry event must be the last event yielded before the repeated model call"
        )

    @pytest.mark.asyncio
    async def test_retry_event_emitted_when_retry_is_also_empty(self):
        """Double empty: the retry still happens, so exactly one retry event
        precedes it; the placeholder text follows the retry event."""
        received: list[StreamEvent] = []
        llm = _OrderRecordingLLM([_empty_sequence(), _empty_sequence()], received)
        sr2 = _make_sr2(llm)

        events = await _drain_recording(sr2, received)

        retries = _retry_events(events)
        assert len(retries) == 1
        retry_index = events.index(retries[0])
        _assert_empty_response_retry(retries[0], iteration=0)
        assert llm.received_at_call[1] == retry_index + 1
        placeholder_indices = [
            i for i, e in enumerate(events) if e.type == "text" and PLACEHOLDER in e.text
        ]
        assert placeholder_indices and placeholder_indices[0] > retry_index
        assert [e.type for e in events].count("end") == 1
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_no_retry_event_for_non_empty_response(self):
        llm = MockLLM(events=[
            StreamEvent(type="text", text="Normal answer."),
            StreamEvent(type="end"),
        ])
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert _retry_events(events) == []

    @pytest.mark.asyncio
    async def test_no_retry_event_for_tool_only_iteration(self):
        """A tool-call iteration without text is not empty — no retry event."""
        llm = SequentialMockLLM([
            [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
            [StreamEvent(type="text", text="Done."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm, tool_executor=stub_executor)

        events = await _drain(sr2)

        assert _retry_events(events) == []

    @pytest.mark.asyncio
    async def test_at_most_one_retry_event_per_turn(self):
        """empty → retry (tool call) → empty again: the second empty iteration
        finalizes without retrying, so only one retry event is yielded."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [tool_use_event(tool_use_id="tc_1"), StreamEvent(type="end")],
            _empty_sequence(),
        ])
        sr2 = _make_sr2(llm, tool_executor=stub_executor)

        events = await _drain(sr2)

        retries = _retry_events(events)
        assert len(retries) == 1
        _assert_empty_response_retry(retries[0], iteration=0)

    @pytest.mark.asyncio
    async def test_pathologically_empty_llm_yields_one_retry_event(self):
        llm = SequentialMockLLM([_empty_sequence()] * 4)
        sr2 = _make_sr2(llm)

        events = await _drain(sr2)

        assert len(_retry_events(events)) == 1

    @pytest.mark.asyncio
    async def test_each_turn_reports_its_own_retry_event(self):
        """The retry budget resets per turn, and so does the retry event."""
        llm = SequentialMockLLM([
            _empty_sequence(),
            [StreamEvent(type="text", text="Turn one."), StreamEvent(type="end")],
            _empty_sequence(),
            [StreamEvent(type="text", text="Turn two."), StreamEvent(type="end")],
        ])
        sr2 = _make_sr2(llm)

        events1 = await _drain(sr2, "first")
        events2 = await _drain(sr2, "second")

        for events in (events1, events2):
            retries = _retry_events(events)
            assert len(retries) == 1
            _assert_empty_response_retry(retries[0], iteration=0)
