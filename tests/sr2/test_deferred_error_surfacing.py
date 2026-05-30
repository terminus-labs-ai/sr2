"""Tests for sr2-6 / FR8: surface deferred-phase errors on join.

Requirements:
  - Errors from the deferred post_process task (scheduled after end_turn)
    surface on the NEXT turn() as StreamEvent(type="error") yielded early,
    before any other events.
  - CancelledError propagates (not caught and wrapped).
  - Bus errors accumulated during the deferred task's execution are also
    surfaced.
  - If the deferred task raises an unhandled exception, it surfaces as an
    error event (not a raw Python exception) so the caller can handle it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sr2.models import ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
)


class TestDeferredErrorSurfacing:
    """FR8: Deferred post_process errors surface as StreamEvent(type='error') on next turn."""

    @pytest.mark.asyncio
    async def test_deferred_post_process_error_surfaces_on_next_turn(self):
        """When post_process raises, the error surfaces as StreamEvent(type='error')
        at the start of the next turn, before any text events."""
        from sr2.orchestrator import SR2

        order: list[str] = []

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Turn 1 response."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Turn 2 response."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        call_count = 0

        async def failing_post_process(response: CompletionResponse) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("post_process boom")

        sr2.post_process = failing_post_process

        # Turn 1 — post_process will fail, but the turn itself completes
        events_t1 = []
        async for event in sr2.turn(make_user_input("Q1")):
            order.append(f"t1:{event.type}")
            events_t1.append(event)

        # Turn 1 should complete normally (end event emitted, error is deferred)
        end_events_t1 = [e for e in events_t1 if e.type == "end"]
        assert len(end_events_t1) == 1, (
            "Turn 1 must still emit end event even though post_process will fail"
        )

        # Turn 2 — the deferred error from Turn 1's post_process should surface first
        events_t2 = []
        async for event in sr2.turn(make_user_input("Q2")):
            order.append(f"t2:{event.type}")
            events_t2.append(event)

        # First event of turn 2 should be the error event
        assert events_t2[0].type == "error", (
            f"First event of turn 2 should be 'error' from deferred post_process, "
            f"got '{events_t2[0].type}'. Order: {order}"
        )
        assert events_t2[0].errors is not None
        assert any("post_process boom" in err for err in events_t2[0].errors), (
            f"Error should contain the post_process exception. Got: {events_t2[0].errors}"
        )

        # Error event must come before any text events in turn 2
        text_idx = next((i for i, e in enumerate(events_t2) if e.type == "text"), None)
        error_idx = next((i for i, e in enumerate(events_t2) if e.type == "error"), None)
        assert error_idx is not None and text_idx is not None
        assert error_idx < text_idx, (
            "Error event must appear before text events in turn 2"
        )

    @pytest.mark.asyncio
    async def test_deferred_error_does_not_propagate_as_raw_exception(self):
        """Deferred post_process errors are wrapped as StreamEvent, not re-raised."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="T1."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="T2."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        async def always_fail_post_process(response: CompletionResponse) -> None:
            raise ValueError("always fails")

        sr2.post_process = always_fail_post_process

        # Turn 1
        async for _ in sr2.turn(make_user_input("Q1")):
            pass

        # Turn 2 — should NOT raise, should yield error event instead
        events_t2 = []
        async for event in sr2.turn(make_user_input("Q2")):
            events_t2.append(event)

        error_events = [e for e in events_t2 if e.type == "error"]
        assert len(error_events) == 1, (
            f"Expected 1 error event from deferred post_process, got {len(error_events)}"
        )

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        """CancelledError from deferred task propagates, not wrapped."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="T1."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="T2."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        # Turn 1
        async for _ in sr2.turn(make_user_input("Q1")):
            pass

        # Cancel the deferred task
        assert sr2._pp_task is not None
        sr2._pp_task.cancel()

        # Turn 2 — should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            async for _ in sr2.turn(make_user_input("Q2")):
                pass

    @pytest.mark.asyncio
    async def test_no_deferred_error_when_post_process_succeeds(self):
        """When post_process succeeds, no error event on next turn."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="T1."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="T2."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        # Turn 1
        async for _ in sr2.turn(make_user_input("Q1")):
            pass

        # Turn 2 — no deferred errors, so no error event
        events_t2 = []
        async for event in sr2.turn(make_user_input("Q2")):
            events_t2.append(event)

        error_events = [e for e in events_t2 if e.type == "error"]
        assert len(error_events) == 0, (
            f"No deferred errors expected, but got {len(error_events)} error event(s)"
        )

    @pytest.mark.asyncio
    async def test_deferred_error_on_first_turn_no_pending_task(self):
        """First turn has no pending task, so no deferred error event."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="First."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        events = []
        async for event in sr2.turn(make_user_input("Q1")):
            events.append(event)

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 0, (
            "First turn should have no deferred error events"
        )

    @pytest.mark.asyncio
    async def test_deferred_error_includes_exception_message(self):
        """Error event's errors field contains the exception message."""
        from sr2.orchestrator import SR2

        error_msg = "database connection lost during extraction"

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="T1."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="T2."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        async def failing_pp(response: CompletionResponse) -> None:
            raise ConnectionError(error_msg)

        sr2.post_process = failing_pp

        # Turn 1
        async for _ in sr2.turn(make_user_input("Q1")):
            pass

        # Turn 2
        events_t2 = []
        async for event in sr2.turn(make_user_input("Q2")):
            events_t2.append(event)

        error_event = events_t2[0]
        assert error_event.type == "error"
        assert error_event.errors is not None
        assert any(error_msg in err for err in error_event.errors), (
            f"Error message '{error_msg}' should be in errors. Got: {error_event.errors}"
        )

    @pytest.mark.asyncio
    async def test_turn_continues_after_deferred_error(self):
        """After surfacing deferred error, turn 2 continues normally."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="T1."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="T2 response."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        async def failing_pp(response: CompletionResponse) -> None:
            raise RuntimeError("deferred failure")

        sr2.post_process = failing_pp

        # Turn 1
        async for _ in sr2.turn(make_user_input("Q1")):
            pass

        # Turn 2
        events_t2 = []
        async for event in sr2.turn(make_user_input("Q2")):
            events_t2.append(event)

        # Should have: error, text, end
        text_events = [e for e in events_t2 if e.type == "text"]
        end_events = [e for e in events_t2 if e.type == "end"]
        error_events = [e for e in events_t2 if e.type == "error"]

        assert len(error_events) == 1
        assert len(text_events) > 0, "Turn 2 should still produce text events"
        assert len(end_events) == 1, "Turn 2 should still end with exactly one 'end'"
        assert events_t2[-1].type == "end", "Last event must be 'end'"
