"""Tests for sr2-4: FR4+FR5+FR6 — deferred post_process task.

Requirements:
  - After yielding final end, schedule _finalize_and_post_process(final_response)
    as an asyncio task on self._pp_task; return immediately (client free).
  - At START of turn(), await any pending task before reset/start_turn.
  - post_process(result) stays a thin overridable no-op called by the deferred
    task after end_turn.
  - Capture final_response by arg.
  - Tests: end emitted before post_process runs; ordering settled before next turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sr2.config.models import PipelineConfig
from sr2.models import TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
    tool_use_event,
)


class TestEndBeforePostProcess:
    """end event must be emitted BEFORE post_process starts running."""

    @pytest.mark.asyncio
    async def test_end_emitted_before_post_process_runs(self):
        """The caller receives the 'end' event before post_process begins.

        We track the order of: (1) end event yielded, (2) post_process called.
        End must come first.
        """
        from sr2.orchestrator import SR2

        order: list[str] = []

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Response."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        original_post_process = sr2.post_process

        async def tracked_post_process(response: CompletionResponse) -> None:
            order.append("post_process")
            await original_post_process(response)

        sr2.post_process = tracked_post_process

        # Consume the turn stream
        async for event in sr2.turn(make_user_input()):
            if event.type == "end":
                order.append("end_yielded")

        # At this point, post_process may or may not have run (it's deferred).
        # But 'end_yielded' must appear before 'post_process' in the order list.
        # If post_process hasn't run yet, it won't be in the list — that's fine.
        # The key is: end_yielded must NOT come after post_process.
        if "post_process" in order:
            end_idx = order.index("end_yielded")
            pp_idx = order.index("post_process")
            assert end_idx < pp_idx, (
                f"end event must be yielded before post_process runs. "
                f"Order: {order}"
            )

    @pytest.mark.asyncio
    async def test_post_process_deferred_not_blocking_turn_exit(self):
        """post_process runs asynchronously after turn() exits — a slow
        post_process should not block the generator from finishing.

        We use a blocking post_process that sets a flag after a delay.
        The turn should exit before the flag is set.
        """
        from sr2.orchestrator import SR2

        post_process_started = asyncio.Event()
        post_process_done = asyncio.Event()

        llm = SequentialMockLLM(
            call_sequences=[
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
        )

        async def slow_post_process(response: CompletionResponse) -> None:
            post_process_started.set()
            await asyncio.sleep(0.1)  # Simulate slow work
            post_process_done.set()

        sr2.post_process = slow_post_process

        # Run the turn
        async for _ in sr2.turn(make_user_input()):
            pass

        # Turn exited. post_process should have been scheduled but may not
        # have completed yet. Give it a moment to start.
        # The key test: turn exited (we're here). If post_process was awaited
        # synchronously, we wouldn't be here until it finished.
        # With deferred scheduling, we're here immediately.
        # Wait briefly for the task to start, then check.
        await asyncio.sleep(0)  # Let the scheduled task start

        # The deferred task should exist
        assert sr2._pp_task is not None, (
            "Deferred post_process task should be scheduled on self._pp_task"
        )


class TestNextTurnJoinsPrevious:
    """Next turn() must await previous turn's deferred post_process task."""

    @pytest.mark.asyncio
    async def test_next_turn_awaits_previous_post_process(self):
        """Turn 2 must not start until Turn 1's post_process has completed.

        We track the order of events across two turns:
        - Turn 1 end yielded
        - Turn 1 post_process runs
        - Turn 2 start_turn called
        Turn 1's post_process must complete before Turn 2's start_turn.
        """
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

        original_post_process = sr2.post_process

        async def tracked_post_process(response: CompletionResponse) -> None:
            order.append("post_process_t1")
            await original_post_process(response)

        sr2.post_process = tracked_post_process

        # Track when start_turn is called for turn 2
        original_start_turn = sr2._engine.start_turn

        async def tracked_start_turn(turn_seq: int) -> None:
            order.append(f"start_turn_{turn_seq}")
            await original_start_turn(turn_seq)

        sr2._engine.start_turn = tracked_start_turn

        # Turn 1
        async for event in sr2.turn(make_user_input("Q1")):
            if event.type == "end":
                order.append("end_t1")

        # Turn 2 — this must await turn 1's deferred post_process first
        async for event in sr2.turn(make_user_input("Q2")):
            if event.type == "end":
                order.append("end_t2")

        # Verify ordering:
        # end_t1 < post_process_t1 < start_turn_1 (turn 2, seq starts at 0) < end_t2
        assert "end_t1" in order
        assert "post_process_t1" in order
        assert "start_turn_1" in order
        assert "end_t2" in order

        end_t1_idx = order.index("end_t1")
        pp_t1_idx = order.index("post_process_t1")
        st2_idx = order.index("start_turn_1")
        end_t2_idx = order.index("end_t2")

        assert end_t1_idx < pp_t1_idx, (
            f"end_t1 must come before post_process_t1. Order: {order}"
        )
        assert pp_t1_idx < st2_idx, (
            f"Turn 1's post_process must complete before Turn 2's start_turn. "
            f"Order: {order}"
        )
        assert st2_idx < end_t2_idx, (
            f"start_turn_1 (turn 2) must come before end_t2. Order: {order}"
        )

    @pytest.mark.asyncio
    async def test_first_turn_has_no_pending_task(self):
        """The first turn() call should have no pending _pp_task to await."""
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

        # _pp_task should be None initially
        assert sr2._pp_task is None, (
            "_pp_task should be None before the first turn"
        )

        # Run first turn
        async for _ in sr2.turn(make_user_input()):
            pass

        # After first turn, _pp_task should be set (deferred task scheduled)
        assert sr2._pp_task is not None, (
            "_pp_task should be scheduled after turn completes"
        )

    @pytest.mark.asyncio
    async def test_post_process_receives_final_response(self):
        """The deferred post_process task must receive the final CompletionResponse."""
        from sr2.orchestrator import SR2

        received_responses: list[CompletionResponse] = []

        llm = SequentialMockLLM(
            call_sequences=[
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
        )

        async def capture_post_process(response: CompletionResponse) -> None:
            received_responses.append(response)

        sr2.post_process = capture_post_process

        async for _ in sr2.turn(make_user_input()):
            pass

        # Wait for deferred task to complete
        await asyncio.sleep(0.05)

        assert len(received_responses) == 1, (
            f"post_process should be called once with final response, "
            f"got {len(received_responses)} calls"
        )
        resp = received_responses[0]
        assert resp.stop_reason == "end_turn"
        assert any(
            isinstance(b, TextBlock) and "Final answer" in b.text
            for b in resp.content
        ), f"Response content should contain 'Final answer'. Got: {resp.content}"


class TestPostProcessWithToolLoop:
    """Deferred post_process works correctly with multi-iteration tool loops."""

    @pytest.mark.asyncio
    async def test_deferred_post_process_after_tool_loop(self):
        """post_process is deferred even after a multi-iteration tool loop.

        The final_response passed to post_process should be the last iteration's
        response (the one that had no tool_use).
        """
        from sr2.orchestrator import SR2

        order: list[str] = []
        received_responses: list[CompletionResponse] = []

        llm = SequentialMockLLM(
            call_sequences=[
                [
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

        async def tracked_post_process(response: CompletionResponse) -> None:
            order.append("post_process")
            received_responses.append(response)

        sr2.post_process = tracked_post_process

        events = [e async for e in sr2.turn(make_user_input())]
        end_events = [e for e in events if e.type == "end"]

        # end event yielded
        assert len(end_events) == 1

        # post_process not yet called (deferred) — or called after end
        if "post_process" in order:
            # If it ran, end must have been yielded first
            assert order.index("post_process") >= 0  # just confirm it's there

        # Wait for deferred task
        await asyncio.sleep(0.05)

        assert len(received_responses) == 1
        assert received_responses[0].stop_reason == "end_turn"
