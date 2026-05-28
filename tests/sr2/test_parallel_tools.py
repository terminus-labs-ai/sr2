"""Tests for sr2-27: FR4+FR15 — parallel tool execution with concurrency cap.

Requirements:
  FR4  — When the LLM returns multiple tool_use blocks in a single response,
         execute them concurrently via asyncio.gather.
  FR15 — PipelineConfig.max_parallel_tools: int | None = None — when set,
         wrap each executor call in an asyncio.Semaphore to cap concurrency.

  Results must be ordered to match the original ToolUseBlock order regardless
  of completion order.

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
# Helpers & fakes  (mirrors test_turn_loop.py patterns)
# ---------------------------------------------------------------------------


def make_user_input(text: str = "Hello") -> list:
    return [TextBlock(text=text)]


def make_minimal_config(max_parallel_tools: int | None = None) -> PipelineConfig:
    """Build a minimal two-layer PipelineConfig.

    max_parallel_tools is passed through once PipelineConfig gains the field.
    Until then the field is not present and tests targeting FR15 will fail at
    construction time (AttributeError) or at runtime — both are acceptable
    red-phase failures.
    """
    kwargs: dict[str, Any] = {
        "layers": [
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
    }
    if max_parallel_tools is not None:
        kwargs["max_parallel_tools"] = max_parallel_tools
    return PipelineConfig(**kwargs)


class SequentialMockLLM:
    """LLM that returns a different event list on each successive stream() call.

    Constructed with a list of call-sequences. Each element is the list of
    StreamEvents returned on that call (0-indexed).  Once all sequences are
    exhausted, subsequent calls repeat the last sequence.
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
    """Minimal executor that returns a synthetic result immediately."""
    return ToolResultBlock(
        tool_use_id=block.id,
        content=f"result_for_{block.name}",
    )


# ---------------------------------------------------------------------------
# FR4 — Parallel execution: multiple tool_use blocks in one LLM response
# ---------------------------------------------------------------------------


class TestParallelToolExecution:
    """FR4: Multiple tool_use blocks in a single LLM response are executed concurrently."""

    @pytest.mark.asyncio
    async def test_two_tool_blocks_both_executors_called(self):
        """Two tool_use blocks in one LLM response → executor called for both.

        The call log must have exactly 2 entries after the turn completes.
        Will FAIL until parallel execution is implemented (currently only the
        first block's executor is called because they run sequentially one-at-a-time
        — or rather, the test verifies that two blocks in the SAME response
        triggers two executor invocations).
        """
        from sr2.orchestrator import SR2

        call_log: list[ToolUseBlock] = []

        async def recording_executor(block: ToolUseBlock) -> ToolResultBlock:
            call_log.append(block)
            return ToolResultBlock(tool_use_id=block.id, content=f"result_{block.name}")

        # Single LLM response with TWO tool_use blocks, then a follow-up text.
        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "search_web", {"query": "foo"}),
                    tool_use_event("call_002", "read_file", {"path": "/tmp/bar"}),
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
            tool_executor=recording_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(call_log) == 2, (
            f"Expected executor called twice (once per tool_use block in the same "
            f"LLM response), got {len(call_log)} calls. "
            f"Both tool_use blocks must be executed."
        )
        ids = {b.id for b in call_log}
        assert ids == {"call_001", "call_002"}, (
            f"Expected executor called for both block IDs, got: {ids}"
        )

    @pytest.mark.asyncio
    async def test_three_tool_blocks_all_executors_called(self):
        """Three tool_use blocks in one response → all three executed.

        Results must all appear in the tool_result message sent back to the LLM.
        """
        from sr2.orchestrator import SR2

        call_log: list[ToolUseBlock] = []

        async def recording_executor(block: ToolUseBlock) -> ToolResultBlock:
            call_log.append(block)
            return ToolResultBlock(tool_use_id=block.id, content=f"result_{block.name}")

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "tool_a"),
                    tool_use_event("call_002", "tool_b"),
                    tool_use_event("call_003", "tool_c"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="All three done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=recording_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(call_log) == 3, (
            f"Expected executor called three times, got {len(call_log)}"
        )
        ids = {b.id for b in call_log}
        assert ids == {"call_001", "call_002", "call_003"}, (
            f"Not all three blocks were executed. Got IDs: {ids}"
        )

    @pytest.mark.asyncio
    async def test_results_returned_in_original_block_order(self):
        """Results must map to ToolUseBlock IDs in original order, regardless of
        which executor finished first.

        We engineer block[2] (call_003) to finish before block[0] (call_001)
        by using asyncio.sleep to delay call_001. The tool_result message sent
        to the LLM must list results in the order: call_001, call_002, call_003.

        Verification: inspect the second LLM call's messages — the user message
        containing ToolResultBlock entries must be ordered to match the original
        ToolUseBlock sequence.
        """
        from sr2.orchestrator import SR2

        # Delays: call_001 is slowest, call_003 is fastest.
        delays = {"call_001": 0.05, "call_002": 0.02, "call_003": 0.0}

        async def slow_executor(block: ToolUseBlock) -> ToolResultBlock:
            await asyncio.sleep(delays.get(block.id, 0))
            return ToolResultBlock(tool_use_id=block.id, content=f"result_{block.id}")

        captured_requests: list[CompletionRequest] = []

        class CapturingLLM:
            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                if len(captured_requests) == 1:
                    yield tool_use_event("call_001", "slow_tool")
                    yield tool_use_event("call_002", "mid_tool")
                    yield tool_use_event("call_003", "fast_tool")
                    yield StreamEvent(type="end")
                else:
                    yield StreamEvent(type="text", text="All results received.")
                    yield StreamEvent(type="end")

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": CapturingLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=slow_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(captured_requests) == 2, "LLM must be called twice"

        # The second request has messages: [system?, user_input, assistant (tool_use), user (tool_result)]
        second_request = captured_requests[1]
        # Find the user message that contains ToolResultBlocks
        tool_result_msg = None
        for msg in second_request.messages:
            if msg.role == "user" and any(
                hasattr(b, "tool_use_id") for b in msg.content
            ):
                tool_result_msg = msg
                break

        assert tool_result_msg is not None, (
            "Second LLM call must include a user message with ToolResultBlocks"
        )

        result_ids = [
            b.tool_use_id
            for b in tool_result_msg.content
            if hasattr(b, "tool_use_id")
        ]
        assert result_ids == ["call_001", "call_002", "call_003"], (
            f"ToolResultBlocks must be ordered to match original ToolUseBlock sequence. "
            f"Expected ['call_001', 'call_002', 'call_003'], got {result_ids}. "
            f"Completion order (fast_003, mid_002, slow_001) must NOT determine result order."
        )

    @pytest.mark.asyncio
    async def test_two_tools_execute_concurrently(self):
        """Verify that two executors actually overlap (both start before either finishes).

        Approach: executor_1 waits for executor_2_started before returning.
        With sequential execution, executor_2 never starts while executor_1
        blocks → asyncio.wait_for raises TimeoutError → orchestrator catches it
        and wraps it as ToolResultBlock(is_error=True). The second LLM call
        then receives an error result for call_001.

        With concurrent (gather) execution, executor_2 starts immediately →
        executor_2_started is set → executor_1 unblocks → both complete cleanly
        with no error results.

        This test will FAIL with the current sequential implementation because
        the tool_result message will contain is_error=True for call_001.
        """
        from sr2.orchestrator import SR2

        executor_2_started = asyncio.Event()
        captured_requests: list[CompletionRequest] = []

        async def dispatching_executor(block: ToolUseBlock) -> ToolResultBlock:
            if block.id == "call_001":
                # Block until executor_2 has started — impossible if sequential.
                await asyncio.wait_for(executor_2_started.wait(), timeout=0.5)
                return ToolResultBlock(tool_use_id=block.id, content="result_1")
            else:
                executor_2_started.set()
                return ToolResultBlock(tool_use_id=block.id, content="result_2")

        class CapturingLLM:
            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                if len(captured_requests) == 1:
                    yield tool_use_event("call_001", "slow_tool")
                    yield tool_use_event("call_002", "fast_tool")
                    yield StreamEvent(type="end")
                else:
                    yield StreamEvent(type="text", text="Both done.")
                    yield StreamEvent(type="end")

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": CapturingLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=dispatching_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        # With sequential: call_001 times out → is_error=True in the second LLM call.
        # With concurrent (gather): both succeed → no error results.
        assert len(captured_requests) == 2, "LLM must be called twice"
        second_request = captured_requests[1]
        error_results = [
            b
            for msg in second_request.messages
            for b in msg.content
            if hasattr(b, "is_error") and b.is_error
        ]
        assert not error_results, (
            f"Found {len(error_results)} error tool result(s) in the second LLM call. "
            f"With concurrent execution, call_001 should unblock immediately when "
            f"call_002 starts and sets executor_2_started. "
            f"Errors indicate sequential execution where call_001 timed out. "
            f"Error content: {[b.content for b in error_results]}"
        )


# ---------------------------------------------------------------------------
# FR15 — Concurrency cap via max_parallel_tools
# ---------------------------------------------------------------------------


class TestConcurrencyCap:
    """FR15: PipelineConfig.max_parallel_tools caps the number of simultaneous executors."""

    @pytest.mark.asyncio
    async def test_max_parallel_tools_1_executes_sequentially(self):
        """max_parallel_tools=1 → tools run one at a time.

        Proof: with cap=1, executor_2 cannot start until executor_1 finishes.
        We detect this with exec_2_started_before_1_finished.

        This test will FAIL if max_parallel_tools=1 is not yet implemented,
        because without a semaphore the default (gather) will run them
        concurrently and exec_2 starts before exec_1 finishes.
        """
        from sr2.orchestrator import SR2

        exec_1_finished = asyncio.Event()
        overlap_detected = False

        async def ordered_executor(block: ToolUseBlock) -> ToolResultBlock:
            nonlocal overlap_detected
            if block.id == "call_001":
                await asyncio.sleep(0.05)
                exec_1_finished.set()
                return ToolResultBlock(tool_use_id=block.id, content="ok_1")
            elif block.id == "call_002":
                # With cap=1 this runs after call_001. With no cap, call_001 is
                # still sleeping → exec_1_finished not yet set → overlap detected.
                if not exec_1_finished.is_set():
                    overlap_detected = True
                return ToolResultBlock(tool_use_id=block.id, content="ok_2")
            else:
                return ToolResultBlock(tool_use_id=block.id, content="ok_3")

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "tool_a"),
                    tool_use_event("call_002", "tool_b"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(max_parallel_tools=1),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=ordered_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert not overlap_detected, (
            "With max_parallel_tools=1, executor_2 must NOT start until executor_1 "
            "finishes. Overlap was detected — the semaphore is not enforcing cap=1."
        )

    @pytest.mark.asyncio
    async def test_max_parallel_tools_2_caps_at_two_concurrent(self):
        """max_parallel_tools=2 with 3 tools → at least 2 run simultaneously.

        Proof: two-way barrier between call_001 and call_002.
        - With gather + Semaphore(2): both start together → barrier clears → no errors.
        - With sequential (no gather): call_001 times out waiting for call_002 → is_error.
        - With Semaphore(1) (cap=1): same deadlock → is_error.

        Checks no error results in the second LLM call. Will FAIL until
        max_parallel_tools=2 is implemented with asyncio.gather + Semaphore(2).
        """
        from sr2.orchestrator import SR2

        slot_a_started = asyncio.Event()
        slot_b_started = asyncio.Event()
        captured_requests: list[CompletionRequest] = []

        async def cap2_executor(block: ToolUseBlock) -> ToolResultBlock:
            if block.id == "call_001":
                slot_a_started.set()
                await asyncio.wait_for(slot_b_started.wait(), timeout=0.5)
                return ToolResultBlock(tool_use_id=block.id, content="ok_1")
            elif block.id == "call_002":
                slot_b_started.set()
                await asyncio.wait_for(slot_a_started.wait(), timeout=0.5)
                return ToolResultBlock(tool_use_id=block.id, content="ok_2")
            else:
                return ToolResultBlock(tool_use_id=block.id, content="ok_3")

        class CapturingLLM:
            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                if len(captured_requests) == 1:
                    yield tool_use_event("call_001", "tool_a")
                    yield tool_use_event("call_002", "tool_b")
                    yield tool_use_event("call_003", "tool_c")
                    yield StreamEvent(type="end")
                else:
                    yield StreamEvent(type="text", text="Done.")
                    yield StreamEvent(type="end")

        sr2 = SR2(
            pipeline_config=make_minimal_config(max_parallel_tools=2),
            llm={"default": CapturingLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=cap2_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(captured_requests) == 2, "LLM must be called twice"
        second_request = captured_requests[1]
        error_results = [
            b
            for msg in second_request.messages
            for b in msg.content
            if hasattr(b, "is_error") and b.is_error
        ]
        assert not error_results, (
            f"Found {len(error_results)} error result(s). "
            f"With max_parallel_tools=2, call_001 and call_002 must overlap. "
            f"Errors indicate sequential or Semaphore(1) execution. "
            f"Error content: {[b.content for b in error_results]}"
        )

    @pytest.mark.asyncio
    async def test_max_parallel_tools_none_all_run_concurrently(self):
        """max_parallel_tools=None (default) → no cap, all 3 tools run simultaneously.

        Proof: three-way barrier — each executor waits for all three to start.
        With no cap (gather), all three start immediately → all wait → all clear.
        With sequential execution, exec_1 waits for exec_2 and exec_3 to start,
        but they never start → TimeoutError → orchestrator wraps as is_error=True.

        The test verifies no error results appear in the second LLM call.
        This test will FAIL with the current sequential implementation.
        """
        from sr2.orchestrator import SR2

        started = [asyncio.Event() for _ in range(3)]
        all_started = asyncio.Event()
        id_to_idx = {"call_001": 0, "call_002": 1, "call_003": 2}
        captured_requests: list[CompletionRequest] = []

        async def barrier_executor(block: ToolUseBlock) -> ToolResultBlock:
            idx = id_to_idx[block.id]
            started[idx].set()
            if all(e.is_set() for e in started):
                all_started.set()
            # All three must be running simultaneously for this to clear.
            await asyncio.wait_for(all_started.wait(), timeout=0.5)
            return ToolResultBlock(tool_use_id=block.id, content="ok")

        class CapturingLLM:
            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                if len(captured_requests) == 1:
                    yield tool_use_event("call_001", "tool_a")
                    yield tool_use_event("call_002", "tool_b")
                    yield tool_use_event("call_003", "tool_c")
                    yield StreamEvent(type="end")
                else:
                    yield StreamEvent(type="text", text="Done.")
                    yield StreamEvent(type="end")

        # Default config: no max_parallel_tools
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": CapturingLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=barrier_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(captured_requests) == 2, "LLM must be called twice"
        second_request = captured_requests[1]
        error_results = [
            b
            for msg in second_request.messages
            for b in msg.content
            if hasattr(b, "is_error") and b.is_error
        ]
        assert not error_results, (
            f"Found {len(error_results)} error tool result(s) — barriers timed out. "
            f"With max_parallel_tools=None, all three executors must start concurrently "
            f"so the three-way barrier clears without timeout. "
            f"Errors indicate sequential execution. "
            f"Error content: {[b.content for b in error_results]}"
        )

    @pytest.mark.asyncio
    async def test_max_parallel_tools_1_second_does_not_start_before_first_finishes(self):
        """Explicit proof: with cap=1, executor-2 hasn't started when executor-1 is running.

        We use asyncio.Event to verify that executor-2 starts AFTER executor-1
        has set its "finished" event.
        """
        from sr2.orchestrator import SR2

        exec_1_finished = asyncio.Event()
        exec_2_started_before_1_finished = False

        async def ordered_executor(block: ToolUseBlock) -> ToolResultBlock:
            nonlocal exec_2_started_before_1_finished
            if block.id == "call_001":
                await asyncio.sleep(0.03)
                exec_1_finished.set()
                return ToolResultBlock(tool_use_id=block.id, content="result_1")
            else:
                # Record whether exec_1 had finished by the time we started
                exec_2_started_before_1_finished = not exec_1_finished.is_set()
                return ToolResultBlock(tool_use_id=block.id, content="result_2")

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "slow_tool"),
                    tool_use_event("call_002", "fast_tool"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Done."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(max_parallel_tools=1),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=ordered_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert not exec_2_started_before_1_finished, (
            "With max_parallel_tools=1, executor-2 must NOT start before executor-1 "
            "has finished. The semaphore must enforce strict sequential execution."
        )

    @pytest.mark.asyncio
    async def test_concurrency_cap_field_exists_on_pipeline_config(self):
        """PipelineConfig must accept max_parallel_tools as an optional int field.

        This test verifies the config model change is in place.
        Will FAIL until max_parallel_tools is added to PipelineConfig.
        """
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "hi"})],
                )
            ],
            max_parallel_tools=3,
        )
        assert config.max_parallel_tools == 3, (
            "PipelineConfig.max_parallel_tools must be stored and retrievable"
        )

    @pytest.mark.asyncio
    async def test_concurrency_cap_field_defaults_to_none(self):
        """PipelineConfig.max_parallel_tools defaults to None (no cap).

        Will FAIL until the field is added to PipelineConfig.
        """
        config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "hi"})],
                )
            ],
        )
        assert config.max_parallel_tools is None, (
            "PipelineConfig.max_parallel_tools must default to None (no cap)"
        )

    @pytest.mark.asyncio
    async def test_results_preserved_in_order_with_cap_1(self):
        """Even with max_parallel_tools=1, results must be returned in block order.

        Sequential execution still must produce tool_result messages ordered to
        match the original ToolUseBlock sequence.
        """
        from sr2.orchestrator import SR2

        captured_requests: list[CompletionRequest] = []
        # Executor: fast_tool (call_002) is "slower" by name but executes first
        # with cap=1 (call_001 goes first because it's first in the block list).

        async def simple_executor(block: ToolUseBlock) -> ToolResultBlock:
            return ToolResultBlock(tool_use_id=block.id, content=f"result_{block.id}")

        class CapturingLLM:
            async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
                captured_requests.append(request)
                if len(captured_requests) == 1:
                    yield tool_use_event("call_001", "tool_a")
                    yield tool_use_event("call_002", "tool_b")
                    yield tool_use_event("call_003", "tool_c")
                    yield StreamEvent(type="end")
                else:
                    yield StreamEvent(type="text", text="Done.")
                    yield StreamEvent(type="end")

        sr2 = SR2(
            pipeline_config=make_minimal_config(max_parallel_tools=1),
            llm={"default": CapturingLLM()},
            token_counter=CharacterTokenCounter(),
            tool_executor=simple_executor,
        )

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(captured_requests) == 2
        second_request = captured_requests[1]

        # Find the tool_result user message
        tool_result_msg = None
        for msg in second_request.messages:
            if msg.role == "user" and any(
                hasattr(b, "tool_use_id") for b in msg.content
            ):
                tool_result_msg = msg
                break

        assert tool_result_msg is not None, "Must have a tool_result user message"

        result_ids = [
            b.tool_use_id
            for b in tool_result_msg.content
            if hasattr(b, "tool_use_id")
        ]
        assert result_ids == ["call_001", "call_002", "call_003"], (
            f"Results must be ordered to match original ToolUseBlock order even with "
            f"max_parallel_tools=1. Got: {result_ids}"
        )
