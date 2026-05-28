"""Tests for sr2-30 + sr2-31: FR9 + FR10 — bus events for tool use and assistant responses.

FR9 (sr2-30):
  - ``tool_use_emitted`` queued on the engine bus per tool iteration.
    data = list[ToolUseBlock]; iteration_seq stamped on the event.
  - ``tool_result_received`` queued on the engine bus per tool iteration.
    data = list[ToolResultBlock]; iteration_seq stamped on the event.

FR10 (sr2-31):
  - ``assistant_response`` fires ONCE on the engine bus at end of turn()
    with the final CompletionResponse as data.
  - ``assistant_iteration_response`` fires ONCE PER ITERATION on the engine bus
    with the intermediate CompletionResponse as data.
  - Memory extractor stays subscribed only to ``assistant_response`` (unchanged).

Tests will FAIL until the feature is implemented (red phase).
"""

from __future__ import annotations

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
from sr2.pipeline.events import Event
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Helpers & fakes (mirrored from test_turn_loop.py patterns)
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
    """LLM that returns a different event list on each successive stream() call."""

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


def collect_bus_events(sr2_instance: Any, event_name: str) -> list[Event]:
    """Subscribe to the engine bus for event_name and return the collector list.

    Must be called BEFORE sr2.turn() to wire up subscription in time.
    """
    collected: list[Event] = []
    sr2_instance._engine.bus.subscribe(event_name, lambda e: collected.append(e))
    return collected


# ---------------------------------------------------------------------------
# FR9 — TestToolUseBusEvents
# ---------------------------------------------------------------------------


class TestToolUseBusEvents:
    """FR9: tool_use_emitted and tool_result_received must be queued on the engine bus."""

    @pytest.mark.asyncio
    async def test_tool_use_emitted_fires_with_tool_blocks(self):
        """After a turn with one tool_use, a ``tool_use_emitted`` bus event fires.

        The event's data must contain the ToolUseBlock(s) from that iteration.

        Will FAIL until tool_use_emitted is queued on the engine bus (currently
        the orchestrator only yields it as a StreamEvent, not a bus event).
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather", {"location": "Oslo"}),
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

        collected = collect_bus_events(sr2, "tool_use_emitted")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 1, (
            f"Expected 1 'tool_use_emitted' bus event for one tool iteration, "
            f"got {len(collected)}"
        )
        event = collected[0]
        # data must contain the ToolUseBlock(s) from the iteration
        assert event.data is not None, "tool_use_emitted event data must not be None"
        tool_blocks = event.data
        assert isinstance(tool_blocks, list), (
            f"tool_use_emitted data must be a list of ToolUseBlock, got {type(tool_blocks)}"
        )
        assert len(tool_blocks) == 1, f"Expected 1 ToolUseBlock, got {len(tool_blocks)}"
        assert isinstance(tool_blocks[0], ToolUseBlock), (
            f"Expected ToolUseBlock in data, got {type(tool_blocks[0])}"
        )
        assert tool_blocks[0].id == "call_001"
        assert tool_blocks[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_result_received_fires_with_result_blocks(self):
        """After a turn with one tool_use, a ``tool_result_received`` bus event fires.

        The event's data must contain the ToolResultBlock(s) from that iteration.

        Will FAIL until tool_result_received is queued on the engine bus.
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

        collected = collect_bus_events(sr2, "tool_result_received")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 1, (
            f"Expected 1 'tool_result_received' bus event for one tool iteration, "
            f"got {len(collected)}"
        )
        event = collected[0]
        assert event.data is not None, "tool_result_received event data must not be None"
        result_blocks = event.data
        assert isinstance(result_blocks, list), (
            f"tool_result_received data must be a list of ToolResultBlock, "
            f"got {type(result_blocks)}"
        )
        assert len(result_blocks) == 1, f"Expected 1 ToolResultBlock, got {len(result_blocks)}"
        assert isinstance(result_blocks[0], ToolResultBlock), (
            f"Expected ToolResultBlock in data, got {type(result_blocks[0])}"
        )
        assert result_blocks[0].tool_use_id == "call_001"
        assert result_blocks[0].content == "result_for_get_weather"

    @pytest.mark.asyncio
    async def test_tool_use_emitted_fires_twice_for_two_iterations(self):
        """With 2 tool iterations, ``tool_use_emitted`` must fire TWICE on the bus
        (once per iteration).

        Will FAIL until bus event emission is per-iteration.
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
                    StreamEvent(type="text", text="All done."),
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

        collected = collect_bus_events(sr2, "tool_use_emitted")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 2, (
            f"Expected 2 'tool_use_emitted' bus events for 2 tool iterations, "
            f"got {len(collected)}"
        )
        # Each event's data should contain the blocks from its own iteration
        first_blocks = collected[0].data
        second_blocks = collected[1].data
        assert isinstance(first_blocks, list) and len(first_blocks) == 1
        assert isinstance(second_blocks, list) and len(second_blocks) == 1
        assert first_blocks[0].name == "search_web"
        assert second_blocks[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_tool_result_received_fires_twice_for_two_iterations(self):
        """With 2 tool iterations, ``tool_result_received`` must fire TWICE on the bus
        (once per iteration).

        Will FAIL until bus event emission is per-iteration.
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
                    StreamEvent(type="text", text="All done."),
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

        collected = collect_bus_events(sr2, "tool_result_received")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 2, (
            f"Expected 2 'tool_result_received' bus events for 2 tool iterations, "
            f"got {len(collected)}"
        )
        # Each event's data should have the result from its own iteration
        first_results = collected[0].data
        second_results = collected[1].data
        assert isinstance(first_results, list) and len(first_results) == 1
        assert isinstance(second_results, list) and len(second_results) == 1
        assert first_results[0].content == "result_for_search_web"
        assert second_results[0].content == "result_for_read_file"


# ---------------------------------------------------------------------------
# FR9 — iteration_seq stamped on bus events
# ---------------------------------------------------------------------------


class TestIterationSeqStamping:
    """FR9: tool_use_emitted and tool_result_received must carry iteration_seq."""

    @pytest.mark.asyncio
    async def test_tool_use_emitted_first_iteration_has_seq_zero(self):
        """The first iteration's ``tool_use_emitted`` event must carry iteration_seq=0.

        Will FAIL until iteration_seq is stamped on bus events.
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

        collected = collect_bus_events(sr2, "tool_use_emitted")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 1
        event = collected[0]
        # iteration_seq must be stamped — could be stored on event.data as a
        # wrapper, or as a dedicated attribute. We check for a recognisable
        # iteration_seq marker on the event or its data dict.
        # Implementation may stamp it as: event.data = {"iteration_seq": 0, "blocks": [...]}
        # OR as a subclass field, OR by using a named dataclass.
        # We test the minimal contract: iteration_seq=0 is accessible.
        iteration_seq = _extract_iteration_seq(event)
        assert iteration_seq == 0, (
            f"Expected iteration_seq=0 on first tool_use_emitted bus event, "
            f"got {iteration_seq!r}. "
            f"The event must carry iteration_seq (e.g. in event.data dict or a field)."
        )

    @pytest.mark.asyncio
    async def test_tool_use_emitted_second_iteration_has_seq_one(self):
        """The second iteration's ``tool_use_emitted`` event must carry iteration_seq=1.

        Will FAIL until iteration_seq is stamped on bus events.
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

        collected = collect_bus_events(sr2, "tool_use_emitted")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 2
        assert _extract_iteration_seq(collected[0]) == 0, (
            "First tool_use_emitted must have iteration_seq=0"
        )
        assert _extract_iteration_seq(collected[1]) == 1, (
            "Second tool_use_emitted must have iteration_seq=1"
        )

    @pytest.mark.asyncio
    async def test_tool_result_received_carries_iteration_seq(self):
        """``tool_result_received`` bus events must also carry iteration_seq.

        Will FAIL until iteration_seq is stamped on tool_result_received bus events.
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

        collected = collect_bus_events(sr2, "tool_result_received")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 2
        assert _extract_iteration_seq(collected[0]) == 0, (
            "First tool_result_received must have iteration_seq=0"
        )
        assert _extract_iteration_seq(collected[1]) == 1, (
            "Second tool_result_received must have iteration_seq=1"
        )


def _extract_iteration_seq(event: Event) -> int | None:
    """Extract iteration_seq from an event regardless of how it's stored.

    Supports these plausible representations:
    - event.data = {"iteration_seq": N, "blocks": [...]}
    - event.data = SomeDataclass(iteration_seq=N, ...)
    - event has a dedicated .iteration_seq attribute
    """
    # Dict-based data
    if isinstance(event.data, dict) and "iteration_seq" in event.data:
        return event.data["iteration_seq"]
    # Attribute-based (dataclass/namedtuple/custom object)
    if hasattr(event.data, "iteration_seq"):
        return event.data.iteration_seq
    # Direct attribute on the event itself
    if hasattr(event, "iteration_seq"):
        return event.iteration_seq
    return None


# ---------------------------------------------------------------------------
# FR10 — TestAssistantResponseEvents
# ---------------------------------------------------------------------------


class TestAssistantResponseEvents:
    """FR10: assistant_response fires once; assistant_iteration_response per iteration."""

    @pytest.mark.asyncio
    async def test_assistant_response_fires_once_after_multi_iteration_turn(self):
        """``assistant_response`` must fire ONCE on the bus after a 2-iteration turn.

        It already fires once in the current implementation (lines 361-368 of
        orchestrator.py), so this test documents the existing contract and guards
        against regression from the FR10 refactor.
        """
        from sr2.orchestrator import SR2

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

        collected = collect_bus_events(sr2, "assistant_response")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 1, (
            f"Expected exactly 1 'assistant_response' bus event for a 2-iteration turn, "
            f"got {len(collected)}. "
            f"assistant_response must fire ONCE after the full turn, not per iteration."
        )

    @pytest.mark.asyncio
    async def test_assistant_iteration_response_fires_once_per_tool_iteration(self):
        """``assistant_iteration_response`` must fire ONCE PER tool iteration.

        Two tool iterations → two ``assistant_iteration_response`` events.

        Will FAIL until assistant_iteration_response is implemented and queued
        on the engine bus per iteration.
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
                    StreamEvent(type="text", text="All done."),
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

        collected = collect_bus_events(sr2, "assistant_iteration_response")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 2, (
            f"Expected 2 'assistant_iteration_response' bus events (one per tool iteration), "
            f"got {len(collected)}. "
            f"assistant_iteration_response fires at the end of each tool iteration, "
            f"NOT at the end of the turn."
        )

    @pytest.mark.asyncio
    async def test_assistant_iteration_response_does_not_fire_for_text_only_turn(self):
        """For a text-only turn (no tool_use), ``assistant_iteration_response``
        must NOT fire — there are no tool iterations.

        Will FAIL if assistant_iteration_response is incorrectly emitted for
        non-tool turns.
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
        )

        collected = collect_bus_events(sr2, "assistant_iteration_response")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 0, (
            f"Expected 0 'assistant_iteration_response' events for a text-only turn "
            f"(no tool iterations), got {len(collected)}."
        )

    @pytest.mark.asyncio
    async def test_assistant_response_data_is_final_completion_response(self):
        """The ``assistant_response`` bus event's data must be the final
        CompletionResponse (the one with stop_reason='end_turn', no tool_use blocks).

        Will FAIL if the wrong response is attached (e.g., an intermediate one).
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="The final answer."),
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

        collected = collect_bus_events(sr2, "assistant_response")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 1
        event = collected[0]
        assert isinstance(event.data, CompletionResponse), (
            f"assistant_response event.data must be CompletionResponse, "
            f"got {type(event.data)}"
        )
        response: CompletionResponse = event.data
        assert response.stop_reason == "end_turn", (
            f"Final CompletionResponse must have stop_reason='end_turn', "
            f"got {response.stop_reason!r}"
        )
        # Final response must NOT contain any ToolUseBlock — it's the terminal response
        tool_use_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_use_blocks) == 0, (
            f"Final CompletionResponse must not contain ToolUseBlocks (it's the terminal "
            f"text response). Found {len(tool_use_blocks)} ToolUseBlock(s)."
        )

    @pytest.mark.asyncio
    async def test_assistant_iteration_response_data_contains_intermediate_response(self):
        """``assistant_iteration_response`` data must be the intermediate CompletionResponse
        from that iteration (the one that triggered tool calls, NOT the final response).

        Will FAIL until assistant_iteration_response is implemented with the
        intermediate response attached as data.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Let me check the weather."),
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

        iteration_collected = collect_bus_events(sr2, "assistant_iteration_response")
        final_collected = collect_bus_events(sr2, "assistant_response")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(iteration_collected) == 1, (
            f"Expected 1 assistant_iteration_response event, got {len(iteration_collected)}"
        )
        assert len(final_collected) == 1, (
            f"Expected 1 assistant_response event, got {len(final_collected)}"
        )

        iter_response: CompletionResponse = iteration_collected[0].data
        final_response: CompletionResponse = final_collected[0].data

        assert isinstance(iter_response, CompletionResponse), (
            f"assistant_iteration_response data must be CompletionResponse, "
            f"got {type(iter_response)}"
        )
        # Intermediate response must contain a ToolUseBlock — it caused the tool call
        tool_use_blocks = [b for b in iter_response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_use_blocks) >= 1, (
            f"assistant_iteration_response data must be the intermediate response "
            f"containing ToolUseBlock(s), got content: {iter_response.content!r}"
        )

        # The intermediate response must be DIFFERENT from the final response
        # (they represent different LLM calls)
        assert iter_response is not final_response, (
            "assistant_iteration_response and assistant_response must carry "
            "different CompletionResponse objects"
        )
        # Final response has no tool_use blocks
        final_tool_blocks = [b for b in final_response.content if isinstance(b, ToolUseBlock)]
        assert len(final_tool_blocks) == 0, (
            "assistant_response (final) must not contain ToolUseBlocks"
        )

    @pytest.mark.asyncio
    async def test_assistant_iteration_response_does_not_fire_for_final_iteration(self):
        """``assistant_iteration_response`` must NOT include the final LLM call
        (the one that returns text only, ending the loop).

        Single tool iteration: assistant_iteration_response fires ONCE (for the
        tool-use iteration). The final text-only LLM call is covered only by
        assistant_response, not assistant_iteration_response.

        Will FAIL if the implementation incorrectly fires assistant_iteration_response
        for the terminal (no-tool) LLM call.
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

        collected = collect_bus_events(sr2, "assistant_iteration_response")
        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(collected) == 1, (
            f"Expected exactly 1 'assistant_iteration_response' for a single-tool-iteration "
            f"turn. The final text-only LLM call must NOT fire assistant_iteration_response. "
            f"Got {len(collected)} events."
        )


# ---------------------------------------------------------------------------
# TestBusEventSubscription — integration: subscribe → turn → verify
# ---------------------------------------------------------------------------


class TestBusEventSubscription:
    """Integration tests: wire subscribers before turn(), run turn(), verify delivery."""

    @pytest.mark.asyncio
    async def test_subscriber_to_tool_use_emitted_receives_event(self):
        """A subscriber wired before turn() must receive the tool_use_emitted bus event.

        This is the canonical integration pattern: subscribe → turn → assert.

        Will FAIL until tool_use_emitted is queued on the engine bus.
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

        # Subscribe BEFORE turn()
        received: list[Event] = []
        sr2._engine.bus.subscribe("tool_use_emitted", lambda e: received.append(e))

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(received) >= 1, (
            f"Subscriber registered before turn() must receive tool_use_emitted events. "
            f"Got {len(received)} events. "
            f"tool_use_emitted must be queued on the engine bus (not just streamed)."
        )

    @pytest.mark.asyncio
    async def test_subscriber_to_assistant_response_called_after_all_tool_iterations(self):
        """A subscriber to ``assistant_response`` must be called only AFTER all tool
        iterations complete — i.e., after the final LLM call.

        We verify this by checking that all tool_use_emitted events fired before
        assistant_response fires.

        The existing implementation queues assistant_response after the loop (line 361),
        so this should pass if the subscription wiring is correct. This test documents
        and guards that ordering contract.
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

        event_order: list[str] = []

        def on_tool_use_emitted(e: Event) -> None:
            event_order.append("tool_use_emitted")

        def on_assistant_response(e: Event) -> None:
            event_order.append("assistant_response")

        sr2._engine.bus.subscribe("tool_use_emitted", on_tool_use_emitted)
        sr2._engine.bus.subscribe("assistant_response", on_assistant_response)

        async for _ in sr2.turn(make_user_input()):
            pass

        # assistant_response must appear after all tool_use_emitted events
        assert "assistant_response" in event_order, (
            "assistant_response must fire at least once"
        )
        # Find the index of the last tool_use_emitted and the first assistant_response
        tool_emitted_indices = [i for i, e in enumerate(event_order) if e == "tool_use_emitted"]
        assistant_idx = event_order.index("assistant_response")

        if tool_emitted_indices:
            last_tool_idx = max(tool_emitted_indices)
            assert assistant_idx > last_tool_idx, (
                f"assistant_response (at index {assistant_idx}) must fire AFTER all "
                f"tool_use_emitted events (last at index {last_tool_idx}). "
                f"Actual order: {event_order}"
            )
