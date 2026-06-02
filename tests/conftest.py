"""Shared test helpers for the entire tests/ tree.

Plain functions and classes — not pytest fixtures. Test files that use these
must import them explicitly:

    from conftest import MockLLM, make_minimal_config, make_user_input
    from conftest import SequentialMockLLM, tool_use_event, stub_executor
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    StreamEvent,
)


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
# sr2/ subdir helpers — also shared here to avoid conftest naming conflicts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PipelineEngine.run() replacement (sr2-8: engine.run() removed)
# ---------------------------------------------------------------------------


async def run_engine(
    engine: PipelineEngine,
    user_input: list | None = None,
) -> "PipelineResult":
    """Replacement for ``PipelineEngine.run()``.

    Uses the three-phase API (start_turn / continue_turn / end_turn) to
    replicate the old ``engine.run(user_input)`` behaviour:
    1. ``start_turn`` with the next auto-incremented turn sequence.
    2. If *user_input* is non-empty, queue a ``user_input`` event and drain.
    3. ``end_turn`` — return the compiled ``PipelineResult``.
    """
    next_seq = engine._turn_seq + 1
    await engine.start_turn(turn_seq=next_seq)

    if user_input:
        user_event = Event(
            name="user_input",
            phase=EventPhase.COMPLETED,
            source_layer="engine",
            data=user_input,
        )
        await engine.continue_turn(events=[user_event], iteration_seq=0)

    return await engine.end_turn()
