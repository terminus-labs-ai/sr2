"""Tests for sr2-5 / FR7: surface in-band bus errors as StreamEvent before end.

Requirements:
  FR7 — Errors collected on EventBus during in-band drains (start_turn + each
        continue_turn) must be yielded as a single StreamEvent(type="error",
        errors=[...]) immediately before the final "end" event. Non-fatal.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sr2.models import ToolResultBlock, ToolUseBlock
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
    tool_use_event,
)


class ErroringTransformer:
    """A transformer that raises on process_pending, injecting errors into the bus."""

    name = "erroring_transformer"
    execution_count = 0

    def __init__(self, raise_on: str = "process") -> None:
        self.raise_on = raise_on

    async def process_pending(self, bus: EventBus, event: Event) -> bool:
        raise RuntimeError("transformer boom")

    def compile(self) -> list:
        return []


class TestBusErrorSurfacing:
    """FR7: In-band bus errors are surfaced as StreamEvent(type='error') before 'end'."""

    @pytest.mark.asyncio
    async def test_no_errors_produces_no_error_event(self):
        """Clean turn with no bus errors must NOT emit an error event."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Hello."),
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
        error_events = [e for e in events if e.type == "error"]

        assert len(error_events) == 0, (
            f"No bus errors occurred, so no error event should be emitted. "
            f"Got {len(error_events)} error event(s)."
        )

    @pytest.mark.asyncio
    async def test_bus_error_during_start_turn_is_surfaced_before_end(self):
        """A callback error during start_turn drain must appear as an error event before 'end'."""
        from sr2.orchestrator import SR2

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

        # Register a broken sync callback on turn_start to inject a bus error.
        def broken_callback(event: Event) -> None:
            raise ValueError("turn_start callback failed")

        sr2._engine.bus.subscribe("turn_start", broken_callback)

        events = [e async for e in sr2.turn(make_user_input())]

        error_events = [e for e in events if e.type == "error"]
        end_events = [e for e in events if e.type == "end"]

        assert len(error_events) == 1, (
            f"Expected exactly 1 error event for bus error during start_turn, "
            f"got {len(error_events)}."
        )
        assert error_events[0].errors is not None
        assert any("turn_start callback failed" in err for err in error_events[0].errors), (
            f"Error message should contain the original exception. Got: {error_events[0].errors}"
        )

        # Error event must come before the end event
        error_idx = next(i for i, e in enumerate(events) if e.type == "error")
        end_idx = next(i for i, e in enumerate(events) if e.type == "end")
        assert error_idx < end_idx, (
            "Error event must appear before the final 'end' event."
        )

    @pytest.mark.asyncio
    async def test_bus_error_during_continue_turn_is_surfaced(self):
        """A callback error during continue_turn drain must be surfaced."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "get_weather"),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Weather is sunny."),
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

        # Register a broken callback on tool_result (fired during continue_turn).
        def broken_tool_result_callback(event: Event) -> None:
            raise RuntimeError("tool_result handler crashed")

        sr2._engine.bus.subscribe("tool_result", broken_tool_result_callback)

        events = [e async for e in sr2.turn(make_user_input())]

        error_events = [e for e in events if e.type == "error"]

        assert len(error_events) == 1, (
            f"Expected 1 error event for bus error during continue_turn, "
            f"got {len(error_events)}."
        )
        assert error_events[0].errors is not None
        assert any("tool_result handler crashed" in err for err in error_events[0].errors), (
            f"Error should reference the tool_result callback. Got: {error_events[0].errors}"
        )

    @pytest.mark.asyncio
    async def test_multiple_bus_errors_collected_into_single_event(self):
        """Multiple bus errors across drains must be collected into one error event."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    tool_use_event("call_001", "search"),
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

        # Two broken callbacks on different events.
        def broken_start(event: Event) -> None:
            raise ValueError("start error")

        def broken_tool_result(event: Event) -> None:
            raise ValueError("tool_result error")

        sr2._engine.bus.subscribe("turn_start", broken_start)
        sr2._engine.bus.subscribe("tool_result", broken_tool_result)

        events = [e async for e in sr2.turn(make_user_input())]

        error_events = [e for e in events if e.type == "error"]

        assert len(error_events) == 1, (
            f"Multiple bus errors should be collected into a single error event, "
            f"got {len(error_events)}."
        )
        all_errors = error_events[0].errors or []
        assert len(all_errors) >= 2, (
            f"Expected at least 2 error strings, got {len(all_errors)}: {all_errors}"
        )
        error_texts = " ".join(all_errors)
        assert "start error" in error_texts
        assert "tool_result error" in error_texts

    @pytest.mark.asyncio
    async def test_error_event_does_not_abort_turn(self):
        """Bus errors are non-fatal — the turn still completes with an 'end' event."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Response despite errors."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        def broken_callback(event: Event) -> None:
            raise ValueError("non-fatal error")

        sr2._engine.bus.subscribe("turn_start", broken_callback)

        events = [e async for e in sr2.turn(make_user_input())]

        end_events = [e for e in events if e.type == "end"]
        text_events = [e for e in events if e.type == "text"]

        assert len(end_events) == 1, (
            "Turn must still emit exactly one 'end' event despite bus errors."
        )
        assert len(text_events) > 0, (
            "Text events should still be streamed despite bus errors."
        )
        assert events[-1].type == "end", (
            "The final event must still be 'end' even with bus errors."
        )
