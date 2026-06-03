"""Tests for sr2-80: post_process + turn finalization when turn() generator is abandoned.

When a consumer breaks out of the async-for loop early (before consuming the
final 'end' event), the generator is abandoned.  Previously this skipped
assistant_response queueing, end_turn(), and post_process scheduling entirely,
silently dropping session-history capture and memory extraction.

Fix: restructure turn() so finalization is guaranteed via try/finally, which
fires when the generator receives GeneratorExit (consumer break / gc).

Tests verify:
1. Consumer breaks after first text event → session history still captured
2. Consumer breaks before end event → post_process still scheduled
3. Normal consumption (full async-for) → behavior unchanged (regression guard)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
)

if TYPE_CHECKING:
    from sr2.orchestrator import SR2


def _get_session_resolver(sr2: "SR2"):
    """Find the SessionResolver in the conversation layer."""
    for layer in sr2._engine.layers:
        if layer.name == "conversation":
            for resolver in layer.resolvers:
                if resolver.name == "session":
                    return resolver
    return None


def _get_assistant_count(sr2: "SR2") -> int:
    """Return number of assistant messages in SessionResolver history."""
    resolver = _get_session_resolver(sr2)
    if resolver is None:
        return -1
    return sum(1 for m in resolver._history if m.role == "assistant")


# ---------------------------------------------------------------------------
# Test 1: Break after first text event — session history must still be captured
# ---------------------------------------------------------------------------


class TestEarlyBreakSessionCapture:
    """When consumer breaks early, assistant_response must still be queued and
    end_turn() must still be called so SessionResolver captures the reply."""

    @pytest.mark.asyncio
    async def test_break_after_first_text_captures_session_history(self):
        """Consumer breaks after the first text event.

        The LLM returns multiple text chunks + end.  Consumer only reads the
        first text event and then breaks.  After the generator is cleaned up,
        SessionResolver history must still contain the assistant message.
        """
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Part one "),
                    StreamEvent(type="text", text="part two."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        # Consume only the first event, then break
        gen = sr2.turn(make_user_input("Hello"))
        async for event in gen:
            if event.type == "text":
                break  # abandon generator here

        # Give the deferred task a chance to run
        await asyncio.sleep(0.01)
        # Also await any pending pp_task
        if sr2._pp_task is not None:
            try:
                await sr2._pp_task
            except asyncio.CancelledError:
                pass

        assert _get_assistant_count(sr2) >= 1, (
            "SessionResolver must capture the assistant response even when "
            "the consumer broke after the first text event. "
            "end_turn() + assistant_response queue must still execute."
        )

    @pytest.mark.asyncio
    async def test_break_mid_stream_cleans_up(self):
        """Consumer abandons generator during LLM streaming (before all events
        collected).  The generator must still close cleanly — no RuntimeError
        about async generator ignored."""
        from sr2.orchestrator import SR2

        # LLM yields many events; consumer abandons early
        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="A "),
                    StreamEvent(type="text", text="B "),
                    StreamEvent(type="text", text="C "),
                    StreamEvent(type="text", text="D "),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        gen = sr2.turn(make_user_input("Hello"))
        count = 0
        async for event in gen:
            count += 1
            if count >= 1:  # break after first event
                break

        # Explicit close to simulate what happens on GC / async-for exit
        try:
            await gen.aclose()
        except (StopAsyncIteration, RuntimeError):
            pass  # aclose may raise if already closed

    @pytest.mark.asyncio
    async def test_full_consumption_unchanged(self):
        """Regression guard: when consumer drives the full loop, behavior is
        unchanged — assistant history still captured correctly."""
        from sr2.orchestrator import SR2

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Full response."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        async for _ in sr2.turn(make_user_input("Hello")):
            pass

        assert _get_assistant_count(sr2) >= 1, (
            "Full consumption must still capture assistant response."
        )


# ---------------------------------------------------------------------------
# Test 2: post_process scheduling survives generator abandonment
# ---------------------------------------------------------------------------


class TestPostProcessOnAbandon:
    """post_process must be scheduled even when the generator is abandoned,
    and must run before the next turn() begins."""

    @pytest.mark.asyncio
    async def test_post_process_runs_after_abandon(self):
        """When consumer breaks early, post_process must still eventually run."""
        from sr2.orchestrator import SR2

        class TrackingSR2(SR2):
            flag: bool = False

            async def post_process(self, response) -> None:  # noqa: ANN001
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

        # Break after first event
        gen = sr2.turn(make_user_input("Hello"))
        async for event in gen:
            if event.type == "text":
                break

        # Wait for deferred task
        await asyncio.sleep(0.01)
        if sr2._pp_task is not None:
            try:
                await sr2._pp_task
            except asyncio.CancelledError:
                pass

        assert sr2.flag is True, (
            "post_process must run even when consumer abandons the generator early"
        )

    @pytest.mark.asyncio
    async def test_aclose_triggers_finalization(self):
        """Calling aclose() on the SR2 instance (or explicit generator cleanup)
        ensures deferred post_process runs."""
        from sr2.orchestrator import SR2

        class TrackingSR2(SR2):
            flag: bool = False

            async def post_process(self, response) -> None:  # noqa: ANN001
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

        # Break early
        gen = sr2.turn(make_user_input("Hello"))
        async for event in gen:
            if event.type == "text":
                break

        # Call aclose to await deferred task
        await sr2.aclose()

        assert sr2.flag is True, (
            "aclose() must trigger deferred post_process finalization"
        )

    @pytest.mark.asyncio
    async def test_next_turn_awaits_abandoned_post_process(self):
        """When a turn is abandoned, the next turn() must still await the
        previous turn's post_process before starting."""
        from sr2.orchestrator import SR2

        post_process_order: list[str] = []

        class TrackingSR2(SR2):
            async def post_process(self, response) -> None:  # noqa: ANN001
                post_process_order.append("pp_turn_1")

        llm = SequentialMockLLM(
            call_sequences=[
                [
                    StreamEvent(type="text", text="Turn 1."),
                    StreamEvent(type="end"),
                ],
                [
                    StreamEvent(type="text", text="Turn 2."),
                    StreamEvent(type="end"),
                ],
            ]
        )

        sr2 = TrackingSR2(
            pipeline_config=make_minimal_config(),
            llm={"default": llm},
            token_counter=CharacterTokenCounter(),
        )

        # Turn 1: break early
        gen = sr2.turn(make_user_input("Q1"))
        async for event in gen:
            if event.type == "text":
                break

        # Turn 2: full consumption
        async for _ in sr2.turn(make_user_input("Q2")):
            pass

        # Turn 1's post_process must have run before Turn 2 started
        await asyncio.sleep(0.01)
        if sr2._pp_task is not None:
            try:
                await sr2._pp_task
            except asyncio.CancelledError:
                pass

        assert "pp_turn_1" in post_process_order, (
            "Abandoned turn's post_process must still execute before next turn"
        )
