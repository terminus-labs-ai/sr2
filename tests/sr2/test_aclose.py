"""Tests for sr2-7: FR9 — SR2.aclose() lifecycle method.

Requirements:
  - async def aclose(): await any pending post_process task, surface errors per FR8.
  - Safe when no pending task (idempotent no-op).
  - After aclose(), _pp_task is None (cleared).
"""

from __future__ import annotations

import asyncio

import pytest

from sr2.config.models import PipelineConfig
from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
)


class TestAcloseNoPending:
    """aclose() is a no-op when no post_process task is pending."""

    @pytest.mark.asyncio
    async def test_aclose_no_pending_task(self):
        """aclose() on a fresh SR2 instance is a no-op."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": SequentialMockLLM(call_sequences=[[StreamEvent(type="text", text="x"), StreamEvent(type="end")]])},
            token_counter=CharacterTokenCounter(),
        )

        assert sr2._pp_task is None
        await sr2.aclose()
        assert sr2._pp_task is None

    @pytest.mark.asyncio
    async def test_aclose_idempotent(self):
        """Calling aclose() multiple times is safe."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": SequentialMockLLM(call_sequences=[[StreamEvent(type="text", text="x"), StreamEvent(type="end")]])},
            token_counter=CharacterTokenCounter(),
        )

        await sr2.aclose()
        await sr2.aclose()
        await sr2.aclose()
        assert sr2._pp_task is None


class TestAcloseWithPending:
    """aclose() awaits and clears a pending post_process task."""

    @pytest.mark.asyncio
    async def test_aclose_awaits_pending_task(self):
        """aclose() waits for the deferred post_process to complete."""
        from sr2.orchestrator import SR2

        pp_completed = asyncio.Event()

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

        async def tracked_pp(response: CompletionResponse) -> None:
            await asyncio.sleep(0.05)
            pp_completed.set()

        sr2.post_process = tracked_pp

        # Run one turn — schedules deferred post_process
        async for _ in sr2.turn(make_user_input()):
            pass

        assert sr2._pp_task is not None
        assert not pp_completed.is_set(), "post_process should not have completed yet"

        # aclose() must await the deferred task
        await sr2.aclose()

        assert pp_completed.is_set(), "aclose() must have awaited post_process"
        assert sr2._pp_task is None, "aclose() must clear _pp_task"

    @pytest.mark.asyncio
    async def test_aclose_surfaces_post_process_error(self):
        """aclose() propagates exceptions from the deferred post_process task."""
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

        async def failing_pp(response: CompletionResponse) -> None:
            raise RuntimeError("post_process failure")

        sr2.post_process = failing_pp

        async for _ in sr2.turn(make_user_input()):
            pass

        assert sr2._pp_task is not None

        with pytest.raises(RuntimeError, match="post_process failure"):
            await sr2.aclose()

        # Task should be cleared even after error
        assert sr2._pp_task is None


class TestAcloseAsContextManager:
    """aclose() serves as the explicit shutdown point."""

    @pytest.mark.asyncio
    async def test_aclose_after_turn_clears_state(self):
        """After aclose(), the instance is in a clean state."""
        from sr2.orchestrator import SR2

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

        async for _ in sr2.turn(make_user_input()):
            pass

        # Deferred task is pending
        assert sr2._pp_task is not None

        # Clean shutdown
        await sr2.aclose()

        # State is clean
        assert sr2._pp_task is None

        # Subsequent aclose() is a no-op
        await sr2.aclose()
        assert sr2._pp_task is None
