"""Tests for sr2-66: active_frame_provider injection through SR2().

The orchestrator has `_stamp_block()` which reads `self._active_frame_provider`,
but SR2.__init__ never accepted the parameter — it was always None.

These tests verify the fix: that active_frame_provider can be passed to SR2()
and is threaded into Dependencies, making frame stamping a real operation
instead of a permanent no-op.
"""

from __future__ import annotations

import pytest

from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent
from conftest import MockLLM, make_minimal_config, make_user_input


class TestActiveFrameProviderInjection:
    """P1: SR2 accepts active_frame_provider and threads it into Dependencies."""

    def test_sr2_accepts_active_frame_provider(self):
        """SR2(active_frame_provider=callable) constructs without error."""
        from sr2.orchestrator import SR2

        def provider(origin: str) -> str | None:
            return "test-frame"

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            active_frame_provider=provider,
        )

        assert sr2 is not None

    def test_provider_reaches_deps(self):
        """The provider passed to SR2() is stored as _active_frame_provider."""
        from sr2.orchestrator import SR2

        def provider(origin: str) -> str | None:
            return "test-frame"

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
            active_frame_provider=provider,
        )

        assert sr2._active_frame_provider is provider

    def test_no_provider_means_none(self):
        """SR2() without active_frame_provider → _active_frame_provider is None."""
        from sr2.orchestrator import SR2

        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": MockLLM()},
            token_counter=CharacterTokenCounter(),
        )

        assert sr2._active_frame_provider is None


class TestActiveFrameProviderStamping:
    """E2E: frame stamping produces meta['frame'] on emitted blocks."""

    @pytest.mark.asyncio
    async def test_text_block_has_frame_meta_when_provider_set(self):
        """When active_frame_provider is set, the final TextBlock carries meta['frame']."""
        import asyncio as _asyncio

        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event

        def provider(origin: str) -> str | None:
            return "plan:test-plan/task-1"

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="Hello"),
            StreamEvent(type="end"),
        ])
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
            active_frame_provider=provider,
        )

        captured: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: captured.append(e))

        async for _ in sr2.turn(make_user_input("test")):
            pass

        await _asyncio.sleep(0)

        assert len(captured) >= 1
        response = captured[0].data
        assert len(response.content) >= 1

        text_block = response.content[0]
        assert "frame" in text_block.meta
        assert text_block.meta["frame"] == "plan:test-plan/task-1"

    @pytest.mark.asyncio
    async def test_no_provider_means_no_frame_meta(self):
        """Without active_frame_provider, blocks have no frame in meta."""
        import asyncio as _asyncio

        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="Hello"),
            StreamEvent(type="end"),
        ])
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
        )

        captured: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: captured.append(e))

        async for _ in sr2.turn(make_user_input("test")):
            pass

        await _asyncio.sleep(0)

        assert len(captured) >= 1
        response = captured[0].data
        text_block = response.content[0]
        assert "frame" not in text_block.meta

    @pytest.mark.asyncio
    async def test_provider_receives_origin(self):
        """The provider is called with the origin string from turn()."""
        import asyncio as _asyncio

        from sr2.orchestrator import SR2
        from sr2.pipeline.events import Event

        call_log: list[str] = []

        def provider(origin: str) -> str | None:
            call_log.append(origin)
            return "ambient"

        mock_llm = MockLLM(events=[
            StreamEvent(type="text", text="Hi"),
            StreamEvent(type="end"),
        ])
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": mock_llm},
            token_counter=CharacterTokenCounter(),
            active_frame_provider=provider,
        )

        captured: list[Event] = []
        sr2._engine.bus.subscribe("assistant_response", lambda e: captured.append(e))

        async for _ in sr2.turn(make_user_input("test"), origin="discord:abc123"):
            pass

        await _asyncio.sleep(0)

        assert len(call_log) >= 1
        assert "discord:abc123" in call_log[0]

    @pytest.mark.asyncio
    async def test_tool_use_blocks_get_frame_stamped(self):
        """ToolUseBlocks created during the tool loop also get frame metadata."""
        from sr2.orchestrator import SR2
        from sr2.models import ToolResultBlock, ToolUseBlock

        stamped_blocks: list[ToolUseBlock] = []

        async def capturing_executor(block: ToolUseBlock) -> ToolResultBlock:
            stamped_blocks.append(block)
            return ToolResultBlock(tool_use_id=block.id, content="ok")

        tool_llm = MockLLM(
            events=[
                StreamEvent(
                    type="tool_use",
                    tool_use_id="tc_1",
                    tool_name="search",
                    tool_input={"query": "test"},
                ),
                StreamEvent(type="end"),
            ],
            follow_up_events=[
                StreamEvent(type="text", text="done"),
                StreamEvent(type="end"),
            ],
        )
        sr2 = SR2(
            pipeline_config=make_minimal_config(),
            llm={"default": tool_llm},
            token_counter=CharacterTokenCounter(),
            tool_executor=capturing_executor,
            active_frame_provider=lambda origin: "tool-frame",
        )

        async for _ in sr2.turn(make_user_input("run tool")):
            pass

        assert len(stamped_blocks) == 1
        assert "frame" in stamped_blocks[0].meta
        assert stamped_blocks[0].meta["frame"] == "tool-frame"
