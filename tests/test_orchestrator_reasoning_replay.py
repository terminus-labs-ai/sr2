from __future__ import annotations

"""Tests for sr2-113 — in-turn reasoning replay in the orchestrator tool loop.

Covers AC 1, 3 and 4:
  (1) A tool-call iteration that streamed thinking puts that thinking as a
      ThinkingBlock, placed first, on the assistant message of the NEXT
      request; no ThinkingBlock when no thinking streamed.
  (2) pipeline.replay_reasoning defaults to True; with False the next
      request's messages are identical to today's (no ThinkingBlock, and
      _build_messages emits no reasoning_content key).
  (3) Session history and caller stream events are unchanged: the
      tool_use_emitted bus data stays ToolUseBlocks-only, the final
      assistant_response carries no ThinkingBlock, and no thinking event
      is yielded to the caller.
"""

import copy

import pytest

from conftest import make_minimal_config, make_user_input, stub_executor, tool_use_event
from sr2.config.models import PipelineConfig
from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, ThinkingBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent


def _make_sr2(llm, *, replay_reasoning: bool | None = None):
    config = make_minimal_config()
    if replay_reasoning is not None:
        config = config.model_copy(update={"replay_reasoning": replay_reasoning})
    return SR2(
        pipeline_config=config,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=stub_executor,
    )


def _thinking_tool_events() -> list[StreamEvent]:
    """Iteration 1: streamed thinking + a tool call."""
    return [
        StreamEvent(type="thinking", text="Let me think about this. "),
        StreamEvent(type="thinking", text="I should call a tool."),
        tool_use_event(tool_use_id="call_001"),
        StreamEvent(type="end"),
    ]


def _plain_text_events() -> list[StreamEvent]:
    """Iteration 2 (final): plain text, no thinking."""
    return [
        StreamEvent(type="text", text="The weather is fine."),
        StreamEvent(type="end"),
    ]


def _sequential_llm(thinking: bool):
    """A two-call SequentialMockLLM: thinking + tool_use on call 1, plain
    text on call 2."""
    from conftest import SequentialMockLLM

    if thinking:
        first = _thinking_tool_events()
    else:
        first = [tool_use_event(tool_use_id="call_001"), StreamEvent(type="end")]
    return SequentialMockLLM([first, _plain_text_events()])


def _assistant_messages(request) -> list[Message]:
    return [m for m in request.messages if m.role == "assistant"]


def _build_wire(sr2, request) -> list[dict]:
    return LiteLLMCallable("model")._build_messages(request)


# ---------------------------------------------------------------------------
# AC 3: flag default
# ---------------------------------------------------------------------------


class TestFlagDefault:
    def test_replay_reasoning_defaults_to_true(self):
        config = make_minimal_config()
        assert config.replay_reasoning is True

    def test_replay_reasoning_false_accepted(self):
        config = make_minimal_config().model_copy(update={"replay_reasoning": False})
        assert config.replay_reasoning is False


# ---------------------------------------------------------------------------
# AC 1: next request carries the iteration's thinking as ThinkingBlock
# ---------------------------------------------------------------------------


class TestNextRequestCarriesThinking:
    @pytest.mark.asyncio
    async def test_tool_iteration_thinking_replayed_in_next_request(self):
        llm = _sequential_llm(thinking=True)
        sr2 = _make_sr2(llm)

        async for _ in sr2.turn(make_user_input("How is the weather?")):
            pass

        assert len(llm.stream_calls) == 2
        second_request = llm.stream_calls[1]
        assistant_msgs = _assistant_messages(second_request)
        assert len(assistant_msgs) == 1
        content = assistant_msgs[0].content

        # ThinkingBlock is created and placed first.
        assert isinstance(content[0], ThinkingBlock)
        assert content[0].text == "Let me think about this. I should call a tool."
        # Followed by the tool-use block(s).
        assert any(isinstance(b, ToolUseBlock) for b in content)

    @pytest.mark.asyncio
    async def test_thinking_serialized_as_reasoning_content_on_wire(self):
        llm = _sequential_llm(thinking=True)
        sr2 = _make_sr2(llm)

        async for _ in sr2.turn(make_user_input()):
            pass

        wire = _build_wire(sr2, llm.stream_calls[1])
        assistant_msgs = [m for m in wire if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["reasoning_content"] == (
            "Let me think about this. I should call a tool."
        )
        # Never in visible content.
        assert "Let me think" not in (assistant_msgs[0].get("content") or "")

    @pytest.mark.asyncio
    async def test_no_thinking_no_thinking_block(self):
        llm = _sequential_llm(thinking=False)
        sr2 = _make_sr2(llm)

        async for _ in sr2.turn(make_user_input()):
            pass

        assert len(llm.stream_calls) == 2
        assistant_msgs = _assistant_messages(llm.stream_calls[1])
        assert len(assistant_msgs) == 1
        assert not any(isinstance(b, ThinkingBlock) for b in assistant_msgs[0].content)

        wire = _build_wire(sr2, llm.stream_calls[1])
        assistant_wire = [m for m in wire if m["role"] == "assistant"]
        assert "reasoning_content" not in assistant_wire[0]


# ---------------------------------------------------------------------------
# AC 3: flag off → messages identical to today
# ---------------------------------------------------------------------------


class TestReplayOff:
    @pytest.mark.asyncio
    async def test_replay_off_next_request_identical_to_today(self):
        llm_on = _sequential_llm(thinking=True)
        sr2_on = _make_sr2(llm_on)
        async for _ in sr2_on.turn(make_user_input()):
            pass

        llm_off = _sequential_llm(thinking=True)
        sr2_off = _make_sr2(llm_off, replay_reasoning=False)
        async for _ in sr2_off.turn(make_user_input()):
            pass

        # With replay on, the assistant message carries a ThinkingBlock;
        # with it off, the message is exactly today's shape.
        msg_on = _assistant_messages(llm_on.stream_calls[1])[0]
        msg_off = _assistant_messages(llm_off.stream_calls[1])[0]

        assert any(isinstance(b, ThinkingBlock) for b in msg_on.content)
        assert not any(isinstance(b, ThinkingBlock) for b in msg_off.content)

        # Identical to today: the replay-on message minus its ThinkingBlock
        # equals the replay-off message.
        content_on_without_thinking = [
            b for b in msg_on.content if not isinstance(b, ThinkingBlock)
        ]
        assert content_on_without_thinking == list(msg_off.content)

        # Wire format: no reasoning_content key anywhere.
        wire_off = _build_wire(sr2_off, llm_off.stream_calls[1])
        assert all("reasoning_content" not in m for m in wire_off)

    @pytest.mark.asyncio
    async def test_replay_off_thinking_still_not_forwarded(self):
        """Thinking events stay internal regardless of the flag."""
        llm = _sequential_llm(thinking=True)
        sr2 = _make_sr2(llm, replay_reasoning=False)

        events = [e async for e in sr2.turn(make_user_input())]

        assert not any(e.type == "thinking" for e in events)


# ---------------------------------------------------------------------------
# AC 4: session history and caller stream events unchanged
# ---------------------------------------------------------------------------


class TestHistoryAndCallerEventsUnchanged:
    @pytest.mark.asyncio
    async def test_tool_use_emitted_bus_data_stays_tool_use_blocks_only(self):
        llm = _sequential_llm(thinking=True)
        sr2 = _make_sr2(llm)
        captured: list = []
        sr2._engine.bus.subscribe(
            "tool_use_emitted", lambda e: captured.append(e.data)
        )

        async for _ in sr2.turn(make_user_input()):
            pass
        import asyncio

        await asyncio.sleep(0)

        assert len(captured) == 1
        data = captured[0]
        assert all(isinstance(b, ToolUseBlock) for b in data)
        assert not any(isinstance(b, ThinkingBlock) for b in data)

    @pytest.mark.asyncio
    async def test_final_assistant_response_carries_no_thinking_block(self):
        llm = _sequential_llm(thinking=True)
        sr2 = _make_sr2(llm)
        captured: list = []
        sr2._engine.bus.subscribe(
            "assistant_response", lambda e: captured.append(e.data)
        )

        async for _ in sr2.turn(make_user_input()):
            pass
        import asyncio

        await asyncio.sleep(0)

        assert len(captured) == 1
        response = captured[0]
        assert not any(isinstance(b, ThinkingBlock) for b in response.content)

    @pytest.mark.asyncio
    async def test_no_thinking_event_yielded_to_caller(self):
        llm = _sequential_llm(thinking=True)
        sr2 = _make_sr2(llm)

        events = [e async for e in sr2.turn(make_user_input())]

        assert not any(e.type == "thinking" for e in events)
        # Visible text is still streamed to the caller.
        assert "".join(e.text for e in events if e.type == "text") == (
            "The weather is fine."
        )

    @pytest.mark.asyncio
    async def test_final_no_tool_iteration_creates_no_thinking_block(self):
        """The final (no-tool) iteration never adds a ThinkingBlock, even
        when it streamed thinking and replay is on."""
        from conftest import SequentialMockLLM

        llm = SequentialMockLLM([
            _thinking_tool_events(),
            [
                StreamEvent(type="thinking", text="Final reasoning"),
                StreamEvent(type="text", text="Done."),
                StreamEvent(type="end"),
            ],
        ])
        sr2 = _make_sr2(llm)

        events = [e async for e in sr2.turn(make_user_input())]

        # No ThinkingBlock may be built from the final iteration: the only
        # request that carries a ThinkingBlock is the second one (replay of
        # iteration 1).
        for request in llm.stream_calls:
            for msg in _assistant_messages(request):
                thinking = [b for b in msg.content if isinstance(b, ThinkingBlock)]
                assert len(thinking) <= 1
                if thinking:
                    assert thinking[0].text == "Let me think about this. I should call a tool."
        # And nothing thinking-like reaches the caller or the final response.
        assert not any(e.type == "thinking" for e in events)
        assert "".join(e.text for e in events if e.type == "text") == "Done."
