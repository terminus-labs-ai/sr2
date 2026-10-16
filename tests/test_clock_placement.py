"""Tests for sr2-117 — clock placement in a messages-target layer (AC2).

Covers spec FR4–FR5 (sr2-spectre/agent-reminders): a ``static_text``
resolver placed between ``session`` and ``input`` in a ``messages``-target
layer compiles after the conversation history and before the new user
input, and the compiled system blocks stay byte-identical across two
turns with different clock values.

Placement mechanics (proven, not redesigned, by these tests):

* The orchestrator compiles the request ONCE per turn (turn()) and feeds
  later tool-loop iterations that same request extended with the
  assistant + tool_result messages. So "holds on every tool-loop
  iteration" reduces to the compile-time order holding in the layer.
* The clock subscribes to ``user_input`` (completed phase). The session
  resolver appends its history snapshot first, then the clock fires —
  resolver config order (session, clock, input) puts the clock block
  after history and before the input resolver's message.
* ``{time}`` renders at resolve time, so the clock carries the time of
  the compile; tool-loop iterations extend that request without
  recompiling, which keeps exactly one clock block per turn.

Two turns run through the real orchestrator (real engine, bus, and
compilation); only the LLM and the tool executor are stubbed. Each clock
fire renders a distinct value via the resolver's patchable date seam.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pytest

from conftest import (
    SequentialMockLLM,
    make_user_input,
    stub_executor,
    tool_use_event,
)
from sr2.config.models import (
    EventSubscriptionConfig,
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
)
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.orchestrator import SR2
from sr2.pipeline.resolvers import static_text as static_text_module
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent

_CLOCK_TEXT = "CLOCK {time}"


@pytest.fixture
def stepping_clock(monkeypatch):
    """Each clock render advances one minute — every firing is distinguishable."""
    state = {"n": 0}

    def _now(tz=None):
        state["n"] += 1
        minutes = 9 * 60 + state["n"] - 1
        return datetime.datetime(
            2026, 7, 22, minutes // 60, minutes % 60, tzinfo=ZoneInfo("America/Phoenix")
        )

    monkeypatch.setattr(static_text_module.date, "now", _now)


def _make_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                target="system",
                resolvers=[
                    ResolverConfig(
                        type="static", config={"text": "You are a placement test assistant."}
                    )
                ],
            ),
            LayerConfig(
                name="conversation",
                target="messages",
                resolvers=[
                    # Canonical FR4 placement: session, clock, input — in order.
                    ResolverConfig(type="session"),
                    ResolverConfig(
                        type="static_text",
                        name="clock",
                        # ``user_input`` fires in the second engine cycle: the
                        # session resolver first appends its history snapshot,
                        # then the clock fires (resolver order) and appends
                        # AFTER history, and ``input`` appends last.
                        config={"timezone": "America/Phoenix", "text": _CLOCK_TEXT},
                        subscriptions=[
                            EventSubscriptionConfig(
                                event="user_input", phase="completed"
                            )
                        ],
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


def _make_sr2(llm) -> SR2:
    return SR2(
        pipeline_config=_make_pipeline_config(),
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=stub_executor,
    )


def _iter_tool() -> list[StreamEvent]:
    return [tool_use_event(tool_use_id="call_001"), StreamEvent(type="end")]


def _iter_final(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


def _text_of(msg: Message) -> str:
    return "".join(b.text for b in msg.content if isinstance(b, TextBlock))


def _clock_indices(messages: list[Message]) -> list[int]:
    return [i for i, m in enumerate(messages) if _text_of(m).startswith("CLOCK ")]


async def _drive_turn(sr2: SR2, text: str) -> None:
    async for _ in sr2.turn(make_user_input(text)):
        pass


def _assert_clock_between_history_and_input(messages: list[Message]) -> None:
    """Exactly one clock block, after all history, directly before the input echo."""
    clocks = _clock_indices(messages)
    assert len(clocks) == 1
    clock_idx = clocks[0]
    assert clock_idx > 0, "history compiles before the clock"
    assert clock_idx == len(messages) - 2, "clock compiles before the new input"
    assert _text_of(messages[-1]) and not _text_of(messages[-1]).startswith("CLOCK ")
    # Consecutive user messages (clock block then input) are expected.
    assert messages[clock_idx].role == "user"
    assert messages[clock_idx + 1].role == "user"


class TestClockPlacementCompiledOncePerTurn:
    @pytest.mark.asyncio
    async def test_clock_between_history_and_input_turn_two(self, stepping_clock):
        """With prior history (turn 2+), the compiled request is
        [history…, clock, input echo] — the canonical FR4 order."""
        llm = SequentialMockLLM([_iter_final("Ack one."), _iter_final("Ack two.")])
        sr2 = _make_sr2(llm)

        await _drive_turn(sr2, "turn one")
        await _drive_turn(sr2, "turn two")

        messages = llm.stream_calls[1].messages
        assert [_text_of(m) for m in messages[:2]] == ["turn one", "Ack one."]
        _assert_clock_between_history_and_input(messages)
        assert _text_of(messages[-1]) == "turn two"

    @pytest.mark.asyncio
    async def test_clock_before_input_on_historyless_first_turn(self, stepping_clock):
        """Turn 1 has no history: the clock still compiles before the input."""
        llm = SequentialMockLLM([_iter_final("Hello back.")])
        sr2 = _make_sr2(llm)

        await _drive_turn(sr2, "first question")

        messages = llm.stream_calls[0].messages
        assert len(messages) == 2
        assert _text_of(messages[0]).startswith("CLOCK ")
        assert _text_of(messages[1]) == "first question"


class TestClockOnToolLoopIterations:
    @pytest.mark.asyncio
    async def test_clock_holds_on_tool_loop_iteration_with_history(
        self, stepping_clock
    ):
        """A tool-loop iteration's request = the turn-compiled request
        (clock between history and input) extended with the tool traffic —
        so the clock sits after history and before the new input on every
        iteration of the turn."""
        llm = SequentialMockLLM(
            [_iter_final("Ack one."), _iter_tool(), _iter_final("All done.")]
        )
        sr2 = _make_sr2(llm)

        await _drive_turn(sr2, "turn one")
        await _drive_turn(sr2, "run a tool")

        # Iteration 1 request: [hist…, clock, input echo].
        first = llm.stream_calls[1].messages
        assert [_text_of(m) for m in first[:2]] == ["turn one", "Ack one."]
        _assert_clock_between_history_and_input(first)

        # Tool-loop iteration request: the compiled prefix keeps its clock →
        # input order; the orchestrator appends the tool traffic after it.
        second = llm.stream_calls[2].messages
        assert second is not first
        assert _text_of(second[0]) == "turn one"
        assert _text_of(second[1]) == "Ack one."
        clocks = _clock_indices(second)
        assert len(clocks) == 1, (
            "exactly one clock block per turn: the tool-loop request extends "
            "the turn-compiled request and must not duplicate the clock"
        )
        clock_idx = clocks[0]
        assert clock_idx > 1, "clock sits after history on the tool-loop iteration"
        # The compiled prefix [hist, hist, clock] is preserved verbatim, and
        # the input echo keeps its place directly after the clock.
        assert _text_of(second[clock_idx + 1]) == "run a tool"
        tool_uses = [
            b for m in second for b in m.content if isinstance(b, ToolUseBlock)
        ]
        tool_results = [
            b for m in second for b in m.content if isinstance(b, ToolResultBlock)
        ]
        assert len(tool_uses) == 1
        assert len(tool_results) == 1

    @pytest.mark.asyncio
    async def test_single_clock_block_per_turn(self, stepping_clock):
        """Across a two-iteration tool turn the clock fired exactly once —
        no duplicate clock blocks accumulate in the messages."""
        llm = SequentialMockLLM([_iter_tool(), _iter_final("All done.")])
        sr2 = _make_sr2(llm)

        await _drive_turn(sr2, "tool please")

        assert len(llm.stream_calls) == 2
        first_clocks = _clock_indices(llm.stream_calls[0].messages)
        second_clocks = _clock_indices(llm.stream_calls[1].messages)
        assert len(first_clocks) == 1
        assert len(second_clocks) == 1
        # Same block — the request was extended, not recompiled.
        assert (
            llm.stream_calls[1].messages[first_clocks[0]]
            is llm.stream_calls[0].messages[first_clocks[0]]
        )


class TestClockAcrossTwoTurns:
    @pytest.mark.asyncio
    async def test_system_blocks_byte_identical_clock_values_differ(
        self, stepping_clock
    ):
        llm = SequentialMockLLM([_iter_final("Ack one."), _iter_final("Ack two.")])
        sr2 = _make_sr2(llm)

        await _drive_turn(sr2, "turn one")
        await _drive_turn(sr2, "turn two")

        turn1_first = llm.stream_calls[0]
        turn2_first = llm.stream_calls[1]

        # System blocks byte-identical across the two turns...
        assert turn1_first.system is not None and turn2_first.system is not None
        t1 = [b.model_dump_json() for b in turn1_first.system]
        t2 = [b.model_dump_json() for b in turn2_first.system]
        assert t1 == t2
        assert not any("CLOCK" in blob for blob in t1)

        # ...and the clock values in the messages really did differ.
        c1 = _clock_indices(turn1_first.messages)[-1]
        c2 = _clock_indices(turn2_first.messages)[-1]
        assert _text_of(turn1_first.messages[c1]) != _text_of(turn2_first.messages[c2])
        assert _text_of(turn1_first.messages[c1]).startswith("CLOCK 09:0")
        assert _text_of(turn2_first.messages[c2]).startswith("CLOCK 09:0")

        # FR4 holds on turn 2 as well: clock after history, before the input.
        _assert_clock_between_history_and_input(turn2_first.messages)
