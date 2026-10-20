"""Tests for sr2-122 — consecutive repeat guard in the tool loop.

Spec: role-run-stop-rules (spc-959), FR 7-18 and 22, AC 4-6.

With ``pipeline.repeat_guard.enabled``, every executed tool call still runs;
an identical call (same tool, same sorted-key JSON arguments as the previous
executed call) with the same result gets ``note`` instead of its real
result, ``warning`` is appended on repeat ``notes_before_stop``, and the
next same-result repeat ends the turn through the sr2-121 forced final
answer with reason ``repeat`` — regardless of ``final_answer.enabled``.
``max_identical_calls`` consecutive identical calls end the turn whatever
the results. With the block absent or disabled, results are never altered.

Case table rows: AC4-r1..r6, AC5-r1..r4, AC6-r1..r7, plus config defaults.
"""

from __future__ import annotations

from typing import Any

import pytest

from sr2.config.models import (
    DEFAULT_FINAL_ANSWER_INSTRUCTION,
    DEFAULT_FINAL_ANSWER_NOT_RUN,
    DEFAULT_REPEAT_NOTE,
    DEFAULT_REPEAT_WARNING,
    PipelineConfig,
    RepeatGuardConfig,
)
from sr2.models import TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    tool_use_event,
)

REASON_1_4 = "you repeated call 1 4 times"
REASON_1_10 = "you repeated call 1 10 times"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(**pipeline_kwargs: Any) -> PipelineConfig:
    """Minimal pipeline config. The case-table default supplies no
    ``final_answer`` block — the repeat stop must force the final answer
    anyway (FR 11)."""
    raw: dict[str, Any] = make_minimal_config().model_dump()
    raw.update(pipeline_kwargs)
    return PipelineConfig(**raw)


def _guard(**overrides: Any) -> dict[str, Any]:
    """repeat_guard block for the pipeline config dict (case-table default)."""
    return {"enabled": True, "notes_before_stop": 2, "max_identical_calls": 10, **overrides}


SHELL_INPUT = {"command": "pytest"}


def _x(call_id: str) -> StreamEvent:
    """Case-table X: shell {"command": "pytest"}."""
    return tool_use_event(tool_use_id=call_id, tool_name="shell", tool_input=dict(SHELL_INPUT))


def _iter_events(*calls: StreamEvent) -> list[StreamEvent]:
    return list(calls) + [StreamEvent(type="end")]


def _text_reply(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


class ScriptedExecutor:
    """Fake executor running one scripted result per call id.

    Each entry maps a tool_use id to either a result content string or a
    ``(content, is_error)`` tuple. Once the script is exhausted the last
    entry repeats (mirrors SequentialMockLLM's repeat-last behaviour).
    """

    def __init__(self, script: dict[str, Any]) -> None:
        self.script = script
        self.order: list[str] = []

    async def __call__(self, block: ToolUseBlock) -> ToolResultBlock:
        self.order.append(block.id)
        spec = self.script[block.id]
        content, is_error = spec if isinstance(spec, tuple) else (spec, False)
        return ToolResultBlock(tool_use_id=block.id, content=content, is_error=is_error)


def _build_sr2(llm, executor, config: PipelineConfig):
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=executor,
    )


async def _run_turn(
    config: PipelineConfig,
    sequences: list[list[StreamEvent]],
    script: dict[str, Any],
    *,
    final_text: str = "FINAL",
):
    """Drive a turn: scripted LLM sequences, then a text reply ending the
    turn (appended unless the caller's sequences already end in text).

    Returns (events, llm, executor).
    """
    seqs = list(sequences)
    if seqs[-1][0].type == "tool_use":
        # Two identical text replies: the guard-stop path consumes the reply
        # after its stop iteration as the forced final answer's reply; turns
        # that never stop simply never reach the second one.
        seqs = seqs + [_text_reply(final_text)] * 2
    executor = ScriptedExecutor(script)
    llm = SequentialMockLLM(seqs)
    sr2 = _build_sr2(llm, executor, config)
    events = [e async for e in sr2.turn(make_user_input())]
    return events, llm, executor


def _note(first_call: int, repeat: int, note: str = DEFAULT_REPEAT_NOTE) -> str:
    return note.format(first_call=first_call, repeat=repeat)


def _expected_result(call_id: str, script: dict[str, Any], notes: dict[str, str]) -> str:
    """What the model must see as call ``call_id``'s result: its note when
    the guard replaced it, else its scripted real content."""
    if call_id in notes:
        return notes[call_id]
    spec = script[call_id]
    return spec[0] if isinstance(spec, tuple) else spec


def _results_seen(llm: SequentialMockLLM) -> list[str]:
    """Every tool_result content the model was ever shown, in order.

    Requests are cumulative, so each request after the first adds exactly
    its own new results — except the guard-stop path's forced call, which
    re-sends the pre-forced request: skip it by message identity.
    """
    seen: list[str] = []
    prev_messages: list | None = None
    for request in llm.stream_calls[1:]:
        if request.messages is prev_messages or (
            prev_messages is not None and len(request.messages) <= len(prev_messages)
        ):
            # Re-sent (or shorter) history: the forced call. Contributes nothing.
            prev_messages = request.messages
            continue
        for message in request.messages[len(prev_messages) :] if prev_messages else request.messages:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    seen.append(block.content)
        prev_messages = request.messages
    return seen


def _assert_results(
    llm: SequentialMockLLM, script: dict[str, Any], notes: dict[str, str]
) -> None:
    expected = [_expected_result(cid, script, notes) for cid in script]
    assert _results_seen(llm) == expected


def _rg_events(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "repeat_guard"]


def _fa_events(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "final_answer"]


def _final_text(events: list[StreamEvent]) -> str:
    """The turn's final text: the text events after the last end event."""
    last_end = max(i for i, e in enumerate(events) if e.type == "end")
    return "".join(e.text for e in events[last_end - 1 : -1] if e.type == "text")


def _malformed_event(call_id: str) -> StreamEvent:
    """A tool_use event flagged invalid by the LLM integration (obsidian-1fmv)."""
    return StreamEvent(
        type="tool_use",
        tool_use_id=call_id,
        tool_name="shell",
        tool_input={},
        meta={"invalid_arguments": True, "error": "Expecting value: line 1 column 1"},
    )


# ---------------------------------------------------------------------------
# AC4 — note, warning, then stop through the forced final answer
# ---------------------------------------------------------------------------


class TestAC4NoteWarningStop:
    @pytest.mark.asyncio
    async def test_ac4_r1_second_call_gets_the_note_no_warning(self):
        """AC4-r1: X, X (same result). Executor runs twice; the model sees
        call 2 as the note naming call 1, with no warning appended."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(_x("c1")), _iter_events(_x("c2"))], script
        )

        assert executor.order == ["c1", "c2"], "every call still executes"
        _assert_results(llm, script, {"c2": _note(1, 1)})
        assert "Do not repeat it." in _results_seen(llm)[-1]
        assert "ends the run" not in _results_seen(llm)[-1], "no warning on repeat 1"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac4_r2_repeat_two_note_plus_warning(self):
        """AC4-r2: X ×3 (same). Call 3's note names call 1 and ends with the
        warning for repeat 2."""
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "1 failed"}
        expected = _note(1, 2) + DEFAULT_REPEAT_WARNING.format(repeat=2)
        assert expected.endswith(" This is repeat 2; the next identical call ends the run with an escalation.")
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x("c1")), _iter_events(_x("c2")), _iter_events(_x("c3"))],
            script,
        )

        assert executor.order == ["c1", "c2", "c3"]
        _assert_results(llm, script, {"c2": _note(1, 1), "c3": expected})

    @pytest.mark.asyncio
    async def test_ac4_r3_fourth_call_stops_via_forced_final_answer(self):
        """AC4-r3: X ×4 (same). Turn ends through the forced final answer:
        next request tools=None, reason filled, one final_answer event with
        reason "repeat" — with no final_answer config block at all."""
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "1 failed", "c4": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 5)],
            script,
        )

        assert executor.order == ["c1", "c2", "c3", "c4"]
        assert len(llm.stream_calls) == 5, "4 iterations plus exactly one forced call"
        # The forced call re-sends the pre-forced request: its history ends
        # with the stop note, and the forced pair never reaches the LLM.
        pre_forced = llm.stream_calls[3]
        stop_note = pre_forced.messages[-1].content[-1].content
        assert stop_note.startswith(_note(1, 3))
        assert stop_note.endswith(DEFAULT_REPEAT_WARNING.format(repeat=2))
        forced_request = llm.stream_calls[4]
        assert forced_request.tools is None
        instruction = "".join(
            b.text
            for b in forced_request.messages[-1].content
            if isinstance(b, TextBlock)
        )
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON_1_4)
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        assert _final_text(events) == "FINAL"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac4_r4_note_names_the_previous_run(self):
        """AC4-r4: A, X, X. The note on call 3 says 'identical to call 2'."""
        a = tool_use_event(tool_use_id="c1", tool_name="read", tool_input={"path": "a"})
        script = {"c1": "src", "c2": "1 failed", "c3": "1 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(a), _iter_events(_x("c2")), _iter_events(_x("c3"))],
            script,
        )

        _assert_results(llm, script, {"c3": _note(2, 1)})

    @pytest.mark.asyncio
    async def test_ac4_r5_argument_key_order_is_irrelevant(self):
        """AC4-r5: {"a":1,"b":2} then {"b":2,"a":1}, same tool and result:
        identical — the second call gets the note."""
        c1 = tool_use_event(tool_use_id="c1", tool_name="shell", tool_input={"a": 1, "b": 2})
        c2 = tool_use_event(tool_use_id="c2", tool_name="shell", tool_input={"b": 2, "a": 1})
        script = {"c1": "same", "c2": "same"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(c1), _iter_events(c2)], script
        )

        _assert_results(llm, script, {"c2": _note(1, 1)})

    @pytest.mark.asyncio
    async def test_ac4_r6_custom_note_placeholders(self):
        """AC4-r6: custom note 'DUP {first_call}/{repeat}' — call 2's result
        is exactly 'DUP 1/1'."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard(note="DUP {first_call}/{repeat}")),
            [_iter_events(_x("c1")), _iter_events(_x("c2"))],
            script,
        )

        _assert_results(llm, script, {"c2": "DUP 1/1"})


# ---------------------------------------------------------------------------
# AC5 — different results, and the identical-call cap
# ---------------------------------------------------------------------------


class TestAC5DifferentResultsAndCap:
    @pytest.mark.asyncio
    async def test_ac5_r1_different_result_passes_and_resets(self):
        """AC5-r1: X→waiting, X→done, X→done. Call 2 gets its real 'done';
        call 3 gets the note with repeat 1."""
        script = {"c1": "waiting", "c2": "done", "c3": "done"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 4)],
            script,
        )

        assert executor.order == ["c1", "c2", "c3"]
        _assert_results(llm, script, {"c3": _note(2, 1)})

    @pytest.mark.asyncio
    async def test_ac5_r2_max_identical_calls_stops_whatever_the_results(self):
        """AC5-r2: X ×10 with distinct results. All 10 execute with their
        real results; after call 10 the turn ends with reason
        'you repeated call 1 10 times'."""
        script = {f"c{i}": f"r{i}" for i in range(1, 11)}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 11)],
            script,
        )

        assert executor.order == [f"c{i}" for i in range(1, 11)]
        assert len(llm.stream_calls) == 11, "10 iterations plus exactly one forced call"
        forced_request = llm.stream_calls[10]
        assert forced_request.tools is None
        instruction = "".join(
            b.text
            for b in forced_request.messages[-1].content
            if isinstance(b, TextBlock)
        )
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON_1_10)
        fa = _fa_events(events)
        assert len(fa) == 1 and fa[0].meta == {"reason": "repeat", "fallback_used": False}
        rg = _rg_events(events)
        assert len(rg) == 1 and rg[0].meta["action"] == "stop"
        assert _final_text(events) == "FINAL"
        # The stop fired on call 10: its result is not shown to the model —
        # the forced request's history ends at r9, and the forced pair carries
        # the not_run result for c10.
        history_results = [
            block
            for message in forced_request.messages[:-2]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert [r.tool_use_id for r in history_results] == [f"c{i}" for i in range(1, 10)]
        assert history_results[-1].content == "r9"
        not_run = [
            block
            for message in forced_request.messages[-2:]
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert not_run[0].tool_use_id == "c10"
        assert not_run[0].content == DEFAULT_FINAL_ANSWER_NOT_RUN

    @pytest.mark.asyncio
    async def test_ac5_r3_comparison_uses_the_real_previous_result(self):
        """AC5-r3: X→x, X→x, X→x. Call 3 is compared against the real 'x',
        never the note: it gets note plus warning (repeat 2)."""
        script = {"c1": "x", "c2": "x", "c3": "x"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 4)],
            script,
        )

        _assert_results(
            llm,
            script,
            {"c2": _note(1, 1), "c3": _note(1, 2) + DEFAULT_REPEAT_WARNING.format(repeat=2)},
        )

    @pytest.mark.asyncio
    async def test_ac5_r4_is_error_difference_is_a_different_result(self):
        """AC5-r4: same content, is_error false then true: not the same
        result — call 2 gets its real result."""
        script = {"c1": "boom", "c2": ("boom", True)}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x("c1")), _iter_events(_x("c2"))],
            script,
        )

        _assert_results(llm, script, {})
        assert _rg_events(events) == []


# ---------------------------------------------------------------------------
# AC6 — resets, disabled behaviour, stream events
# ---------------------------------------------------------------------------


class TestAC6ResetsDisabledAndEvents:
    @pytest.mark.asyncio
    async def test_ac6_r1_different_tool_resets_both_counts(self):
        """AC6-r1: X, X, Y, Y. Call 3 (Y) is a real result; call 4's note
        says 'identical to call 3', repeat 1."""
        y1 = tool_use_event(tool_use_id="c3", tool_name="grep", tool_input={"pattern": "x"})
        y2 = tool_use_event(tool_use_id="c4", tool_name="grep", tool_input={"pattern": "x"})
        script = {"c1": "1 failed", "c2": "1 failed", "c3": "hit", "c4": "hit"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [
                _iter_events(_x("c1")),
                _iter_events(_x("c2")),
                _iter_events(y1),
                _iter_events(y2),
            ],
            script,
        )

        _assert_results(llm, script, {"c2": _note(1, 1), "c4": _note(3, 1)})
        assert [e.meta for e in _rg_events(events)] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
            {"call": 4, "first_call": 3, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
        ]

    @pytest.mark.asyncio
    async def test_ac6_r2_different_arguments_no_note(self):
        """AC6-r2: 'pytest' then 'pytest -x': both real results, no note."""
        c2 = tool_use_event(
            tool_use_id="c2", tool_name="shell", tool_input={"command": "pytest -x"}
        )
        script = {"c1": "1 failed", "c2": "2 failed"}
        events, llm, _ = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(_x("c1")), _iter_events(c2)], script
        )

        _assert_results(llm, script, {})
        assert _rg_events(events) == []

    @pytest.mark.asyncio
    async def test_ac6_r3_guard_absent_results_never_altered(self):
        """AC6-r3: no repeat_guard block; X ×5 same result. All 5 real, no
        repeat_guard events, the turn continues normally."""
        script = {f"c{i}": "1 failed" for i in range(1, 6)}
        events, llm, executor = await _run_turn(
            _config(), [_iter_events(_x(f"c{i}")) for i in range(1, 6)], script
        )

        assert executor.order == [f"c{i}" for i in range(1, 6)]
        _assert_results(llm, script, {})
        assert _rg_events(events) == []
        assert _fa_events(events) == []
        assert _final_text(events) == "FINAL"

    @pytest.mark.asyncio
    async def test_ac6_r4_repeat_guard_event_meta_sequence(self):
        """AC6-r4: the AC4-r3 sequence emits three repeat_guard events with
        the specified meta, the last one action 'stop'."""
        script = {f"c{i}": "1 failed" for i in range(1, 5)}
        events, _, _ = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x(f"c{i}")) for i in range(1, 5)],
            script,
        )

        rg = _rg_events(events)
        assert [e.meta for e in rg] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"},
            {"call": 3, "first_call": 1, "identical_calls": 3, "same_result_repeats": 2, "action": "note"},
            {"call": 4, "first_call": 1, "identical_calls": 4, "same_result_repeats": 3, "action": "stop"},
        ]

    @pytest.mark.asyncio
    async def test_ac6_r5_two_identical_calls_in_one_message(self):
        """AC6-r5: one assistant message with two X calls, same result: the
        second gets the note 'identical to call 1'."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()), [_iter_events(_x("c1"), _x("c2"))], script
        )

        assert executor.order == ["c1", "c2"]
        _assert_results(llm, script, {"c2": _note(1, 1)})
        assert [e.meta for e in _rg_events(events)] == [
            {"call": 2, "first_call": 1, "identical_calls": 2, "same_result_repeats": 1, "action": "note"}
        ]

    @pytest.mark.asyncio
    async def test_ac6_r6_malformed_call_not_numbered_no_reset(self):
        """AC6-r6: X, malformed call, X. The malformed call is not numbered
        and does not reset: the second X is call 2 and gets the note
        'identical to call 1'."""
        script = {"c1": "1 failed", "c3": "1 failed"}
        events, llm, executor = await _run_turn(
            _config(repeat_guard=_guard()),
            [_iter_events(_x("c1")), _iter_events(_malformed_event("c2")), _iter_events(_x("c3"))],
            script,
        )

        assert executor.order == ["c1", "c3"], "the malformed call never executes"
        # The malformed call's explanatory result is real and passes through;
        # the guard never numbers it (FR 7).
        malformed_note = (
            "Tool call 'shell' was not executed: its arguments are invalid JSON "
            "(parse error: Expecting value: line 1 column 1). "
            "Retry with valid JSON arguments."
        )
        notes = {"c2": malformed_note, "c3": _note(1, 1)}
        full_script = dict(script)
        full_script["c2"] = malformed_note
        # Results reach the model in message order: c1's real result, then
        # the malformed call's explanation, then c3's note.
        ordered_ids = ["c1", "c2", "c3"]
        expected = [_expected_result(cid, full_script, notes) for cid in ordered_ids]
        assert _results_seen(llm) == expected
        assert [e.meta["call"] for e in _rg_events(events)] == [2]

    @pytest.mark.asyncio
    async def test_ac6_r7_guard_state_is_per_turn(self):
        """AC6-r7: turn 1 ends after X; turn 2 starts with X, same result:
        turn 2's X gets its real result."""
        script = {"c1": "1 failed", "c2": "1 failed"}
        executor = ScriptedExecutor(script)
        llm = SequentialMockLLM(
            [_iter_events(_x("c1")), _text_reply("T1"), _iter_events(_x("c2")), _text_reply("T2")]
        )
        sr2 = _build_sr2(llm, executor, _config(repeat_guard=_guard()))

        events1 = [e async for e in sr2.turn(make_user_input())]
        events2 = [e async for e in sr2.turn(make_user_input())]

        assert _final_text(events1) == "T1" and _final_text(events2) == "T2"
        assert executor.order == ["c1", "c2"]
        _assert_results(llm, script, {})
        assert _rg_events(events1 + events2) == []
        await sr2.aclose()


# ---------------------------------------------------------------------------
# Config defaults (FR 22: every text is a YAML field with a default)
# ---------------------------------------------------------------------------


class TestRepeatGuardConfigDefaults:
    def test_defaults_off_with_spec_texts(self):
        g = RepeatGuardConfig()
        assert g.enabled is False
        assert g.notes_before_stop == 2
        assert g.max_identical_calls == 10
        assert g.note == (
            "Result not shown: this call is identical to call {first_call} and "
            "returned the same result. Do not repeat it. Change your approach, or if "
            "you cannot make progress, stop and escalate in your contract's format."
        )
        assert g.warning == (
            " This is repeat {repeat}; the next identical call ends the run with an escalation."
        )

    def test_pipeline_default_is_off(self):
        assert make_minimal_config().repeat_guard.enabled is False

    def test_reason_text_shape(self):
        from sr2.orchestrator import _repeat_reason_text

        assert _repeat_reason_text(1, 4) == REASON_1_4
        assert _repeat_reason_text(2, 10) == "you repeated call 2 10 times"
