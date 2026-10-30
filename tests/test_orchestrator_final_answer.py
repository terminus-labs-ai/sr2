"""Tests for sr2-121 — forced final answer at the tool-iteration limit.

Spec: role-run-stop-rules (spc-959), FR 1-6 and 22, AC 1-3.

With ``pipeline.final_answer.enabled``, an iteration that brings tool calls at
``max_tool_iterations`` executes none of them, gives each the ``not_run``
result, appends the ``instruction`` user message and makes exactly one LLM
call with ``tools=None``. The text reply (or the ``fallback`` text) becomes
the turn's final response through the normal final-response path. With the
block absent or disabled the limit raises ``ToolLoopLimitError`` exactly as
before.

Case table rows: AC1-r1, AC1-r2, AC2-r1..r4, AC3-r1..r3.
"""

from __future__ import annotations

from typing import Any

import pytest

from sr2.config.models import (
    DEFAULT_FINAL_ANSWER_FALLBACK,
    DEFAULT_FINAL_ANSWER_INSTRUCTION,
    DEFAULT_FINAL_ANSWER_NOT_RUN,
    FinalAnswerConfig,
    PipelineConfig,
    ToolLoopLimitError,
)
from sr2.models import TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, StreamEvent
from conftest import (
    SequentialMockLLM,
    make_minimal_config,
    make_user_input,
    stub_executor,
    tool_use_event,
)

REASON = "you reached the tool-call limit of 2 iterations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fa(**overrides: Any) -> dict[str, Any]:
    """final_answer block for the pipeline config dict."""
    return {"enabled": True, **overrides}


def _config(**pipeline_kwargs: Any) -> PipelineConfig:
    """Minimal pipeline config with max_tool_iterations=2 (case-table default).

    A dict goes through Pydantic's validation of the nested FinalAnswerConfig,
    the same path a YAML ``pipeline:`` section takes.
    """
    raw: dict[str, Any] = make_minimal_config().model_dump()
    raw["max_tool_iterations"] = 2
    raw.update(pipeline_kwargs)
    return PipelineConfig(**raw)


class RecordingExecutor:
    """Fake tool executor recording every block it is asked to run."""

    def __init__(self) -> None:
        self.calls: list[ToolUseBlock] = []

    async def __call__(self, block: ToolUseBlock) -> ToolResultBlock:
        self.calls.append(block)
        return ToolResultBlock(tool_use_id=block.id, content=f"result_for_{block.name}")


def _tool_iteration(call_id: str) -> list[StreamEvent]:
    return [tool_use_event(tool_use_id=call_id), StreamEvent(type="end")]


def _tool_iteration_at_limit(call_id: str = "tc_2") -> list[StreamEvent]:
    """Case-table default: a tool call on EVERY iteration until the forced call.

    SequentialMockLLM repeats its last sequence once exhausted, so the list
    ends with a tool iteration: the limit iteration (seq == 2) brings a tool
    call, which is what triggers the forced final answer.
    """
    return [
        _tool_iteration("tc_0"),
        _tool_iteration("tc_1"),
        _tool_iteration(call_id),
    ]


def _text_reply(text: str) -> list[StreamEvent]:
    return [StreamEvent(type="text", text=text), StreamEvent(type="end")]


def _build_sr2(llm, executor, config: PipelineConfig):
    from sr2.orchestrator import SR2

    return SR2(
        pipeline_config=config,
        llm={"default": llm},
        token_counter=CharacterTokenCounter(),
        tool_executor=executor,
    )


async def _run_turn(config: PipelineConfig, forced_reply: list[StreamEvent]):
    """Drive the case-table default: a tool call every iteration until the
    forced call, which answers with ``forced_reply``.

    Returns (events, llm, executor).
    """
    executor = RecordingExecutor()
    sequences = _tool_iteration_at_limit() + [forced_reply]
    if forced_reply[0].type == "tool_use":
        # The forced reply itself brings a tool call: append a final text so
        # SequentialMockLLM stays defined past the forced call. The forced
        # path must end the turn without ever reaching that sequence.
        sequences = sequences + [_text_reply("MUST NOT APPEAR")]
    llm = SequentialMockLLM(sequences)
    sr2 = _build_sr2(llm, executor, config)
    events = [e async for e in sr2.turn(make_user_input())]
    return events, llm, executor


def _final_text(events: list[StreamEvent]) -> str:
    """The turn's final text: the text events after the final_answer event."""
    last_fa = max(
        (i for i, e in enumerate(events) if e.type == "final_answer"),
        default=len(events) - 1,
    )
    return "".join(e.text for e in events[last_fa + 1 :] if e.type == "text")


def _fa_events(events: list[StreamEvent]) -> list[StreamEvent]:
    return [e for e in events if e.type == "final_answer"]


def _last_message_text(request: CompletionRequest) -> str:
    message = request.messages[-1]
    assert message.role == "user", f"expected a user instruction message, got {message.role!r}"
    return "".join(b.text for b in message.content if isinstance(b, TextBlock))


# ---------------------------------------------------------------------------
# AC1 — nothing runs at the limit; the forced request; the final text
# ---------------------------------------------------------------------------


class TestAC1ForcedPath:
    @pytest.mark.asyncio
    async def test_ac1_multi_call_limit_pairs_every_unexecuted_call(self):
        """AC1: every call in the limit iteration retains its not_run pairing."""
        executor = RecordingExecutor()
        llm = SequentialMockLLM([
            _tool_iteration("tc_0"),
            _tool_iteration("tc_1"),
            [
                tool_use_event(tool_use_id="tc_2a", tool_input={"location": "Oslo"}),
                tool_use_event(tool_use_id="tc_2b", tool_input={"location": "Bergen"}),
                StreamEvent(type="end"),
            ],
            _text_reply("VERDICT: done"),
        ])
        sr2 = _build_sr2(llm, executor, _config(final_answer=_fa()))
        events = [e async for e in sr2.turn(make_user_input())]

        assert [b.id for b in executor.calls] == ["tc_0", "tc_1"]
        assert len(llm.stream_calls) == 4
        forced_request = llm.stream_calls[-1]
        assistant, results, instruction = forced_request.messages[-3:]
        assert [assistant.role, results.role, instruction.role] == [
            "assistant", "user", "user"
        ]
        assert all(isinstance(b, ToolUseBlock) for b in assistant.content)
        assert [b.id for b in assistant.content] == ["tc_2a", "tc_2b"]
        assert [(b.name, b.input) for b in assistant.content] == [
            ("get_weather", {"location": "Oslo"}),
            ("get_weather", {"location": "Bergen"}),
        ]
        assert all(isinstance(b, ToolResultBlock) for b in results.content)
        assert [(b.tool_use_id, b.content, b.is_error) for b in results.content] == [
            ("tc_2a", DEFAULT_FINAL_ANSWER_NOT_RUN, False),
            ("tc_2b", DEFAULT_FINAL_ANSWER_NOT_RUN, False),
        ]
        assert forced_request.tools is None
        assert _last_message_text(forced_request) == (
            DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON)
        )
        assert _final_text(events) == "VERDICT: done"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac1_r1_executor_runs_only_before_limit(self):
        """AC1-r1: the executor runs for iterations 0 and 1 only."""
        events, llm, executor = await _run_turn(
            _config(final_answer=_fa()), _text_reply("VERDICT: done")
        )

        assert [b.name for b in executor.calls] == ["get_weather", "get_weather"], (
            "the tool call at the limit must not be executed"
        )
        assert [b.id for b in executor.calls] == ["tc_0", "tc_1"]
        assert len(llm.stream_calls) == 4, (
            "iterations 0 and 1, the limit call, and exactly one forced call"
        )

    @pytest.mark.asyncio
    async def test_ac1_r1_unexecuted_call_gets_not_run_result(self):
        """AC1-r1: the iteration-2 call gets result 'Not run: the run is ending.'"""
        events, llm, _ = await _run_turn(
            _config(final_answer=_fa()), _text_reply("VERDICT: done")
        )

        forced_request = llm.stream_calls[-1]
        tool_results = [
            block
            for message in forced_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert sum(r.content == DEFAULT_FINAL_ANSWER_NOT_RUN for r in tool_results) == 1, (
            f"exactly one not_run result expected; got {[r.content for r in tool_results]!r}"
        )
        assert tool_results[-1].tool_use_id == "tc_2", (
            "the not_run result must pair with the unexecuted call"
        )
        assert tool_results[-1].is_error is False

    @pytest.mark.asyncio
    async def test_ac1_r1_forced_request_has_no_tools_and_ends_with_instruction(self):
        """AC1-r1: forced request has tools=None and the default instruction
        with the limit reason filled."""
        events, llm, _ = await _run_turn(
            _config(final_answer=_fa()), _text_reply("VERDICT: done")
        )

        forced_request = llm.stream_calls[-1]
        assert forced_request.tools is None
        instruction = _last_message_text(forced_request)
        assert instruction == DEFAULT_FINAL_ANSWER_INSTRUCTION.format(reason=REASON)

    @pytest.mark.asyncio
    async def test_ac1_r1_text_reply_is_final_text(self):
        """AC1-r1: the forced text reply is the turn's final text."""
        events, _, _ = await _run_turn(
            _config(final_answer=_fa()), _text_reply("VERDICT: done")
        )

        assert _final_text(events) == "VERDICT: done"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac1_r2_custom_instruction_and_not_run(self):
        """AC1-r2: custom instruction and not_run texts are used verbatim."""
        events, llm, executor = await _run_turn(
            _config(final_answer=_fa(instruction="STOP {reason}", not_run="SKIPPED")),
            _text_reply("VERDICT: done"),
        )

        assert _last_message_text(llm.stream_calls[-1]) == f"STOP {REASON}"
        forced_request = llm.stream_calls[-1]
        results = [
            block.content
            for message in forced_request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert results[-1] == "SKIPPED"
        assert len(executor.calls) == 2

    @pytest.mark.asyncio
    async def test_final_answer_goes_through_finalize_turn(self):
        """AC1 (implementation requirement): the forced response reaches
        session history via _finalize_turn like the normal final iteration."""
        executor = RecordingExecutor()
        llm = SequentialMockLLM(
            _tool_iteration_at_limit() + [_text_reply("VERDICT: done")]
        )
        config = _config(final_answer=_fa())
        sr2 = _build_sr2(llm, executor, config)

        captured: list = []
        original = sr2._finalize_turn

        async def spy(final_response):
            captured.append(final_response)
            return await original(final_response)

        sr2._finalize_turn = spy  # type: ignore[method-assign]
        events = [e async for e in sr2.turn(make_user_input())]

        assert len(captured) == 1
        content = captured[0].content
        assert len(content) == 1 and isinstance(content[0], TextBlock)
        assert content[0].text == "VERDICT: done"
        assert captured[0].stop_reason == "end_turn"
        assert _final_text(events) == "VERDICT: done"


# ---------------------------------------------------------------------------
# AC2 — fallback text when the forced reply is unusable
# ---------------------------------------------------------------------------


class TestAC2Fallback:
    @pytest.mark.asyncio
    async def test_ac2_r1_empty_reply_uses_fallback(self):
        """AC2-r1: an empty reply yields the default fallback with the reason."""
        events, llm, _ = await _run_turn(_config(final_answer=_fa()), _text_reply(""))

        assert _final_text(events) == DEFAULT_FINAL_ANSWER_FALLBACK.format(reason=REASON)
        assert len(llm.stream_calls) == 4, "no retry after an unusable forced reply"

    @pytest.mark.asyncio
    async def test_ac2_r2_whitespace_reply_uses_fallback(self):
        """AC2-r2: a whitespace-only reply yields the same fallback."""
        events, _, _ = await _run_turn(_config(final_answer=_fa()), _text_reply("  \n"))

        assert _final_text(events) == DEFAULT_FINAL_ANSWER_FALLBACK.format(reason=REASON)

    @pytest.mark.asyncio
    async def test_ac2_r3_tool_call_in_forced_reply_uses_fallback_and_never_runs(self):
        """AC2-r3: a tool call in the forced reply yields the fallback and is
        not executed."""
        events, llm, executor = await _run_turn(
            _config(final_answer=_fa()), _tool_iteration("tc_forced")
        )

        assert _final_text(events) == DEFAULT_FINAL_ANSWER_FALLBACK.format(reason=REASON)
        assert [b.id for b in executor.calls] == ["tc_0", "tc_1"], (
            "the tool call in the forced reply must not execute"
        )
        assert len(llm.stream_calls) == 4, "no retry after a tool-calling forced reply"
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_ac2_r4_custom_fallback_with_reason(self):
        """AC2-r4: custom fallback text with {reason} filled."""
        events, _, _ = await _run_turn(
            _config(final_answer=_fa(fallback="FAILED: {reason}")), _text_reply("")
        )

        assert _final_text(events) == f"FAILED: {REASON}"


# ---------------------------------------------------------------------------
# AC3 — disabled keeps the old error; exactly one final_answer event
# ---------------------------------------------------------------------------


class TestAC3DisabledAndEvents:
    @pytest.mark.asyncio
    async def test_ac3_r1_no_final_answer_block_raises_as_today(self):
        """AC3-r1: with no final_answer block the limit raises ToolLoopLimitError."""
        executor = RecordingExecutor()
        llm = SequentialMockLLM([_tool_iteration("tc_0"), _tool_iteration("tc_1"), _tool_iteration("tc_2")])
        sr2 = _build_sr2(llm, executor, _config())

        with pytest.raises(ToolLoopLimitError, match="Tool loop limit exceeded"):
            async for _ in sr2.turn(make_user_input()):
                pass

    @pytest.mark.asyncio
    async def test_ac3_r1_explicitly_disabled_also_raises(self):
        """AC3-r1: final_answer.enabled false keeps the old error path."""
        executor = RecordingExecutor()
        llm = SequentialMockLLM([_tool_iteration("tc_0"), _tool_iteration("tc_1"), _tool_iteration("tc_2")])
        sr2 = _build_sr2(llm, executor, _config(final_answer={"enabled": False}))

        with pytest.raises(ToolLoopLimitError, match="Tool loop limit exceeded"):
            async for _ in sr2.turn(make_user_input()):
                pass

    @pytest.mark.asyncio
    async def test_disabled_path_executes_before_the_error(self):
        """Regression: the disabled path still executes iterations 0 and 1 and
        raises only at the limit, as before this change."""
        executor = RecordingExecutor()
        llm = SequentialMockLLM([_tool_iteration("tc_0"), _tool_iteration("tc_1"), _tool_iteration("tc_2")])
        sr2 = _build_sr2(llm, executor, _config())

        with pytest.raises(ToolLoopLimitError):
            async for _ in sr2.turn(make_user_input()):
                pass

        assert len(executor.calls) == 2
        assert len(llm.stream_calls) == 3

    @pytest.mark.asyncio
    async def test_ac3_r2_exactly_one_final_answer_event_before_final_text(self):
        """AC3-r2: exactly one final_answer event with the specified meta,
        positioned before the final text event."""
        events, _, _ = await _run_turn(
            _config(final_answer=_fa()), _text_reply("VERDICT: done")
        )

        fa = _fa_events(events)
        assert len(fa) == 1
        assert fa[0].meta == {"reason": "limit", "fallback_used": False}
        first_text_after = next(
            i for i, e in enumerate(events) if i > events.index(fa[0]) and e.type == "text"
        )
        assert events[first_text_after].text == "VERDICT: done"

    @pytest.mark.asyncio
    async def test_ac3_r3_fallback_reply_marks_fallback_used(self):
        """AC3-r3: an empty forced reply yields one final_answer event with
        fallback_used true."""
        events, _, _ = await _run_turn(_config(final_answer=_fa()), _text_reply(""))

        fa = _fa_events(events)
        assert len(fa) == 1
        assert fa[0].meta == {"reason": "limit", "fallback_used": True}
        assert _final_text(events) == DEFAULT_FINAL_ANSWER_FALLBACK.format(reason=REASON)


# ---------------------------------------------------------------------------
# Config defaults (FR 22: every text is a YAML field with a default)
# ---------------------------------------------------------------------------


class TestFinalAnswerConfigDefaults:
    def test_defaults_off_with_spec_texts(self):
        fa = FinalAnswerConfig()
        assert fa.enabled is False
        assert fa.instruction == (
            "The harness is ending this run: {reason}. Tools are no longer available. "
            "Write your final answer now in your contract's format: your result if you "
            "have one, otherwise an escalation that says what blocked you and what you tried."
        )
        assert fa.fallback == "Run stopped by the harness: {reason}. The model produced no final answer."
        assert fa.not_run == "Not run: the run is ending."

    def test_pipeline_default_is_off(self):
        assert make_minimal_config().final_answer.enabled is False

    def test_reason_text_shape(self):
        from sr2.orchestrator import _limit_reason_text

        assert _limit_reason_text(2) == REASON
        assert _limit_reason_text(200) == "you reached the tool-call limit of 200 iterations"
