"""obsidian-1fmv — survive malformed tool-call arguments.

A model occasionally emits tool-call arguments that are not valid JSON (for
example a string value that is never closed, or a call cut off by the token
limit). Required behavior:

  1. Invalid tool-call arguments do not raise out of ``LiteLLMCallable.stream()``
     or ``LiteLLMCallable.complete()``. In a turn, the loop continues and the
     model receives a tool result stating the arguments were invalid JSON, so it
     can retry.
  2. On failure, the raw argument text (truncated to a bounded length), the tool
     name and the ``finish_reason`` are logged at WARNING.
  3. ``finish_reason == "length"`` with a pending tool call is reported as
     truncation.
  4. Valid tool calls behave exactly as before.

litellm.acompletion is the system boundary and is the only thing mocked. The
turn-level tests drive a real ``SR2`` orchestrator over a real
``LiteLLMCallable`` and inspect what the *next* LLM call receives — that is
what "the model receives a tool result" means observably.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from conftest import make_minimal_config, make_user_input
from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest


# ---------------------------------------------------------------------------
# litellm boundary fakes
# ---------------------------------------------------------------------------


def _delta(*, content=None, tool_calls=None) -> MagicMock:
  delta = MagicMock()
  delta.content = content
  delta.tool_calls = tool_calls
  delta.reasoning_content = None
  delta.thinking_blocks = None
  return delta


def _chunk(delta: MagicMock, finish_reason: str | None = None) -> MagicMock:
  choice = MagicMock()
  choice.delta = delta
  choice.finish_reason = finish_reason
  chunk = MagicMock()
  chunk.choices = [choice]
  chunk.usage = None
  return chunk


def _tool_delta(index: int, id: str | None, name: str | None, arguments: str) -> MagicMock:
  tc = MagicMock()
  tc.index = index
  tc.id = id
  tc.function = MagicMock()
  tc.function.name = name
  tc.function.arguments = arguments
  return tc


def _tool_call_stream(
  calls: list[tuple[str, str, str]],
  finish_reason: str = "tool_calls",
) -> list[MagicMock]:
  """Stream chunks for the given (id, name, arguments) tool calls.

  Each call's arguments are split across two delta chunks, as real backends
  do. The final chunk carries ``finish_reason``.
  """
  chunks: list[MagicMock] = []
  for index, (call_id, name, arguments) in enumerate(calls):
    half = len(arguments) // 2
    chunks.append(_chunk(_delta(tool_calls=[_tool_delta(index, call_id, name, arguments[:half])])))
    chunks.append(_chunk(_delta(tool_calls=[_tool_delta(index, None, None, arguments[half:])])))
  chunks.append(_chunk(_delta(), finish_reason=finish_reason))
  return chunks


def _text_stream(text: str) -> list[MagicMock]:
  return [_chunk(_delta(content=text)), _chunk(_delta(), finish_reason="stop")]


async def _agen(items):
  for item in items:
    yield item


class _FakeLiteLLM:
  """Replaces litellm.acompletion; returns one scripted stream per call."""

  def __init__(self, streams: list[list[MagicMock]]) -> None:
    self._streams = streams
    self.calls: list[dict] = []

  async def __call__(self, *args, **kwargs):
    self.calls.append(kwargs)
    idx = min(len(self.calls) - 1, len(self._streams) - 1)
    return _agen(self._streams[idx])


def _completion_response(
  tool_calls: list[tuple[str, str, str]],
  finish_reason: str = "tool_calls",
) -> MagicMock:
  tcs = []
  for call_id, name, arguments in tool_calls:
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    tcs.append(tc)
  choice = MagicMock()
  choice.message.content = None
  choice.message.tool_calls = tcs
  choice.finish_reason = finish_reason
  usage = MagicMock()
  usage.prompt_tokens = 10
  usage.completion_tokens = 5
  resp = MagicMock()
  resp.id = "resp-1"
  resp.choices = [choice]
  resp.usage = usage
  return resp


# ---------------------------------------------------------------------------
# Turn helpers
# ---------------------------------------------------------------------------


class _RecordingExecutor:
  def __init__(self) -> None:
    self.blocks: list[ToolUseBlock] = []

  async def __call__(self, block: ToolUseBlock) -> ToolResultBlock:
    self.blocks.append(block)
    return ToolResultBlock(tool_use_id=block.id, content=f"EXECUTED {block.name}")


async def _run_turn(fake: _FakeLiteLLM, executor: _RecordingExecutor):
  from sr2.orchestrator import SR2

  sr2 = SR2(
    pipeline_config=make_minimal_config(),
    llm=LiteLLMCallable("test-model"),
    token_counter=CharacterTokenCounter(),
    tool_executor=executor,
  )
  with patch("litellm.acompletion", new=fake):
    return [e async for e in sr2.turn(make_user_input("scan the vault"))]


def _tool_messages(call_kwargs: dict) -> dict[str, str]:
  """tool_call_id → content for every tool-role message sent to the LLM."""
  return {
    m["tool_call_id"]: m["content"]
    for m in call_kwargs["messages"]
    if m.get("role") == "tool"
  }


def _assistant_tool_call_ids(call_kwargs: dict) -> set[str]:
  ids: set[str] = set()
  for m in call_kwargs["messages"]:
    if m.get("role") == "assistant":
      for tc in m.get("tool_calls") or []:
        ids.add(tc["id"])
  return ids


def _warning_text(caplog) -> str:
  return "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


def _request() -> CompletionRequest:
  return CompletionRequest(messages=[Message(role="user", content=[TextBlock(text="hi")])])


UNTERMINATED = '{"path": "/vault/notes/unterminated'


# ---------------------------------------------------------------------------
# AC 1 — stream(): invalid arguments do not raise
# ---------------------------------------------------------------------------


class TestStreamDoesNotRaise:
  @pytest.mark.asyncio
  async def test_truncated_streamed_arguments_do_not_raise_and_stream_ends(self):
    fake = _FakeLiteLLM([_tool_call_stream([("tc_bad", "read_file", UNTERMINATED)])])

    with patch("litellm.acompletion", new=fake):
      events = [e async for e in LiteLLMCallable("m").stream(_request())]

    assert events, "stream() yielded nothing"
    assert events[-1].type == "end"


# ---------------------------------------------------------------------------
# AC 1 — turn continues and the model is told the arguments were invalid JSON
# ---------------------------------------------------------------------------


class TestTurnSurvivesTruncatedStreamedArguments:
  @pytest.mark.asyncio
  async def test_turn_completes_and_model_gets_invalid_json_tool_result(self):
    fake = _FakeLiteLLM([
      _tool_call_stream([("tc_bad", "read_file", UNTERMINATED)]),
      _text_stream("Retrying with valid arguments."),
    ])
    executor = _RecordingExecutor()

    events = await _run_turn(fake, executor)

    assert events[-1].type == "end"
    assert "Retrying with valid arguments." in "".join(e.text for e in events if e.type == "text")
    assert len(fake.calls) == 2, "the model must get a follow-up call after the malformed tool call"

    followup = fake.calls[1]
    assert "tc_bad" in _assistant_tool_call_ids(followup), (
      "the malformed call must stay in the assistant message so its tool result is paired"
    )
    tool_results = _tool_messages(followup)
    assert "tc_bad" in tool_results, f"no tool result for the malformed call: {tool_results!r}"
    result_text = tool_results["tc_bad"].lower()
    assert "invalid" in result_text and "json" in result_text, (
      f"tool result must state the arguments were invalid JSON, got {tool_results['tc_bad']!r}"
    )
    assert "truncat" not in result_text, (
      "a tool_calls finish is not a truncation and must not be reported to the model as one"
    )

  @pytest.mark.asyncio
  async def test_valid_call_alongside_malformed_call_still_executes_normally(self):
    fake = _FakeLiteLLM([
      _tool_call_stream([
        ("tc_good", "list_dir", '{"path": "/vault"}'),
        ("tc_bad", "read_file", UNTERMINATED),
      ]),
      _text_stream("done"),
    ])
    executor = _RecordingExecutor()

    await _run_turn(fake, executor)

    good = [b for b in executor.blocks if b.id == "tc_good"]
    assert len(good) == 1
    assert good[0].name == "list_dir"
    assert good[0].input == {"path": "/vault"}

    tool_results = _tool_messages(fake.calls[1])
    assert tool_results["tc_good"] == "EXECUTED list_dir"
    bad_text = tool_results["tc_bad"].lower()
    assert "invalid" in bad_text and "json" in bad_text


# ---------------------------------------------------------------------------
# AC 2 — WARNING log with raw arguments (bounded), tool name, finish_reason
# ---------------------------------------------------------------------------


class TestStreamWarningLog:
  @pytest.mark.asyncio
  async def test_warning_contains_raw_arguments_tool_name_and_finish_reason(self, caplog):
    fake = _FakeLiteLLM([_tool_call_stream([("tc_bad", "read_file", UNTERMINATED)], finish_reason="tool_calls")])

    with caplog.at_level(logging.WARNING):
      with patch("litellm.acompletion", new=fake):
        [e async for e in LiteLLMCallable("m").stream(_request())]

    logged = _warning_text(caplog)
    assert UNTERMINATED in logged, f"raw arguments missing from WARNING log: {logged!r}"
    assert "read_file" in logged
    assert "tool_calls" in logged, "finish_reason missing from WARNING log"

  @pytest.mark.asyncio
  async def test_logged_raw_arguments_are_truncated_to_a_bounded_length(self, caplog):
    huge = '{"content": "' + "x" * 100_000
    fake = _FakeLiteLLM([_tool_call_stream([("tc_bad", "write_file", huge)])])

    with caplog.at_level(logging.WARNING):
      with patch("litellm.acompletion", new=fake):
        [e async for e in LiteLLMCallable("m").stream(_request())]

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "no WARNING logged for malformed arguments"
    joined = "\n".join(warnings)
    assert '{"content": "xxxx' in joined, "the start of the raw arguments must be logged"
    assert len(joined) < 10_000, f"raw arguments not truncated: {len(joined)} chars logged"


# ---------------------------------------------------------------------------
# AC 3 — finish_reason "length" with a partial tool call is reported as truncation
# ---------------------------------------------------------------------------


class TestLengthFinishWithPartialToolCall:
  @pytest.mark.asyncio
  async def test_length_finish_turn_continues_and_truncation_is_reported(self, caplog):
    fake = _FakeLiteLLM([
      _tool_call_stream([("tc_cut", "write_file", '{"path": "a.md", "content": "partial')], finish_reason="length"),
      _text_stream("I will write a shorter file."),
    ])
    executor = _RecordingExecutor()

    with caplog.at_level(logging.WARNING):
      events = await _run_turn(fake, executor)

    assert events[-1].type == "end"
    assert len(fake.calls) == 2
    tool_results = _tool_messages(fake.calls[1])
    assert "tc_cut" in tool_results, "the cut-off call must receive a tool result"

    logged = _warning_text(caplog)
    assert "length" in logged, "finish_reason missing from WARNING log"
    assert "truncat" in tool_results["tc_cut"].lower(), (
      "the model must be told its tool call was truncated, "
      f"got tool result {tool_results['tc_cut']!r}"
    )

  @pytest.mark.asyncio
  async def test_length_finish_with_parseable_arguments_is_reported_as_truncation(self, caplog):
    fake = _FakeLiteLLM([_tool_call_stream([("tc_len", "write_file", '{"path": "a.md"}')], finish_reason="length")])

    with caplog.at_level(logging.WARNING):
      with patch("litellm.acompletion", new=fake):
        events = [e async for e in LiteLLMCallable("m").stream(_request())]

    assert events[-1].type == "end"
    logged = _warning_text(caplog)
    assert "length" in logged, f"finish_reason missing from WARNING log: {logged!r}"
    assert "truncat" in logged.lower(), f"truncation not reported in WARNING log: {logged!r}"


# ---------------------------------------------------------------------------
# AC 1 + AC 2 — complete(): invalid non-streamed arguments
# ---------------------------------------------------------------------------


class TestCompleteInvalidArguments:
  @pytest.mark.asyncio
  async def test_invalid_arguments_do_not_raise_and_call_is_returned(self):
    resp = _completion_response([("tc_bad", "read_file", UNTERMINATED)])

    with patch("litellm.acompletion", return_value=resp):
      result = await LiteLLMCallable("m").complete(_request())

    tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
    assert [(b.id, b.name) for b in tool_blocks] == [("tc_bad", "read_file")], (
      "the malformed call must still be returned so the caller can answer it"
    )
    assert result.stop_reason == "tool_use"

  @pytest.mark.asyncio
  async def test_valid_call_alongside_invalid_call_is_parsed_normally(self):
    resp = _completion_response([
      ("tc_good", "list_dir", '{"path": "/vault"}'),
      ("tc_bad", "read_file", "{not json"),
    ])

    with patch("litellm.acompletion", return_value=resp):
      result = await LiteLLMCallable("m").complete(_request())

    by_id = {b.id: b for b in result.content if isinstance(b, ToolUseBlock)}
    assert by_id["tc_good"].input == {"path": "/vault"}
    assert "tc_bad" in by_id

  @pytest.mark.asyncio
  async def test_warning_contains_raw_arguments_tool_name_and_finish_reason(self, caplog):
    resp = _completion_response([("tc_bad", "read_file", UNTERMINATED)], finish_reason="content_filter")

    with caplog.at_level(logging.WARNING):
      with patch("litellm.acompletion", return_value=resp):
        await LiteLLMCallable("m").complete(_request())

    logged = _warning_text(caplog)
    assert UNTERMINATED in logged, f"raw arguments missing from WARNING log: {logged!r}"
    assert "read_file" in logged
    assert "content_filter" in logged, "finish_reason missing from WARNING log"


# ---------------------------------------------------------------------------
# AC 5 — valid tool calls behave exactly as before, end to end
# ---------------------------------------------------------------------------


class TestValidToolCallUnchanged:
  @pytest.mark.asyncio
  async def test_valid_streamed_call_executes_with_parsed_input_and_no_warning(self, caplog):
    args = {"path": "/vault", "depth": 2}
    fake = _FakeLiteLLM([
      _tool_call_stream([("tc_ok", "list_dir", json.dumps(args))]),
      _text_stream("Here is the listing."),
    ])
    executor = _RecordingExecutor()

    with caplog.at_level(logging.WARNING):
      await _run_turn(fake, executor)

    assert [(b.id, b.name, b.input) for b in executor.blocks] == [("tc_ok", "list_dir", args)]
    assert _tool_messages(fake.calls[1]) == {"tc_ok": "EXECUTED list_dir"}
    assert "list_dir" not in _warning_text(caplog)


# ---------------------------------------------------------------------------
# AC 1 + AC 5 — the conversation stays protocol-valid across turns
# ---------------------------------------------------------------------------


def _unpaired_tool_results(call_kwargs: dict) -> list[str]:
  """tool_call_ids of tool messages with no matching id in a PRECEDING assistant tool_calls."""
  seen: set[str] = set()
  orphans: list[str] = []
  for m in call_kwargs["messages"]:
    if m.get("role") == "assistant":
      for tc in m.get("tool_calls") or []:
        seen.add(tc["id"])
    elif m.get("role") == "tool" and m["tool_call_id"] not in seen:
      orphans.append(m["tool_call_id"])
  return orphans


def _unanswered_tool_calls(call_kwargs: dict) -> list[str]:
  """Assistant tool_calls ids with no LATER tool message carrying that tool_call_id."""
  pending: list[str] = []
  for m in call_kwargs["messages"]:
    if m.get("role") == "assistant":
      pending.extend(tc["id"] for tc in m.get("tool_calls") or [])
    elif m.get("role") == "tool" and m["tool_call_id"] in pending:
      pending.remove(m["tool_call_id"])
  return pending


async def _run_two_turns(fake: _FakeLiteLLM, executor: _RecordingExecutor) -> dict:
  """Run two user turns on one SR2 session; return the kwargs of the final LLM call."""
  from sr2.orchestrator import SR2

  sr2 = SR2(
    pipeline_config=make_minimal_config(),
    llm=LiteLLMCallable("test-model"),
    token_counter=CharacterTokenCounter(),
    tool_executor=executor,
  )
  with patch("litellm.acompletion", new=fake):
    turn1 = [e async for e in sr2.turn(make_user_input("scan the vault"))]
    assert turn1[-1].type == "end"
    turn2 = [e async for e in sr2.turn(make_user_input("now summarize"))]
    assert turn2[-1].type == "end"
  return fake.calls[-1]


class TestSessionHistoryStaysPairedAcrossTurns:
  @pytest.mark.asyncio
  async def test_mixed_batch_every_tool_result_in_next_turn_is_paired(self):
    fake = _FakeLiteLLM([
      _tool_call_stream([
        ("tc_good", "list_dir", '{"path": "/vault"}'),
        ("tc_bad", "read_file", UNTERMINATED),
      ]),
      _text_stream("Scanned."),
      _text_stream("Summary."),
    ])

    turn2_request = await _run_two_turns(fake, _RecordingExecutor())

    assert len(fake.calls) == 3
    assert _unpaired_tool_results(turn2_request) == [], (
      "every tool result in the next turn's history must follow an assistant tool_calls entry with its id"
    )
    assert _unanswered_tool_calls(turn2_request) == [], (
      "every assistant tool_calls entry in the next turn's history must have a later tool result"
    )
    assert {"tc_good", "tc_bad"} <= _assistant_tool_call_ids(turn2_request)

  @pytest.mark.asyncio
  async def test_all_malformed_batch_every_tool_result_in_next_turn_is_paired(self):
    fake = _FakeLiteLLM([
      _tool_call_stream([("tc_bad", "read_file", UNTERMINATED)]),
      _text_stream("Retried."),
      _text_stream("Summary."),
    ])

    turn2_request = await _run_two_turns(fake, _RecordingExecutor())

    assert len(fake.calls) == 3
    assert _unpaired_tool_results(turn2_request) == [], (
      "every tool result in the next turn's history must follow an assistant tool_calls entry with its id"
    )
    assert _unanswered_tool_calls(turn2_request) == [], (
      "every assistant tool_calls entry in the next turn's history must have a later tool result"
    )
    assert "tc_bad" in _assistant_tool_call_ids(turn2_request), (
      "the malformed call must remain in the assistant tool_calls history"
    )

  @pytest.mark.asyncio
  async def test_valid_batch_every_tool_result_in_next_turn_is_paired(self):
    fake = _FakeLiteLLM([
      _tool_call_stream([("tc_ok", "list_dir", '{"path": "/vault"}')]),
      _text_stream("Scanned."),
      _text_stream("Summary."),
    ])

    turn2_request = await _run_two_turns(fake, _RecordingExecutor())

    assert _unpaired_tool_results(turn2_request) == []
    assert _unanswered_tool_calls(turn2_request) == []
    assert _tool_messages(turn2_request) == {"tc_ok": "EXECUTED list_dir"}


# ---------------------------------------------------------------------------
# AC 5 — text alongside tool calls: history and tool_use_emitted unchanged
# ---------------------------------------------------------------------------


def _text_then_tool_calls(text: str, calls: list[tuple[str, str, str]]) -> list[MagicMock]:
  return [_chunk(_delta(content=text))] + _tool_call_stream(calls)


class TestTextWithToolCallsUnchanged:
  @pytest.mark.asyncio
  async def test_text_plus_valid_call_next_turn_history_matches_pre_change_behavior(self):
    fake = _FakeLiteLLM([
      _text_then_tool_calls("Let me look.", [("tc_ok", "list_dir", '{"path": "/vault"}')]),
      _text_stream("Scanned."),
      _text_stream("Summary."),
    ])

    turn2_request = await _run_two_turns(fake, _RecordingExecutor())

    # Captured from the pre-change code (aa7229e) for this exact scenario.
    non_system = [m for m in turn2_request["messages"] if m["role"] != "system"]
    assert non_system == [
      {"role": "user", "content": "scan the vault"},
      {
        "role": "assistant",
        "content": None,
        "tool_calls": [
          {
            "id": "tc_ok",
            "type": "function",
            "function": {"name": "list_dir", "arguments": '{"path": "/vault"}'},
          }
        ],
      },
      {"role": "tool", "tool_call_id": "tc_ok", "content": "EXECUTED list_dir"},
      {"role": "assistant", "content": "Scanned."},
      {"role": "user", "content": "now summarize"},
    ]


async def _tool_use_emitted_payloads(fake: _FakeLiteLLM) -> list[list]:
  """Run one turn and return the data of every tool_use_emitted engine-bus event."""
  from sr2.orchestrator import SR2

  sr2 = SR2(
    pipeline_config=make_minimal_config(),
    llm=LiteLLMCallable("test-model"),
    token_counter=CharacterTokenCounter(),
    tool_executor=_RecordingExecutor(),
  )
  collected: list = []
  sr2.bus.subscribe("tool_use_emitted", lambda e: collected.append(e))
  with patch("litellm.acompletion", new=fake):
    [e async for e in sr2.turn(make_user_input("scan the vault"))]
  return [e.data for e in collected]


class TestToolUseEmittedBusPayload:
  @pytest.mark.asyncio
  async def test_text_plus_valid_call_payload_holds_only_tool_use_blocks(self):
    fake = _FakeLiteLLM([
      _text_then_tool_calls("Let me look.", [("tc_ok", "list_dir", '{"path": "/vault"}')]),
      _text_stream("Scanned."),
    ])

    payloads = await _tool_use_emitted_payloads(fake)

    assert len(payloads) == 1
    payload = payloads[0]
    assert all(isinstance(b, ToolUseBlock) for b in payload), (
      f"tool_use_emitted data must be a list of ToolUseBlock only, got {[type(b).__name__ for b in payload]}"
    )
    assert [(b.id, b.name, b.input) for b in payload] == [("tc_ok", "list_dir", {"path": "/vault"})]

  @pytest.mark.asyncio
  async def test_text_plus_mixed_calls_payload_holds_only_tool_use_blocks_including_malformed(self):
    fake = _FakeLiteLLM([
      _text_then_tool_calls("Let me look.", [
        ("tc_good", "list_dir", '{"path": "/vault"}'),
        ("tc_bad", "read_file", UNTERMINATED),
      ]),
      _text_stream("Scanned."),
    ])

    payloads = await _tool_use_emitted_payloads(fake)

    assert len(payloads) == 1
    payload = payloads[0]
    assert all(isinstance(b, ToolUseBlock) for b in payload), (
      f"tool_use_emitted data must be a list of ToolUseBlock only, got {[type(b).__name__ for b in payload]}"
    )
    assert {b.id for b in payload} == {"tc_good", "tc_bad"}, (
      "every tool call, malformed ones included, must be in tool_use_emitted"
    )
