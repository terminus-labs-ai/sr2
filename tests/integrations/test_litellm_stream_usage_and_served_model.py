"""obsidian-la2z: streamed usage and served-model reporting.

The relay only sends its usage chunk when the streaming request asks for it
with ``stream_options={"include_usage": True}``, and the pool name in the
request is not necessarily the model id the backend served. These tests pin
both ends of the contract at the litellm boundary:

* the streaming request carries stream_options with include_usage defaulting
  to True, and a caller-supplied stream_options keeps its own keys;
* the non-streaming request carries no stream_options;
* the first non-empty chunk.model reaches the terminal end event as
  meta["served_model"], and is None when no chunk reports a model.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock
from sr2.protocols.llm import CompletionRequest, StreamEvent


# ---------------------------------------------------------------------------
# Helpers — deliberately independent of test_litellm_callable.py's helpers so
# this file pins only what it tests.
# ---------------------------------------------------------------------------


def _make_request() -> CompletionRequest:
  return CompletionRequest(messages=[Message(role="user", content=[TextBlock(text="Hi")])])


async def _async_gen(*chunks):
  for chunk in chunks:
    yield chunk


def _recording_acompletion(chunks):
  """fake litellm.acompletion returning an async iterator, recording kwargs."""
  calls: list[dict] = []

  async def fake(*args, **kwargs):
    calls.append(kwargs)
    return _async_gen(*chunks)

  fake.calls = calls
  return fake


def _chunk(content=None, *, model=None, finish_reason=None, usage=None):
  delta = MagicMock()
  delta.content = content
  delta.tool_calls = None
  delta.reasoning_content = None
  delta.thinking_blocks = None

  choice = MagicMock()
  choice.delta = delta
  choice.finish_reason = finish_reason

  chunk = MagicMock()
  # The usage-only trailing chunk of the OpenAI include_usage shape carries
  # no choices at all.
  chunk.choices = [] if (content is None and finish_reason is None and usage is not None) else [choice]
  chunk.model = model
  chunk.usage = usage
  return chunk


def _usage(prompt_tokens: int, completion_tokens: int) -> MagicMock:
  usage = MagicMock()
  usage.prompt_tokens = prompt_tokens
  usage.completion_tokens = completion_tokens
  return usage


async def _drive(chunks, **callable_kwargs) -> tuple[list[StreamEvent], dict]:
  fake = _recording_acompletion(chunks)
  with patch("litellm.acompletion", new=fake):
    client = LiteLLMCallable("model", **callable_kwargs)
    events = [e async for e in client.stream(_make_request())]
  assert len(fake.calls) == 1
  return events, fake.calls[0]


def _end(events: list[StreamEvent]) -> StreamEvent:
  ends = [e for e in events if e.type == "end"]
  assert len(ends) == 1
  assert events[-1].type == "end"
  assert ends[0].meta is not None
  return ends[0]


# ---------------------------------------------------------------------------
# 1. The streaming request asks for usage
# ---------------------------------------------------------------------------


class TestStreamOptionsOnTheRequest:
  @pytest.mark.asyncio
  async def test_stream_request_carries_include_usage(self):
    _, kwargs = await _drive([_chunk(content="Hi", model="m-1")])

    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}

  @pytest.mark.asyncio
  async def test_caller_stream_options_keys_win_include_usage_defaults(self):
    """A caller-supplied stream_options keeps its keys; only include_usage defaults."""
    _, kwargs = await _drive(
      [_chunk(content="Hi", model="m-1")],
      stream_options={"continuous_usage_stats": True},
    )

    assert kwargs["stream_options"] == {
      "include_usage": True,
      "continuous_usage_stats": True,
    }

  @pytest.mark.asyncio
  async def test_caller_can_still_turn_include_usage_off(self):
    _, kwargs = await _drive(
      [_chunk(content="Hi", model="m-1")],
      stream_options={"include_usage": False},
    )

    assert kwargs["stream_options"] == {"include_usage": False}

  @pytest.mark.asyncio
  async def test_non_streaming_request_unchanged(self):
    fake = _recording_acompletion([])
    with patch("litellm.acompletion", new=fake):
      client = LiteLLMCallable("model")
      try:
        await client.complete(_make_request())
      except Exception:
        pass  # a bare MagicMock response may fail post-processing; the request is what is pinned

    assert len(fake.calls) == 1
    assert "stream" not in fake.calls[0]
    assert "stream_options" not in fake.calls[0]


# ---------------------------------------------------------------------------
# 2. Served model on the end event
# ---------------------------------------------------------------------------


class TestServedModelOnEndEvent:
  @pytest.mark.asyncio
  async def test_stream_with_usage_and_model_reports_both(self):
    """The relay shape: chunks carry the served model; only the last carries usage."""
    events, _ = await _drive([
      _chunk(content="Hello", model="qwen3.8-flash-next"),
      _chunk(content=None, model="qwen3.8-flash-next", finish_reason="stop"),
      _chunk(model="qwen3.8-flash-next", usage=_usage(14, 1)),
    ])

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage.input_tokens == 14
    assert usage_events[0].usage.output_tokens == 1

    end = _end(events)
    assert end.meta["finish_reason"] == "stop"
    assert end.meta["served_model"] == "qwen3.8-flash-next"

  @pytest.mark.asyncio
  async def test_first_non_empty_model_wins(self):
    events, _ = await _drive([
      _chunk(content="a", model=None),
      _chunk(content="b", model=""),
      _chunk(content="c", model="served-a"),
      _chunk(content="d", model="served-b"),
    ])

    assert _end(events).meta["served_model"] == "served-a"

  @pytest.mark.asyncio
  async def test_stream_without_usage_or_model_reports_neither(self):
    """Absence is not an error: no crash, both fields null."""
    events, _ = await _drive([
      _chunk(content="Hello", model=None),
      _chunk(content=" world", model=None, finish_reason="stop"),
    ])

    assert [e for e in events if e.type == "usage"] == []
    end = _end(events)
    assert end.meta["served_model"] is None
    assert end.meta["finish_reason"] == "stop"

  @pytest.mark.asyncio
  async def test_model_present_only_on_trailing_usage_chunk(self):
    events, _ = await _drive([
      _chunk(content="Hi", model=None),
      _chunk(model="served-x", usage=_usage(3, 4)),
    ])

    end = _end(events)
    assert end.meta["served_model"] == "served-x"
    assert end.meta["finish_reason"] is None

  @pytest.mark.asyncio
  async def test_empty_stream_end_event_has_both_keys(self):
    events, _ = await _drive([])

    end = _end(events)
    assert end.meta["served_model"] is None
    assert end.meta["finish_reason"] is None

  @pytest.mark.asyncio
  async def test_served_model_is_per_call(self):
    """A second stream() must not inherit the first call's served model."""
    sequences = [
      [_chunk(content="a", model="served-one", finish_reason="stop")],
      [_chunk(content="b", model=None)],
    ]
    calls = iter(sequences)

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*next(calls))

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      first = [e async for e in client.stream(_make_request())]
      second = [e async for e in client.stream(_make_request())]

    assert _end(first).meta["served_model"] == "served-one"
    assert _end(second).meta["served_model"] is None
