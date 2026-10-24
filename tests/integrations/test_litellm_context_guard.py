"""obsidian-pt2k.1 — LiteLLMCallable sheds conversation instead of dying.

The endpoint (llm-relay) admits a request only when its estimate fits a
node; a long run used to crash on the resulting 400. With ``context_limit``
set, ``LiteLLMCallable`` must:

  1. shed the oldest turns before sending when the estimate is over limit;
  2. on a context-length rejection, shed and retry (bounded), then raise
     only when nothing droppable is left;
  3. leave non-context errors and unlimited callers untouched.

``litellm.acompletion`` is the only mocked boundary.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock
from sr2.protocols.llm import CompletionRequest

from .test_litellm_callable import (
  _make_stream_chunk,
  _mock_litellm_response,
  _async_gen,
)


def _big_text(n: int) -> str:
  """Deterministic filler whose JSON-encoded length is roughly n chars."""
  return "x" * n


def _request(turns: int, size: int = 4000) -> CompletionRequest:
  messages = []
  for i in range(turns):
    role = "user" if i % 2 == 0 else "assistant"
    messages.append(Message(role=role, content=[TextBlock(text=_big_text(size) + f" turn{i}")]))
  return CompletionRequest(messages=messages)


def _wire_tokens(messages: list[dict], tools=None) -> int:
  from sr2.integrations.context_guard import estimate_request_tokens

  return estimate_request_tokens(messages, tools)


def _capture_fake(responses):
  """Fake acompletion recording kwargs; yields the given responses/raises."""
  calls: list[dict] = []
  it = iter(responses)

  async def fake(**kwargs):
    calls.append(kwargs)
    outcome = next(it)
    if isinstance(outcome, Exception):
      raise outcome
    return outcome

  return fake, calls


class TestProactiveShed:
  async def test_request_over_limit_is_shed_before_send(self):
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test", context_limit=2000)
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(_request(turns=8, size=4000))
    sent = calls[0]["messages"]
    assert len(sent) < 8
    assert _wire_tokens(sent, calls[0].get("tools")) <= 2000

  async def test_request_under_limit_is_untouched(self):
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test", context_limit=200_000)
    request = _request(turns=2, size=50)
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(request)
    assert len(calls[0]["messages"]) == 2

  async def test_no_context_limit_never_sheds(self):
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(_request(turns=6, size=4000))
    assert len(calls[0]["messages"]) == 6

  async def test_output_tokens_count_toward_the_limit(self):
    # max_tokens is reserved: with a tight limit the input must shrink so
    # that estimate + max_tokens fits, mirroring the relay's admission rule.
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([resp])
    callable_ = LiteLLMCallable(model="openai/test", context_limit=6000, max_tokens=4000)
    with patch("litellm.acompletion", new=fake):
      await callable_.complete(_request(turns=8, size=4000))
    sent = calls[0]["messages"]
    assert len(sent) < 8
    assert _wire_tokens(sent, calls[0].get("tools")) + 4000 <= 6000

  async def test_shed_applies_to_streams_too(self):
    chunk = _make_stream_chunk(content="ok")
    fake_calls: list[dict] = []

    async def fake(**kwargs):
      fake_calls.append(kwargs)

      async def gen():
        yield chunk

      return gen()

    callable_ = LiteLLMCallable(model="openai/test", context_limit=2000)
    with patch("litellm.acompletion", new=fake):
      events = [e async for e in callable_.stream(_request(turns=8, size=4000))]
    assert len(fake_calls[0]["messages"]) < 8
    assert any(e.type == "text" for e in events)


class TestReactiveShed:
  async def test_context_rejection_sheds_and_retries(self):
    overflow = RuntimeError(
      "litellm.BadRequestError: OpenAIException - no node has context large"
      " enough for model 'qwen3.8'"
    )
    resp = _mock_litellm_response()
    fake, calls = _capture_fake([overflow, resp])
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      out = await callable_.complete(_request(turns=4, size=100))
    assert len(calls) == 2
    assert len(calls[1]["messages"]) < len(calls[0]["messages"])
    assert out.content[0].text == "Hi there"

  async def test_non_context_error_propagates_after_one_attempt(self):
    boom = RuntimeError("litellm.BadRequestError: bad tool schema")
    fake, calls = _capture_fake([boom])
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      with pytest.raises(RuntimeError, match="bad tool schema"):
        await callable_.complete(_request(turns=2, size=10))
    assert len(calls) == 1

  async def test_gives_up_when_nothing_left_to_shed(self):
    overflow = RuntimeError("no node has context large enough for model 'x'")
    fake, calls = _capture_fake([overflow] * 10)
    callable_ = LiteLLMCallable(model="openai/test")
    with patch("litellm.acompletion", new=fake):
      with pytest.raises(RuntimeError, match="no node has context"):
        await callable_.complete(CompletionRequest(
          messages=[Message(role="user", content=[TextBlock(text="huge single turn")])]
        ))
    # one initial attempt, no pointless retries on an undroppable request
    assert len(calls) == 1
