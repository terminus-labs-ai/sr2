"""obsidian-pt2k.1 — context-window guard for OpenAI-compatible endpoints.

A long run grows the conversation until the serving endpoint's admission
check fails with a context-length 400 ("no node has context large enough
for model 'qwen3.8'" from llm-relay). The run must shed its oldest turns
instead of dying mid-flight.

Pure-function tests for ``sr2.integrations.context_guard``; the wire-in of
the guard into ``LiteLLMCallable`` lives in ``test_litellm_context_guard.py``.
"""

from __future__ import annotations

import json
import math

from sr2.integrations.context_guard import (
  estimate_request_tokens,
  is_context_overflow_error,
  requested_output_tokens,
  shed_oldest_messages,
)


def _msg(role: str, content: str = "x", **extra) -> dict:
  return {"role": role, "content": content, **extra}


def _assistant_tool_call(tool_id: str = "tu_1") -> dict:
  return {
    "role": "assistant",
    "content": None,
    "tool_calls": [
      {"id": tool_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}
    ],
  }


def _tool_result(tool_id: str = "tu_1") -> dict:
  return {"role": "tool", "tool_call_id": tool_id, "content": "ok"}


# ---------------------------------------------------------------------------
# estimate_request_tokens — mirrors llm-relay router.estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimate:
  def test_mirrors_relay_formula(self):
    messages = [_msg("user", "hello")]
    tools = [{"type": "function", "function": {"name": "t"}}]
    expected = math.ceil(
      (len(json.dumps(messages)) + len(json.dumps(tools))) / 4
    )
    assert estimate_request_tokens(messages, tools) == expected

  def test_no_tools_counts_messages_only(self):
    messages = [_msg("user", "hello")]
    expected = math.ceil(len(json.dumps(messages)) / 4)
    assert estimate_request_tokens(messages, None) == expected


class TestRequestedOutputTokens:
  def test_max_completion_tokens_wins(self):
    assert requested_output_tokens({"max_tokens": 100, "max_completion_tokens": 50}) == 50

  def test_max_tokens_fallback(self):
    assert requested_output_tokens({"max_tokens": 100}) == 100

  def test_absent_is_zero(self):
    assert requested_output_tokens({}) == 0

  def test_null_counts_as_absent(self):
    assert requested_output_tokens({"max_tokens": None}) == 0


# ---------------------------------------------------------------------------
# shed_oldest_messages
# ---------------------------------------------------------------------------


class TestShedOldest:
  def test_keeps_system_and_drops_oldest_non_system(self):
    messages = [
      _msg("system", "sys"),
      _msg("user", "first"),
      _msg("assistant", "reply"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [messages[0], messages[2], messages[3]]

  def test_drops_tool_call_group_together(self):
    messages = [
      _msg("system", "sys"),
      _assistant_tool_call("tu_1"),
      _tool_result("tu_1"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [messages[0], messages[3]]

  def test_drops_orphaned_leading_tool_results(self):
    # A tool result whose parent assistant was already dropped must never
    # survive at the head of the request — upstream rejects the pairing.
    messages = [
      _msg("system", "sys"),
      _tool_result("tu_gone"),
      _msg("user", "now"),
    ]
    assert shed_oldest_messages(messages) == [messages[0], messages[2]]

  def test_never_drops_the_current_turn(self):
    messages = [_msg("system", "sys"), _msg("user", "only")]
    assert shed_oldest_messages(messages) is None

  def test_none_when_group_would_consume_current_turn(self):
    messages = [_msg("system", "sys"), _assistant_tool_call(), _tool_result()]
    assert shed_oldest_messages(messages) is None

  def test_no_system_message(self):
    messages = [_msg("user", "first"), _msg("user", "now")]
    assert shed_oldest_messages(messages) == [messages[1]]

  def test_empty_returns_none(self):
    assert shed_oldest_messages([]) is None


# ---------------------------------------------------------------------------
# is_context_overflow_error
# ---------------------------------------------------------------------------


class TestIsContextOverflowError:
  def test_matches_llm_relay_admission_message(self):
    msg = "no node has context large enough for model 'qwen3.8'"
    assert is_context_overflow_error(ValueError(msg))

  def test_matches_openai_style_message(self):
    msg = "This model's maximum context length is 131072 tokens"
    assert is_context_overflow_error(ValueError(msg))

  def test_matches_error_code_spelling(self):
    assert is_context_overflow_error("context_length_exceeded")

  def test_rejects_unrelated_bad_request(self):
    assert not is_context_overflow_error(ValueError("old_string not found in file"))

  def test_accepts_raw_string(self):
    assert is_context_overflow_error("no node has context large enough for model 'x'")
