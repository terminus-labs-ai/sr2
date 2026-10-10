from __future__ import annotations

"""Tests for sr2-113 — LiteLLMCallable serializes ThinkingBlock replay.

Covers AC 2:
  (1) assistant message with tool_calls: ThinkingBlock → reasoning_content,
      never in visible content
  (2) plain-text assistant message: ThinkingBlock → reasoning_content,
      never in visible content
  (3) multiple ThinkingBlocks on one message are joined in order
  (4) no ThinkingBlock → no reasoning_content key (both branches)
  (5) user messages with ThinkingBlock never get reasoning_content
"""

import json

import pytest

from sr2.integrations.litellm import LiteLLMCallable
from sr2.models import Message, TextBlock, ThinkingBlock, ToolUseBlock
from sr2.protocols.llm import CompletionRequest


def _make_request(messages: list[Message], system=None, tools=None) -> CompletionRequest:
    return CompletionRequest(messages=messages, system=system, tools=tools)


# ---------------------------------------------------------------------------
# 1. assistant message with tool_calls
# ---------------------------------------------------------------------------


class TestThinkingBlockWithToolCalls:
    def test_thinking_serialized_as_reasoning_content(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(text="Planning the tool call"),
                ToolUseBlock(id="tu_1", name="get_weather", input={"location": "Oslo"}),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["reasoning_content"] == "Planning the tool call"
        # Thinking text must never appear in visible content.
        assert "Planning" not in (out[0].get("content") or "")
        assert out[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_thinking_with_text_and_tool_calls_keeps_text_content(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(text="Thinking first"),
                TextBlock(text="I'll check the weather."),
                ToolUseBlock(id="tu_1", name="get_weather", input={}),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["content"] == "I'll check the weather."
        assert out[0]["reasoning_content"] == "Thinking first"
        assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {}

    def test_thinking_only_tool_call_message_content_is_none(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(text="Only thinking"),
                ToolUseBlock(id="tu_1", name="tool_a", input={"x": 1}),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["content"] is None
        assert out[0]["reasoning_content"] == "Only thinking"

    def test_no_thinking_block_no_reasoning_content_key(self):
        msg = Message(
            role="assistant",
            content=[
                TextBlock(text="I'll check."),
                ToolUseBlock(id="tu_1", name="get_weather", input={}),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert "reasoning_content" not in out[0]
        assert out[0]["content"] == "I'll check."


# ---------------------------------------------------------------------------
# 2. plain-text assistant message
# ---------------------------------------------------------------------------


class TestThinkingBlockPlainText:
    def test_thinking_serialized_as_reasoning_content(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(text="Working through the answer"),
                TextBlock(text="The answer is 42."),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "The answer is 42."
        assert out[0]["reasoning_content"] == "Working through the answer"

    def test_thinking_only_message_content_is_empty_string(self):
        msg = Message(
            role="assistant",
            content=[ThinkingBlock(text="Just thinking")],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["content"] == ""
        assert out[0]["reasoning_content"] == "Just thinking"

    def test_no_thinking_block_no_reasoning_content_key(self):
        msg = Message(
            role="assistant",
            content=[TextBlock(text="Plain reply")],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert "reasoning_content" not in out[0]
        assert out[0]["content"] == "Plain reply"


# ---------------------------------------------------------------------------
# 3. multiple ThinkingBlocks joined in order
# ---------------------------------------------------------------------------


class TestMultipleThinkingBlocks:
    def test_multiple_thinking_blocks_joined_in_order_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(text="First thought "),
                ThinkingBlock(text="second thought"),
                ToolUseBlock(id="tu_1", name="tool_a", input={}),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["reasoning_content"] == "First thought second thought"

    def test_multiple_thinking_blocks_joined_in_order_plain_text(self):
        msg = Message(
            role="assistant",
            content=[
                ThinkingBlock(text="a"),
                TextBlock(text="visible"),
                ThinkingBlock(text="b"),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["reasoning_content"] == "ab"
        assert out[0]["content"] == "visible"


# ---------------------------------------------------------------------------
# 4. non-assistant messages never carry reasoning_content
# ---------------------------------------------------------------------------


class TestNonAssistantMessages:
    def test_user_message_with_thinking_block_never_gets_reasoning_content(self):
        msg = Message(
            role="user",
            content=[
                ThinkingBlock(text="User-side thinking"),
                TextBlock(text="Hello"),
            ],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(_make_request([msg]))

        assert out[0]["role"] == "user"
        assert "reasoning_content" not in out[0]
        # Thinking text is still excluded from visible content.
        assert out[0]["content"] == "Hello"

    def test_system_message_unchanged(self):
        request = _make_request(
            [Message(role="user", content=[TextBlock(text="Hi")])],
            system=[TextBlock(text="You are helpful.")],
        )
        client = LiteLLMCallable("model")
        out = client._build_messages(request)

        assert out[0] == {"role": "system", "content": "You are helpful."}
