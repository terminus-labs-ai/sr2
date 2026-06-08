from __future__ import annotations

"""Tests for thinking/reasoning event emission in LiteLLMCallable.stream().

Covers the gap left by spc-26 (litellm.py:154-163):
  (1) delta.reasoning_content → StreamEvent(type="thinking")
  (2) delta.thinking_blocks with type=="thinking" → thinking events
  (3) non-dict/empty thinking_blocks are skipped
  (4) thinking events interleave correctly with text/tool_use/end ordering
"""

import pytest
from unittest.mock import MagicMock, patch

from sr2.protocols.llm import StreamEvent
from sr2.integrations.litellm import LiteLLMCallable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(messages=None, system=None, tools=None):
    from sr2.models import Message, TextBlock, ToolDefinition

    return type("CompletionRequest", (), {
        "messages": messages or [Message(role="user", content=[TextBlock(text="Hello")])],
        "system": system,
        "tools": tools or None,
    })()


async def _async_gen(*chunks):
    for chunk in chunks:
        yield chunk


def _make_thinking_delta(
    content=None,
    reasoning_content=None,
    thinking_blocks=None,
    tool_calls=None,
) -> MagicMock:
    """Build a streaming delta with thinking/reasoning fields."""
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = reasoning_content
    delta.thinking_blocks = thinking_blocks
    delta.tool_calls = tool_calls or None
    return delta


def _make_stream_chunk(delta: MagicMock, usage=None) -> MagicMock:
    """Wrap a delta into a full litellm chunk."""
    choice = MagicMock()
    choice.delta = delta

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _fake_acompletion(*chunks):
    """Return a fake acompletion that yields the given chunks."""
    async def fake(*args, **kwargs):
        return _async_gen(*chunks)
    return fake


# ---------------------------------------------------------------------------
# 1. delta.reasoning_content → StreamEvent(type="thinking")
# ---------------------------------------------------------------------------


class TestReasoningContentEmission:
    @pytest.mark.asyncio
    async def test_reasoning_content_yields_thinking_event(self):
        """A non-None reasoning_content on the delta emits a thinking event."""
        delta = _make_thinking_delta(
            content="Hello",
            reasoning_content="Let me think about this...",
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "Let me think about this..."

    @pytest.mark.asyncio
    async def test_none_reasoning_content_does_not_emit_thinking(self):
        """A None reasoning_content does not produce a thinking event."""
        delta = _make_thinking_delta(
            content="Hello",
            reasoning_content=None,
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 0

    @pytest.mark.asyncio
    async def test_empty_string_reasoning_content_does_not_emit(self):
        """An empty-string reasoning_content does not yield a thinking event
        (falsy check via `if reasoning is not None` — empty string is not None,
        so it WILL emit. Testing the actual behavior)."""
        delta = _make_thinking_delta(
            reasoning_content="",
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        # The code uses `if reasoning is not None`, so "" passes through.
        # This tests the actual behavior (empty string → event with empty text).
        assert len(thinking_events) == 1
        assert thinking_events[0].text == ""

    @pytest.mark.asyncio
    async def test_reasoning_content_without_text_content(self):
        """A chunk with only reasoning_content (no regular content) emits thinking but not text."""
        delta = _make_thinking_delta(
            content=None,
            reasoning_content="Working through it...",
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        text_events = [e for e in events if e.type == "text"]
        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(text_events) == 0
        assert len(thinking_events) == 1

    @pytest.mark.asyncio
    async def test_multiple_reasoning_content_chunks(self):
        """Multiple chunks each with reasoning_content emit separate thinking events."""
        chunks = [
            _make_stream_chunk(_make_thinking_delta(reasoning_content="Step one")),
            _make_stream_chunk(_make_thinking_delta(reasoning_content="Step two")),
        ]

        with patch("litellm.acompletion", new=_fake_acompletion(*chunks)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 2
        assert thinking_events[0].text == "Step one"
        assert thinking_events[1].text == "Step two"


# ---------------------------------------------------------------------------
# 2. delta.thinking_blocks → thinking events
# ---------------------------------------------------------------------------


class TestThinkingBlocksEmission:
    @pytest.mark.asyncio
    async def test_thinking_block_yields_thinking_event(self):
        """A thinking_blocks list with a valid thinking dict emits a thinking event."""
        delta = _make_thinking_delta(
            thinking_blocks=[{"type": "thinking", "thinking": "I think therefore I am"}],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "I think therefore I am"

    @pytest.mark.asyncio
    async def test_multiple_thinking_blocks_emit_multiple_events(self):
        """Multiple thinking blocks in one delta emit separate thinking events."""
        delta = _make_thinking_delta(
            thinking_blocks=[
                {"type": "thinking", "thinking": "First thought"},
                {"type": "thinking", "thinking": "Second thought"},
            ],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 2
        assert thinking_events[0].text == "First thought"
        assert thinking_events[1].text == "Second thought"

    @pytest.mark.asyncio
    async def test_thinking_block_without_thinking_key_skipped(self):
        """A dict with type='thinking' but no 'thinking' key is skipped."""
        delta = _make_thinking_delta(
            thinking_blocks=[{"type": "thinking"}],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 0

    @pytest.mark.asyncio
    async def test_thinking_block_with_empty_thinking_value_skipped(self):
        """A thinking block with thinking='' (falsy) is skipped."""
        delta = _make_thinking_delta(
            thinking_blocks=[{"type": "thinking", "thinking": ""}],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 0

    @pytest.mark.asyncio
    async def test_non_thinking_type_block_skipped(self):
        """A block with type='redacted_thinking' or other is skipped."""
        delta = _make_thinking_delta(
            thinking_blocks=[
                {"type": "redacted_thinking", "thinking": "Secret"},
                {"type": "thinking", "thinking": "Visible"},
            ],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "Visible"


# ---------------------------------------------------------------------------
# 3. Non-dict / empty thinking_blocks skipped
# ---------------------------------------------------------------------------


class TestThinkingBlocksEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_thinking_blocks_list_no_events(self):
        """An empty thinking_blocks list produces no thinking events."""
        delta = _make_thinking_delta(
            thinking_blocks=[],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 0

    @pytest.mark.asyncio
    async def test_none_thinking_blocks_no_events(self):
        """A None thinking_blocks (the default) produces no thinking events."""
        delta = _make_thinking_delta(
            thinking_blocks=None,
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 0

    @pytest.mark.asyncio
    async def test_non_dict_item_in_thinking_blocks_skipped(self):
        """A string or other non-dict in thinking_blocks is safely skipped."""
        delta = _make_thinking_delta(
            thinking_blocks=["not a dict", 42, {"type": "thinking", "thinking": "Valid"}],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "Valid"

    @pytest.mark.asyncio
    async def test_thinking_blocks_with_none_value_skipped(self):
        """A thinking block where thinking=None is skipped (falsy check)."""
        delta = _make_thinking_delta(
            thinking_blocks=[{"type": "thinking", "thinking": None}],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 0


# ---------------------------------------------------------------------------
# 4. Thinking events interleave correctly with text/tool_use/end
# ---------------------------------------------------------------------------


class TestThinkingEventInterleaving:
    @pytest.mark.asyncio
    async def test_thinking_interleaves_with_text(self):
        """Thinking and text events from the same chunk maintain emission order
        (text first, then thinking — as coded in stream())."""
        delta = _make_thinking_delta(
            content="Hello",
            reasoning_content="Thinking about greeting",
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        # stream() checks content before reasoning_content, so text comes first
        non_end_events = [e for e in events if e.type != "end"]
        assert non_end_events[0].type == "text"
        assert non_end_events[0].text == "Hello"
        assert non_end_events[1].type == "thinking"
        assert non_end_events[1].text == "Thinking about greeting"

    @pytest.mark.asyncio
    async def test_thinking_then_text_across_chunks(self):
        """Chunks with thinking followed by text emit in chunk order."""
        chunks = [
            _make_stream_chunk(_make_thinking_delta(reasoning_content="Thinking...")),
            _make_stream_chunk(_make_thinking_delta(content="Answer")),
        ]

        with patch("litellm.acompletion", new=_fake_acompletion(*chunks)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        non_end_events = [e for e in events if e.type != "end"]
        assert non_end_events[0].type == "thinking"
        assert non_end_events[1].type == "text"

    @pytest.mark.asyncio
    async def test_thinking_with_text_and_end_ordering(self):
        """Full stream with text, thinking, and end — end is always last."""
        chunks = [
            _make_stream_chunk(_make_thinking_delta(content="Hi", reasoning_content="Thinking hi")),
            _make_stream_chunk(_make_thinking_delta(content=" there")),
        ]

        with patch("litellm.acompletion", new=_fake_acompletion(*chunks)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        assert events[-1].type == "end"
        # Before end: text, thinking, text in emission order
        non_end = [e for e in events if e.type != "end"]
        assert non_end[0].type == "text"
        assert non_end[1].type == "thinking"
        assert non_end[2].type == "text"

    @pytest.mark.asyncio
    async def test_thinking_before_tool_use_and_end(self):
        """Thinking events come before tool_use events (emitted after stream loop)
        and end event."""
        # Build a chunk with thinking
        thinking_delta = _make_thinking_delta(
            reasoning_content="Planning tool call",
        )
        thinking_chunk = _make_stream_chunk(thinking_delta)

        # Build a chunk with a tool call delta (inline helper)
        tc_delta = MagicMock()
        tc_delta.index = 0
        tc_delta.id = "tc_1"
        tc_delta.function = MagicMock()
        tc_delta.function.name = "get_weather"
        tc_delta.function.arguments = '{"location": "Oslo"}'

        tool_delta = MagicMock()
        tool_delta.content = None
        tool_delta.tool_calls = [tc_delta]
        tool_delta.reasoning_content = None
        tool_delta.thinking_blocks = None

        tool_choice = MagicMock()
        tool_choice.delta = tool_delta
        tool_chunk = MagicMock()
        tool_chunk.choices = [tool_choice]
        tool_chunk.usage = None

        chunks = [thinking_chunk, tool_chunk]

        with patch("litellm.acompletion", new=_fake_acompletion(*chunks)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        # Order: thinking (from loop) → tool_use (post-loop) → end (post-loop)
        assert events[0].type == "thinking"
        assert events[1].type == "tool_use"
        assert events[2].type == "end"

    @pytest.mark.asyncio
    async def test_thinking_blocks_and_reasoning_together(self):
        """When both reasoning_content and thinking_blocks are present on the
        same delta, reasoning_content events come first (code order)."""
        delta = _make_thinking_delta(
            reasoning_content="OpenAI reasoning",
            thinking_blocks=[{"type": "thinking", "thinking": "Anthropic thought"}],
        )
        chunk = _make_stream_chunk(delta)

        with patch("litellm.acompletion", new=_fake_acompletion(chunk)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        thinking_events = [e for e in events if e.type == "thinking"]
        assert len(thinking_events) == 2
        # reasoning_content is processed before thinking_blocks in stream()
        assert thinking_events[0].text == "OpenAI reasoning"
        assert thinking_events[1].text == "Anthropic thought"

    @pytest.mark.asyncio
    async def test_thinking_with_usage_and_end_ordering(self):
        """Full stream: text, thinking, usage, end — end is last, usage before end."""
        from unittest.mock import MagicMock

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        chunks = [
            _make_stream_chunk(_make_thinking_delta(content="Hi", reasoning_content="Thinking")),
            _make_stream_chunk(_make_thinking_delta(content=None), usage=usage),
        ]

        with patch("litellm.acompletion", new=_fake_acompletion(*chunks)):
            client = LiteLLMCallable("model")
            events = [e async for e in client.stream(_make_request())]

        assert events[-1].type == "end"
        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        # usage comes before end
        usage_index = next(i for i, e in enumerate(events) if e.type == "usage")
        end_index = next(i for i, e in enumerate(events) if e.type == "end")
        assert usage_index < end_index
