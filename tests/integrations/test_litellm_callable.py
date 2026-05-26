from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import Message, TextBlock, ToolDefinition, ToolResultBlock, ToolUseBlock, TokenUsage
from sr2.protocols.llm import (
  CompletionRequest,
  CompletionResponse,
  LLMCallable,
  StreamEvent,
)
from sr2.integrations.litellm import LiteLLMCallable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(role: str, *texts: str) -> Message:
  return Message(role=role, content=[TextBlock(text=t) for t in texts])


def _make_request(
  messages: list[Message] | None = None,
  system: list[TextBlock] | None = None,
  tools: list[ToolDefinition] | None = None,
) -> CompletionRequest:
  return CompletionRequest(
    messages=messages or [_make_message("user", "Hello")],
    system=system,
    tools=tools,
  )


def _make_tool(
  name: str = "get_weather",
  description: str = "Get the weather for a location",
  input_schema: dict | None = None,
) -> ToolDefinition:
  return ToolDefinition(
    name=name,
    description=description,
    input_schema=input_schema or {"type": "object", "properties": {"location": {"type": "string"}}},
  )


def _make_tool_use_message(
  tool_id: str = "tu_1",
  tool_name: str = "get_weather",
  tool_input: dict | None = None,
) -> Message:
  block = ToolUseBlock(
    id=tool_id,
    name=tool_name,
    input=tool_input or {"location": "Stockholm"},
  )
  return Message(role="assistant", content=[block])


def _make_tool_result_message(
  tool_use_id: str = "tu_1",
  content: str | list[TextBlock] = "Sunny, 22°C",
  is_error: bool = False,
) -> Message:
  block = ToolResultBlock(
    tool_use_id=tool_use_id,
    content=content,
    is_error=is_error,
  )
  return Message(role="user", content=[block])


def _mock_litellm_response(
  id: str = "resp-1",
  content: str = "Hi there",
  finish_reason: str = "stop",
  prompt_tokens: int = 10,
  completion_tokens: int = 5,
) -> MagicMock:
  usage = MagicMock()
  usage.prompt_tokens = prompt_tokens
  usage.completion_tokens = completion_tokens

  choice = MagicMock()
  choice.message.content = content
  choice.message.tool_calls = None
  choice.finish_reason = finish_reason

  resp = MagicMock()
  resp.id = id
  resp.choices = [choice]
  resp.usage = usage
  return resp


async def _async_gen(*chunks):
  for chunk in chunks:
    yield chunk


def _make_stream_chunk(
  content: str | None = None,
  prompt_tokens: int | None = None,
  completion_tokens: int | None = None,
) -> MagicMock:
  delta = MagicMock()
  delta.content = content
  delta.tool_calls = None

  choice = MagicMock()
  choice.delta = delta

  chunk = MagicMock()
  chunk.choices = [choice]

  if prompt_tokens is not None:
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    chunk.usage = usage
  else:
    chunk.usage = None

  return chunk


# ---------------------------------------------------------------------------
# 1. Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
  def test_isinstance_llm_callable(self):
    instance = LiteLLMCallable("test-model")
    assert isinstance(instance, LLMCallable)


# ---------------------------------------------------------------------------
# 2. complete — basic call
# ---------------------------------------------------------------------------


class TestCompleteBasic:
  @pytest.mark.asyncio
  async def test_calls_litellm_with_correct_model(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("claude-sonnet-4-5")
      await client.complete(_make_request())
      mock_ac.assert_called_once()
      assert mock_ac.call_args.kwargs["model"] == "claude-sonnet-4-5"

  @pytest.mark.asyncio
  async def test_returns_completion_response(self):
    resp = _mock_litellm_response(id="resp-42", content="Answer", finish_reason="stop")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("claude-sonnet-4-5")
      result = await client.complete(_make_request())

    assert isinstance(result, CompletionResponse)
    assert result.id == "resp-42"
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "Answer"
    assert result.stop_reason == "stop"

  @pytest.mark.asyncio
  async def test_messages_passed_to_litellm(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      request = _make_request(messages=[_make_message("user", "Hello")])
      await client.complete(request)
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert len(messages_arg) == 1
      assert messages_arg[0]["role"] == "user"
      assert messages_arg[0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# 3. complete — system prompt passed as kwarg
# ---------------------------------------------------------------------------


class TestCompleteSystemPrompt:
  @pytest.mark.asyncio
  async def test_system_prompt_passed_as_kwarg(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      request = _make_request(system=[TextBlock(text="You are a helpful assistant.")])
      await client.complete(request)
      assert mock_ac.call_args.kwargs.get("system") == "You are a helpful assistant."

  @pytest.mark.asyncio
  async def test_multiple_system_blocks_joined(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      request = _make_request(
        system=[TextBlock(text="Block one. "), TextBlock(text="Block two.")]
      )
      await client.complete(request)
      assert mock_ac.call_args.kwargs.get("system") == "Block one. Block two."


# ---------------------------------------------------------------------------
# 4. complete — no system prompt
# ---------------------------------------------------------------------------


class TestCompleteNoSystemPrompt:
  @pytest.mark.asyncio
  async def test_no_system_kwarg_when_none(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(system=None))
      assert "system" not in mock_ac.call_args.kwargs


# ---------------------------------------------------------------------------
# 5. complete — message content joining
# ---------------------------------------------------------------------------


class TestCompleteMessageContentJoining:
  @pytest.mark.asyncio
  async def test_multiple_text_blocks_joined_into_one_string(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      request = _make_request(
        messages=[_make_message("user", "Part one. ", "Part two.")]
      )
      await client.complete(request)
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert messages_arg[0]["content"] == "Part one. Part two."

  @pytest.mark.asyncio
  async def test_multiple_messages_each_joined_independently(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      request = _make_request(
        messages=[
          _make_message("user", "Hello"),
          _make_message("assistant", "Hi", " there"),
        ]
      )
      await client.complete(request)
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert messages_arg[0]["content"] == "Hello"
      assert messages_arg[1]["content"] == "Hi there"


# ---------------------------------------------------------------------------
# 6. complete — usage populated
# ---------------------------------------------------------------------------


class TestCompleteUsage:
  @pytest.mark.asyncio
  async def test_input_tokens_from_litellm_response(self):
    resp = _mock_litellm_response(prompt_tokens=42, completion_tokens=17)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())
      assert result.usage.input_tokens == 42

  @pytest.mark.asyncio
  async def test_output_tokens_from_litellm_response(self):
    resp = _mock_litellm_response(prompt_tokens=42, completion_tokens=17)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())
      assert result.usage.output_tokens == 17

  @pytest.mark.asyncio
  async def test_usage_is_token_usage_instance(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())
      assert isinstance(result.usage, TokenUsage)


# ---------------------------------------------------------------------------
# 7. complete — kwargs forwarded
# ---------------------------------------------------------------------------


class TestCompleteKwargsForwarded:
  @pytest.mark.asyncio
  async def test_init_kwargs_forwarded_to_litellm(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model", temperature=0.7, max_tokens=256)
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs.get("temperature") == 0.7
      assert mock_ac.call_args.kwargs.get("max_tokens") == 256

  @pytest.mark.asyncio
  async def test_base_url_forwarded_to_litellm(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model", base_url="http://localhost:4000")
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs.get("base_url") == "http://localhost:4000"

  @pytest.mark.asyncio
  async def test_no_base_url_kwarg_when_none(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("gpt-4o")  # no base_url
      await client.complete(_make_request())
      assert "base_url" not in mock_ac.call_args.kwargs

  @pytest.mark.asyncio
  async def test_bare_model_prefixed_with_openai_when_base_url_given(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("my-model", base_url="http://localhost:4000")
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs["model"] == "openai/my-model"

  @pytest.mark.asyncio
  async def test_model_with_slash_not_prefixed_when_base_url_given(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("openai/gpt-4o", base_url="http://localhost:4000")
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs["model"] == "openai/gpt-4o"

  @pytest.mark.asyncio
  async def test_model_not_prefixed_without_base_url(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("my-model")
      await client.complete(_make_request())
      assert mock_ac.call_args.kwargs["model"] == "my-model"


# ---------------------------------------------------------------------------
# 8. stream — yields text events
# ---------------------------------------------------------------------------


class TestStreamTextEvents:
  @pytest.mark.asyncio
  async def test_yields_text_event_per_content_chunk(self):
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=" world"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 2
    assert text_events[0].text == "Hello"
    assert text_events[1].text == " world"

  @pytest.mark.asyncio
  async def test_stream_event_type_is_text(self):
    chunks = [_make_stream_chunk(content="Hi")]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert all(isinstance(e, StreamEvent) for e in text_events)

  @pytest.mark.asyncio
  async def test_stream_true_passed_to_litellm(self):
    chunk = _make_stream_chunk(content="Hello")
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
      mock_ac.return_value = _async_gen(chunk)
      client = LiteLLMCallable("gpt-4o")
      async for _ in client.stream(_make_request()):
        pass
    assert mock_ac.call_args.kwargs.get("stream") is True


# ---------------------------------------------------------------------------
# 9. stream — yields usage event
# ---------------------------------------------------------------------------


class TestStreamUsageEvent:
  @pytest.mark.asyncio
  async def test_usage_event_yielded_when_chunk_has_usage(self):
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=None, prompt_tokens=8, completion_tokens=3),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1
    assert usage_events[0].usage.input_tokens == 8
    assert usage_events[0].usage.output_tokens == 3

  @pytest.mark.asyncio
  async def test_no_usage_event_when_no_chunk_has_usage(self):
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=" there"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 0


# ---------------------------------------------------------------------------
# 10. stream — yields end event
# ---------------------------------------------------------------------------


class TestStreamEndEvent:
  @pytest.mark.asyncio
  async def test_end_event_always_last(self):
    chunks = [_make_stream_chunk(content="Hi")]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    assert events[-1].type == "end"

  @pytest.mark.asyncio
  async def test_end_event_present_even_with_no_content_chunks(self):
    async def fake_acompletion(*args, **kwargs):
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    assert len(events) == 1
    assert events[0].type == "end"


# ---------------------------------------------------------------------------
# 11. stream — system prompt passed as kwarg
# ---------------------------------------------------------------------------


class TestStreamSystemPrompt:
  @pytest.mark.asyncio
  async def test_system_prompt_passed_as_kwarg_in_stream(self):
    captured: dict = {}

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      request = _make_request(system=[TextBlock(text="Be concise.")])
      _ = [e async for e in client.stream(request)]

    assert captured.get("system") == "Be concise."

  @pytest.mark.asyncio
  async def test_no_system_kwarg_in_stream_when_none(self):
    captured: dict = {}

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      _ = [e async for e in client.stream(_make_request(system=None))]

    assert "system" not in captured


# ---------------------------------------------------------------------------
# 12. stream — skips empty/None content chunks
# ---------------------------------------------------------------------------


class TestStreamSkipsEmptyChunks:
  @pytest.mark.asyncio
  async def test_none_content_chunk_does_not_yield_text_event(self):
    chunks = [
      _make_stream_chunk(content=None),
      _make_stream_chunk(content="Real content"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 1
    assert text_events[0].text == "Real content"

  @pytest.mark.asyncio
  async def test_empty_string_content_chunk_does_not_yield_text_event(self):
    chunks = [
      _make_stream_chunk(content=""),
      _make_stream_chunk(content="Valid"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    text_events = [e for e in events if e.type == "text"]
    assert len(text_events) == 1
    assert text_events[0].text == "Valid"


# ---------------------------------------------------------------------------
# 13. complete — tools passed to litellm
# ---------------------------------------------------------------------------


class TestCompleteTools:
  @pytest.mark.asyncio
  async def test_no_tools_kwarg_when_request_tools_is_none(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(tools=None))
      assert "tools" not in mock_ac.call_args.kwargs

  @pytest.mark.asyncio
  async def test_single_tool_passed_in_openai_function_format(self):
    resp = _mock_litellm_response()
    tool = _make_tool(
      name="get_weather",
      description="Get the weather for a location",
      input_schema={"type": "object", "properties": {"location": {"type": "string"}}},
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(tools=[tool]))
      tools_arg = mock_ac.call_args.kwargs.get("tools")
      assert tools_arg is not None
      assert len(tools_arg) == 1
      assert tools_arg[0] == {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the weather for a location",
          "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
      }

  @pytest.mark.asyncio
  async def test_multiple_tools_all_passed(self):
    resp = _mock_litellm_response()
    tools = [
      _make_tool(name="get_weather"),
      _make_tool(name="search_web", description="Search the web"),
    ]
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(tools=tools))
      tools_arg = mock_ac.call_args.kwargs.get("tools")
      assert tools_arg is not None
      assert len(tools_arg) == 2
      names = [t["function"]["name"] for t in tools_arg]
      assert "get_weather" in names
      assert "search_web" in names

  @pytest.mark.asyncio
  async def test_tool_type_is_function(self):
    resp = _mock_litellm_response()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(tools=[_make_tool()]))
      tools_arg = mock_ac.call_args.kwargs["tools"]
      assert all(t["type"] == "function" for t in tools_arg)


# ---------------------------------------------------------------------------
# 14. stream — tools passed to litellm
# ---------------------------------------------------------------------------


class TestStreamTools:
  @pytest.mark.asyncio
  async def test_no_tools_kwarg_when_request_tools_is_none(self):
    captured: dict = {}

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      _ = [e async for e in client.stream(_make_request(tools=None))]

    assert "tools" not in captured

  @pytest.mark.asyncio
  async def test_single_tool_passed_in_openai_function_format(self):
    captured: dict = {}
    tool = _make_tool(
      name="search_web",
      description="Search the web",
      input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      _ = [e async for e in client.stream(_make_request(tools=[tool]))]

    tools_arg = captured.get("tools")
    assert tools_arg is not None
    assert len(tools_arg) == 1
    assert tools_arg[0] == {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
      },
    }

  @pytest.mark.asyncio
  async def test_multiple_tools_all_passed(self):
    captured: dict = {}
    tools = [_make_tool(name="tool_a"), _make_tool(name="tool_b")]

    async def fake_acompletion(*args, **kwargs):
      captured.update(kwargs)
      return _async_gen()

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      _ = [e async for e in client.stream(_make_request(tools=tools))]

    tools_arg = captured.get("tools")
    assert tools_arg is not None
    assert len(tools_arg) == 2


# ---------------------------------------------------------------------------
# 15. _build_messages — tool use blocks in assistant messages
# ---------------------------------------------------------------------------


class TestBuildMessagesToolUse:
  @pytest.mark.asyncio
  async def test_tool_use_block_emits_tool_calls_on_assistant_message(self):
    resp = _mock_litellm_response()
    msg = _make_tool_use_message(tool_id="tu_abc", tool_name="get_weather", tool_input={"location": "Oslo"})
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert len(messages_arg) == 1
      out = messages_arg[0]
      assert out["role"] == "assistant"
      assert "tool_calls" in out
      assert len(out["tool_calls"]) == 1
      tc = out["tool_calls"][0]
      assert tc["id"] == "tu_abc"
      assert tc["type"] == "function"
      assert tc["function"]["name"] == "get_weather"
      assert tc["function"]["arguments"] == json.dumps({"location": "Oslo"})
      assert isinstance(tc["function"]["arguments"], str)

  @pytest.mark.asyncio
  async def test_tool_use_block_message_has_no_plain_content_key(self):
    resp = _mock_litellm_response()
    msg = _make_tool_use_message()
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      out = messages_arg[0]
      # content should be absent or None/empty — not a joined text string from tool blocks
      assert out.get("content") in (None, "", [])

  @pytest.mark.asyncio
  async def test_mixed_text_and_tool_use_in_assistant_message(self):
    resp = _mock_litellm_response()
    msg = Message(
      role="assistant",
      content=[
        TextBlock(text="I'll check the weather."),
        ToolUseBlock(id="tu_1", name="get_weather", input={"location": "Berlin"}),
      ],
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      out = messages_arg[0]
      assert out["role"] == "assistant"
      assert "I'll check the weather." in (out.get("content") or "")
      assert "tool_calls" in out
      assert out["tool_calls"][0]["function"]["name"] == "get_weather"

  @pytest.mark.asyncio
  async def test_multiple_tool_use_blocks_all_emitted(self):
    resp = _mock_litellm_response()
    msg = Message(
      role="assistant",
      content=[
        ToolUseBlock(id="tu_1", name="tool_a", input={"x": 1}),
        ToolUseBlock(id="tu_2", name="tool_b", input={"y": 2}),
      ],
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      out = messages_arg[0]
      assert len(out["tool_calls"]) == 2
      ids = [tc["id"] for tc in out["tool_calls"]]
      assert "tu_1" in ids
      assert "tu_2" in ids


# ---------------------------------------------------------------------------
# 16. _build_messages — tool result blocks → tool role messages
# ---------------------------------------------------------------------------


class TestBuildMessagesToolResult:
  @pytest.mark.asyncio
  async def test_tool_result_block_emits_tool_role_message(self):
    resp = _mock_litellm_response()
    msg = _make_tool_result_message(tool_use_id="tu_abc", content="Sunny, 22°C")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      assert any(m["role"] == "tool" for m in messages_arg)

  @pytest.mark.asyncio
  async def test_tool_result_message_has_correct_tool_call_id(self):
    resp = _mock_litellm_response()
    msg = _make_tool_result_message(tool_use_id="tu_xyz")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      tool_msgs = [m for m in messages_arg if m["role"] == "tool"]
      assert len(tool_msgs) == 1
      assert tool_msgs[0]["tool_call_id"] == "tu_xyz"

  @pytest.mark.asyncio
  async def test_multiple_tool_results_in_one_user_message_emit_multiple_tool_messages(self):
    resp = _mock_litellm_response()
    msg = Message(
      role="user",
      content=[
        ToolResultBlock(tool_use_id="tu_1", content="Result A"),
        ToolResultBlock(tool_use_id="tu_2", content="Result B"),
      ],
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      tool_msgs = [m for m in messages_arg if m["role"] == "tool"]
      assert len(tool_msgs) == 2
      ids = {m["tool_call_id"] for m in tool_msgs}
      assert ids == {"tu_1", "tu_2"}

  @pytest.mark.asyncio
  async def test_tool_result_messages_not_emitted_as_user_role(self):
    resp = _mock_litellm_response()
    msg = _make_tool_result_message(tool_use_id="tu_1")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      # The original user message with ToolResultBlock must NOT appear as a user role message
      user_msgs = [m for m in messages_arg if m["role"] == "user"]
      assert len(user_msgs) == 0


# ---------------------------------------------------------------------------
# 17. _build_messages — ToolResultBlock.content as str vs list[TextBlock]
# ---------------------------------------------------------------------------


class TestBuildMessagesToolResultContent:
  @pytest.mark.asyncio
  async def test_tool_result_string_content_passed_as_is(self):
    resp = _mock_litellm_response()
    msg = _make_tool_result_message(content="Plain string result")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      tool_msgs = [m for m in messages_arg if m["role"] == "tool"]
      assert tool_msgs[0]["content"] == "Plain string result"

  @pytest.mark.asyncio
  async def test_tool_result_list_content_joined_to_string(self):
    resp = _mock_litellm_response()
    msg = _make_tool_result_message(
      content=[TextBlock(text="Part one. "), TextBlock(text="Part two.")]
    )
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      tool_msgs = [m for m in messages_arg if m["role"] == "tool"]
      assert tool_msgs[0]["content"] == "Part one. Part two."

  @pytest.mark.asyncio
  async def test_tool_result_single_text_block_list_content(self):
    resp = _mock_litellm_response()
    msg = _make_tool_result_message(content=[TextBlock(text="Only block.")])
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp) as mock_ac:
      client = LiteLLMCallable("model")
      await client.complete(_make_request(messages=[msg]))
      messages_arg = mock_ac.call_args.kwargs["messages"]
      tool_msgs = [m for m in messages_arg if m["role"] == "tool"]
      assert tool_msgs[0]["content"] == "Only block."


# ---------------------------------------------------------------------------
# Helpers for tool call response mocking
# ---------------------------------------------------------------------------


def _make_tool_call_mock(
  id: str = "tc_1",
  name: str = "get_weather",
  arguments: str = '{"location": "Oslo"}',
) -> MagicMock:
  """Build a single litellm tool call object as it appears on choice.message.tool_calls."""
  tc = MagicMock()
  tc.id = id
  tc.function = MagicMock()
  tc.function.name = name
  tc.function.arguments = arguments
  return tc


def _mock_litellm_response_with_tool_calls(
  tool_calls: list[MagicMock],
  id: str = "resp-tc-1",
  content: str | None = None,
  prompt_tokens: int = 10,
  completion_tokens: int = 5,
) -> MagicMock:
  """Build a litellm response whose choice.message has tool_calls set.

  Pass content=None for a tool-call-only response; pass a non-empty string
  for a mixed text + tool call response.
  """
  usage = MagicMock()
  usage.prompt_tokens = prompt_tokens
  usage.completion_tokens = completion_tokens

  choice = MagicMock()
  choice.message.content = content
  choice.message.tool_calls = tool_calls
  choice.finish_reason = "tool_calls"

  resp = MagicMock()
  resp.id = id
  resp.choices = [choice]
  resp.usage = usage
  return resp


def _make_tool_call_chunk(
  index: int,
  id: str | None,
  name: str | None,
  arguments_fragment: str,
) -> MagicMock:
  """Build a streaming delta chunk that carries a partial tool call.

  Mirrors the litellm streaming delta shape:
    chunk.choices[0].delta.tool_calls = [MagicMock(index=..., id=..., function=...)]
  """
  tc_delta = MagicMock()
  tc_delta.index = index
  tc_delta.id = id
  tc_delta.function = MagicMock()
  tc_delta.function.name = name
  tc_delta.function.arguments = arguments_fragment

  delta = MagicMock()
  delta.content = None
  delta.tool_calls = [tc_delta]

  choice = MagicMock()
  choice.delta = delta

  chunk = MagicMock()
  chunk.choices = [choice]
  chunk.usage = None
  return chunk


# ---------------------------------------------------------------------------
# 18. complete — capture tool calls from model response
# ---------------------------------------------------------------------------


class TestCompleteToolCallResponse:
  @pytest.mark.asyncio
  async def test_single_tool_call_produces_one_tool_use_block(self):
    tc = _make_tool_call_mock(id="tc_1", name="get_weather", arguments='{"location": "Oslo"}')
    resp = _mock_litellm_response_with_tool_calls([tc])
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 1

  @pytest.mark.asyncio
  async def test_tool_use_block_id_matches_tool_call_id(self):
    tc = _make_tool_call_mock(id="tc_abc", name="get_weather", arguments='{"location": "Oslo"}')
    resp = _mock_litellm_response_with_tool_calls([tc])
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_block = next(b for b in result.content if isinstance(b, ToolUseBlock))
    assert tool_block.id == "tc_abc"

  @pytest.mark.asyncio
  async def test_tool_use_block_name_matches_function_name(self):
    tc = _make_tool_call_mock(id="tc_1", name="search_web", arguments='{"query": "Oslo weather"}')
    resp = _mock_litellm_response_with_tool_calls([tc])
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_block = next(b for b in result.content if isinstance(b, ToolUseBlock))
    assert tool_block.name == "search_web"

  @pytest.mark.asyncio
  async def test_tool_use_block_input_is_parsed_from_json(self):
    tc = _make_tool_call_mock(id="tc_1", name="get_weather", arguments='{"location": "Oslo", "units": "celsius"}')
    resp = _mock_litellm_response_with_tool_calls([tc])
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_block = next(b for b in result.content if isinstance(b, ToolUseBlock))
    assert tool_block.input == {"location": "Oslo", "units": "celsius"}

  @pytest.mark.asyncio
  async def test_stop_reason_is_tool_use_when_tool_calls_present(self):
    tc = _make_tool_call_mock()
    resp = _mock_litellm_response_with_tool_calls([tc])
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    assert result.stop_reason == "tool_use"

  @pytest.mark.asyncio
  async def test_no_text_block_when_only_tool_calls_present(self):
    tc = _make_tool_call_mock()
    resp = _mock_litellm_response_with_tool_calls([tc], content=None)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    text_blocks = [b for b in result.content if isinstance(b, TextBlock)]
    assert len(text_blocks) == 0

  @pytest.mark.asyncio
  async def test_multiple_tool_calls_produce_multiple_tool_use_blocks(self):
    tcs = [
      _make_tool_call_mock(id="tc_1", name="tool_a", arguments='{"x": 1}'),
      _make_tool_call_mock(id="tc_2", name="tool_b", arguments='{"y": 2}'),
    ]
    resp = _mock_litellm_response_with_tool_calls(tcs)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 2
    ids = {b.id for b in tool_blocks}
    assert ids == {"tc_1", "tc_2"}

  @pytest.mark.asyncio
  async def test_multiple_tool_calls_names_and_inputs_correct(self):
    tcs = [
      _make_tool_call_mock(id="tc_1", name="tool_a", arguments='{"x": 1}'),
      _make_tool_call_mock(id="tc_2", name="tool_b", arguments='{"y": 2}'),
    ]
    resp = _mock_litellm_response_with_tool_calls(tcs)
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_blocks = {b.id: b for b in result.content if isinstance(b, ToolUseBlock)}
    assert tool_blocks["tc_1"].name == "tool_a"
    assert tool_blocks["tc_1"].input == {"x": 1}
    assert tool_blocks["tc_2"].name == "tool_b"
    assert tool_blocks["tc_2"].input == {"y": 2}

  @pytest.mark.asyncio
  async def test_no_tool_calls_does_not_produce_tool_use_blocks(self):
    """Regression: plain text response must not produce ToolUseBlocks."""
    resp = _mock_litellm_response(content="Hello", finish_reason="stop")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
    assert len(tool_blocks) == 0
    assert result.stop_reason == "stop"


# ---------------------------------------------------------------------------
# 19. complete — mixed text + tool call response
# ---------------------------------------------------------------------------


class TestCompleteTextAndToolCallResponse:
  @pytest.mark.asyncio
  async def test_both_text_block_and_tool_use_block_present(self):
    tc = _make_tool_call_mock(id="tc_1", name="get_weather", arguments='{"location": "Oslo"}')
    resp = _mock_litellm_response_with_tool_calls([tc], content="I'll check the weather.")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    text_blocks = [b for b in result.content if isinstance(b, TextBlock)]
    tool_blocks = [b for b in result.content if isinstance(b, ToolUseBlock)]
    assert len(text_blocks) == 1
    assert len(tool_blocks) == 1

  @pytest.mark.asyncio
  async def test_text_block_content_correct_in_mixed_response(self):
    tc = _make_tool_call_mock()
    resp = _mock_litellm_response_with_tool_calls([tc], content="Let me look that up.")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    text_block = next(b for b in result.content if isinstance(b, TextBlock))
    assert text_block.text == "Let me look that up."

  @pytest.mark.asyncio
  async def test_stop_reason_is_tool_use_in_mixed_response(self):
    tc = _make_tool_call_mock()
    resp = _mock_litellm_response_with_tool_calls([tc], content="Checking now.")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=resp):
      client = LiteLLMCallable("model")
      result = await client.complete(_make_request())

    assert result.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# 20. stream — tool_use StreamEvents from streaming deltas
# ---------------------------------------------------------------------------


class TestStreamToolCallEvents:
  @pytest.mark.asyncio
  async def test_single_tool_call_in_one_chunk_yields_tool_use_event(self):
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_1", name="get_weather", arguments_fragment='{"location": "Oslo"}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_events = [e for e in events if e.type == "tool_use"]
    assert len(tool_events) == 1

  @pytest.mark.asyncio
  async def test_tool_use_event_has_correct_id(self):
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_xyz", name="get_weather", arguments_fragment='{"location": "Oslo"}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_event = next(e for e in events if e.type == "tool_use")
    assert tool_event.tool_use_id == "tc_xyz"

  @pytest.mark.asyncio
  async def test_tool_use_event_has_correct_name(self):
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_1", name="search_web", arguments_fragment='{"query": "Oslo"}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_event = next(e for e in events if e.type == "tool_use")
    assert tool_event.tool_name == "search_web"

  @pytest.mark.asyncio
  async def test_tool_use_event_input_parsed_from_json(self):
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_1", name="get_weather", arguments_fragment='{"location": "Oslo", "units": "celsius"}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_event = next(e for e in events if e.type == "tool_use")
    assert tool_event.tool_input == {"location": "Oslo", "units": "celsius"}

  @pytest.mark.asyncio
  async def test_arguments_accumulated_across_chunks(self):
    """Tool call arguments split across multiple delta chunks must be reassembled."""
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_1", name="get_weather", arguments_fragment='{"loc'),
      _make_tool_call_chunk(index=0, id=None, name=None, arguments_fragment='ation": "Oslo"}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_event = next(e for e in events if e.type == "tool_use")
    assert tool_event.tool_input == {"location": "Oslo"}

  @pytest.mark.asyncio
  async def test_tool_use_events_precede_end_event(self):
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_1", name="get_weather", arguments_fragment='{"location": "Oslo"}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    assert events[-1].type == "end"
    tool_indices = [i for i, e in enumerate(events) if e.type == "tool_use"]
    end_index = next(i for i, e in enumerate(events) if e.type == "end")
    assert all(i < end_index for i in tool_indices)

  @pytest.mark.asyncio
  async def test_two_concurrent_tool_calls_emit_two_tool_use_events(self):
    """Two tool calls with different indices must each produce a tool_use event."""
    chunks = [
      _make_tool_call_chunk(index=0, id="tc_1", name="tool_a", arguments_fragment='{"x": 1}'),
      _make_tool_call_chunk(index=1, id="tc_2", name="tool_b", arguments_fragment='{"y": 2}'),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_events = [e for e in events if e.type == "tool_use"]
    assert len(tool_events) == 2
    ids = {e.tool_use_id for e in tool_events}
    assert ids == {"tc_1", "tc_2"}
    inputs = {e.tool_use_id: e.tool_input for e in tool_events}
    assert inputs["tc_1"] == {"x": 1}
    assert inputs["tc_2"] == {"y": 2}

  @pytest.mark.asyncio
  async def test_no_tool_use_events_when_no_tool_calls_in_stream(self):
    """Regression: text-only stream must not emit tool_use events."""
    chunks = [
      _make_stream_chunk(content="Hello"),
      _make_stream_chunk(content=" world"),
    ]

    async def fake_acompletion(*args, **kwargs):
      return _async_gen(*chunks)

    with patch("litellm.acompletion", new=fake_acompletion):
      client = LiteLLMCallable("model")
      events = [e async for e in client.stream(_make_request())]

    tool_events = [e for e in events if e.type == "tool_use"]
    assert len(tool_events) == 0
