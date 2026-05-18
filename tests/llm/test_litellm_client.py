"""Tests for sr2.llm.litellm_client.LiteLLMClient.

Covers:
  1. Construction — model string, kwargs (api_key, base_url), protocol compliance
  2. complete() — LiteLLM call args, response translation
  3. stream() — streaming call args, event sequence translation
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.models import TextBlock, TokenUsage
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMCallable,
    StreamEvent,
)

# ---------------------------------------------------------------------------
# Import under test — expected to fail with ModuleNotFoundError until
# implementation exists.  All tests will show as errors until then.
# ---------------------------------------------------------------------------

from sr2.llm.litellm_client import LiteLLMClient  # noqa: E402


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def make_litellm_response(
    text: str = "Hello",
    finish_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 5,
    response_id: str = "chatcmpl-test-001",
) -> MagicMock:
    """Mimic a non-streaming litellm response object."""
    resp = MagicMock()
    resp.id = response_id
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.choices[0].finish_reason = finish_reason
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = input_tokens
    resp.usage.completion_tokens = output_tokens
    return resp


def make_stream_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    """Mimic a single litellm streaming chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = usage
    return chunk


def make_usage_object(prompt_tokens: int = 10, completion_tokens: int = 5) -> MagicMock:
    """Mimic a litellm usage object attached to a streaming chunk."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    return usage


async def make_async_stream(chunks: list) -> AsyncIterator:
    """Async generator that yields the given chunks, mimicking litellm streaming."""
    for chunk in chunks:
        yield chunk


def make_minimal_request(
    system_text: str | None = "You are helpful.",
    user_text: str = "Hello",
) -> CompletionRequest:
    system = [TextBlock(text=system_text)] if system_text is not None else None
    from sr2.models import Message

    return CompletionRequest(
        system=system,
        messages=[
            Message(role="user", content=[TextBlock(text=user_text)]),
        ],
    )


# ---------------------------------------------------------------------------
# 1. Construction tests
# ---------------------------------------------------------------------------


class TestLiteLLMClientConstruction:
    def test_constructs_with_model_string(self):
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        assert client is not None

    def test_stores_model(self):
        client = LiteLLMClient(model="openai/gpt-4o")
        assert client.model == "openai/gpt-4o"

    def test_constructs_with_api_key_kwarg(self):
        """api_key should be accepted without error."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6", api_key="sk-test")
        assert client is not None

    def test_constructs_with_base_url_kwarg(self):
        """base_url should be accepted without error."""
        client = LiteLLMClient(
            model="openai/local-model",
            base_url="http://localhost:8080/v1",
        )
        assert client is not None

    def test_constructs_with_multiple_kwargs(self):
        """Multiple extra kwargs should all be accepted."""
        client = LiteLLMClient(
            model="anthropic/claude-sonnet-4-6",
            api_key="sk-test",
            base_url="https://example.com",
        )
        assert client is not None

    def test_satisfies_llmcallable_protocol(self):
        """LiteLLMClient must satisfy the LLMCallable runtime-checkable protocol."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        assert isinstance(client, LLMCallable)


# ---------------------------------------------------------------------------
# 2. complete() tests
# ---------------------------------------------------------------------------


class TestComplete:
    @pytest.mark.asyncio
    async def test_calls_litellm_with_correct_model(self):
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        mock_call.assert_awaited_once()
        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_passes_system_prompt_as_string(self):
        """system list[TextBlock] must be joined into a single string."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(
            system=[TextBlock(text="You are helpful.")],
            messages=[],
        )
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_joins_multiple_system_blocks(self):
        """Multiple system TextBlocks must be joined (concatenated)."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(
            system=[
                TextBlock(text="Block one."),
                TextBlock(text=" Block two."),
            ],
            messages=[],
        )
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        assert "Block one." in call_kwargs["system"]
        assert "Block two." in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_no_system_key_when_system_is_none(self):
        """When request.system is None, system must not be passed to litellm
        (or passed as None/empty — but it must not raise)."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(system=None, messages=[])
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)  # must not raise

        mock_call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_messages_correctly(self):
        """Messages must be translated to list[dict] with role and content (text)."""
        from sr2.models import Message

        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(
            system=None,
            messages=[
                Message(role="user", content=[TextBlock(text="Hi")]),
                Message(role="assistant", content=[TextBlock(text="Hello!")]),
            ],
        )
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hi"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_message_with_multiple_content_blocks(self):
        """Multiple TextBlocks in a message should be joined into one string."""
        from sr2.models import Message

        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(
            system=None,
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="Part A."),
                        TextBlock(text=" Part B."),
                    ],
                )
            ],
        )
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        content = call_kwargs["messages"][0]["content"]
        assert "Part A." in content
        assert "Part B." in content

    @pytest.mark.asyncio
    async def test_does_not_pass_stream_true(self):
        """complete() must NOT pass stream=True to litellm."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        # stream must be absent or False — never True
        assert call_kwargs.get("stream", False) is not True

    @pytest.mark.asyncio
    async def test_returns_completion_response_type(self):
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.complete(request)

        assert isinstance(result, CompletionResponse)

    @pytest.mark.asyncio
    async def test_response_id_mapped_correctly(self):
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response(response_id="chatcmpl-xyz-999")

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.complete(request)

        assert result.id == "chatcmpl-xyz-999"

    @pytest.mark.asyncio
    async def test_response_content_is_text_block(self):
        """Response content must be a list containing a TextBlock with the response text."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response(text="The answer is 42.")

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.complete(request)

        assert len(result.content) == 1
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == "The answer is 42."

    @pytest.mark.asyncio
    async def test_response_stop_reason_mapped(self):
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response(finish_reason="stop")

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.complete(request)

        assert result.stop_reason == "stop"

    @pytest.mark.asyncio
    async def test_response_token_usage_mapped(self):
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        mock_resp = make_litellm_response(input_tokens=42, output_tokens=17)

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.complete(request)

        assert result.usage.input_tokens == 42
        assert result.usage.output_tokens == 17

    @pytest.mark.asyncio
    async def test_passes_api_key_from_init(self):
        """api_key provided at construction must be forwarded to litellm."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6", api_key="sk-my-key")
        request = make_minimal_request()
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("api_key") == "sk-my-key"

    @pytest.mark.asyncio
    async def test_passes_base_url_from_init(self):
        """base_url provided at construction must be forwarded to litellm."""
        client = LiteLLMClient(
            model="openai/local",
            base_url="http://localhost:1234/v1",
        )
        request = make_minimal_request()
        mock_resp = make_litellm_response()

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ) as mock_call:
            await client.complete(request)

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("base_url") == "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# 3. stream() tests
# ---------------------------------------------------------------------------


class TestStream:
    @pytest.mark.asyncio
    async def test_calls_litellm_with_stream_true(self):
        """stream() must pass stream=True to litellm.acompletion."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(content="Hi"),
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ) as mock_call:
            events = [event async for event in client.stream(request)]

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs.get("stream") is True

    @pytest.mark.asyncio
    async def test_yields_text_event_for_content_chunk(self):
        """A chunk with delta.content must produce a StreamEvent(type='text')."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(content="Hello "),
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello "

    @pytest.mark.asyncio
    async def test_each_chunk_text_preserved_individually(self):
        """Each content chunk must produce its own StreamEvent — texts are NOT concatenated."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(content="Hello "),
            make_stream_chunk(content="world"),
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello "
        assert text_events[1].text == "world"

    @pytest.mark.asyncio
    async def test_yields_usage_event_for_usage_chunk(self):
        """A chunk with usage must produce a StreamEvent(type='usage')."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        usage_obj = make_usage_object(prompt_tokens=20, completion_tokens=8)
        chunks = [
            make_stream_chunk(content="Hi"),
            make_stream_chunk(usage=usage_obj, finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        usage_events = [e for e in events if e.type == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0].usage is not None
        assert usage_events[0].usage.input_tokens == 20
        assert usage_events[0].usage.output_tokens == 8

    @pytest.mark.asyncio
    async def test_final_event_is_end(self):
        """The last event in a stream must always be StreamEvent(type='end')."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(content="Hi"),
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_skips_chunk_with_none_content(self):
        """Chunks where delta.content is None must not produce a text event."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(content=None),   # no content — skip
            make_stream_chunk(content="real"),
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 1
        assert text_events[0].text == "real"

    @pytest.mark.asyncio
    async def test_skips_chunk_with_empty_string_content(self):
        """Chunks where delta.content is an empty string must not produce a text event."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(content=""),     # empty string — skip
            make_stream_chunk(content="word"),
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 1
        assert text_events[0].text == "word"

    @pytest.mark.asyncio
    async def test_stream_with_no_content_chunks_still_ends(self):
        """Even an empty stream (no text) must produce a final end event."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [
            make_stream_chunk(finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        assert any(e.type == "end" for e in events)
        assert events[-1].type == "end"

    @pytest.mark.asyncio
    async def test_stream_passes_model_to_litellm(self):
        """stream() must pass the correct model string to litellm."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        chunks = [make_stream_chunk(finish_reason="stop")]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ) as mock_call:
            events = [event async for event in client.stream(request)]

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_stream_passes_system_prompt(self):
        """stream() must forward the system prompt just like complete()."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(
            system=[TextBlock(text="Be concise.")],
            messages=[],
        )
        chunks = [make_stream_chunk(finish_reason="stop")]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ) as mock_call:
            events = [event async for event in client.stream(request)]

        call_kwargs = mock_call.call_args.kwargs
        assert call_kwargs["system"] == "Be concise."

    @pytest.mark.asyncio
    async def test_stream_passes_messages(self):
        """stream() must translate messages the same way complete() does."""
        from sr2.models import Message

        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = CompletionRequest(
            system=None,
            messages=[
                Message(role="user", content=[TextBlock(text="Ping")]),
            ],
        )
        chunks = [make_stream_chunk(finish_reason="stop")]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ) as mock_call:
            events = [event async for event in client.stream(request)]

        call_kwargs = mock_call.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Ping"

    @pytest.mark.asyncio
    async def test_stream_yields_stream_event_objects(self):
        """All yielded values must be StreamEvent instances."""
        client = LiteLLMClient(model="anthropic/claude-sonnet-4-6")
        request = make_minimal_request()
        usage_obj = make_usage_object(prompt_tokens=5, completion_tokens=3)
        chunks = [
            make_stream_chunk(content="Hi"),
            make_stream_chunk(usage=usage_obj, finish_reason="stop"),
        ]

        with patch(
            "sr2.llm.litellm_client.litellm.acompletion",
            return_value=make_async_stream(chunks),
        ):
            events = [event async for event in client.stream(request)]

        for event in events:
            assert isinstance(event, StreamEvent)
