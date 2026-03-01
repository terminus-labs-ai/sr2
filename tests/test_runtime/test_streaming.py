"""Tests for streaming events, LLM client streaming, and loop streaming."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from runtime.config import LLMModelConfig, StreamContentConfig
from runtime.llm import (
    LLMClient,
    LLMResponse,
    LLMLoop,
    LoopResult,
    StreamEndEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolStartEvent,
)


# ---------------------------------------------------------------------------
# Event dataclass construction
# ---------------------------------------------------------------------------


class TestStreamEvents:
    def test_text_delta(self):
        e = TextDeltaEvent(content="hello")
        assert e.content == "hello"

    def test_tool_start(self):
        e = ToolStartEvent(tool_name="search", tool_call_id="tc_1", arguments={"q": "x"})
        assert e.tool_name == "search"
        assert e.tool_call_id == "tc_1"
        assert e.arguments == {"q": "x"}

    def test_tool_start_default_args(self):
        e = ToolStartEvent(tool_name="read", tool_call_id="tc_2")
        assert e.arguments == {}

    def test_tool_result(self):
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1", result="found", success=True)
        assert e.success is True

    def test_stream_end(self):
        e = StreamEndEvent(full_text="complete response")
        assert e.full_text == "complete response"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(**overrides):
    defaults = {
        "model": LLMModelConfig(name="test-model"),
        "fast_model": LLMModelConfig(name="test-fast", max_tokens=1000),
        "embedding": LLMModelConfig(name="test-embed"),
    }
    defaults.update(overrides)
    return LLMClient(**defaults)


def _make_stream_chunks(
    text_deltas: list[str] | None = None,
    tool_call_chunks: list[dict] | None = None,
    usage: dict | None = None,
):
    """Build a list of mock streaming chunks."""
    chunks = []

    for text in (text_deltas or []):
        delta = SimpleNamespace(content=text, tool_calls=None)
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=delta)],
            usage=None,
        )
        chunks.append(chunk)

    if tool_call_chunks:
        for tc in tool_call_chunks:
            fn = SimpleNamespace(
                name=tc.get("name"),
                arguments=tc.get("arguments", ""),
            )
            tc_delta = SimpleNamespace(
                index=tc.get("index", 0),
                id=tc.get("id"),
                function=fn,
            )
            delta = SimpleNamespace(content=None, tool_calls=[tc_delta])
            chunk = SimpleNamespace(
                choices=[SimpleNamespace(delta=delta)],
                usage=None,
            )
            chunks.append(chunk)

    # Final chunk with usage
    u = usage or {"prompt_tokens": 100, "completion_tokens": 50}
    usage_ns = SimpleNamespace(**u)
    final_delta = SimpleNamespace(content=None, tool_calls=None)
    final_chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=final_delta)],
        usage=usage_ns,
    )
    chunks.append(final_chunk)

    return chunks


async def _async_iter(items):
    """Turn a list into an async iterator."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# LLMClient.stream_complete() tests
# ---------------------------------------------------------------------------


class TestLLMClientStreamComplete:
    @pytest.mark.asyncio
    async def test_text_only_streaming(self):
        chunks = _make_stream_chunks(text_deltas=["Hello", " world", "!"])

        async def mock_acompletion(**kwargs):
            return _async_iter(chunks)

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            client = _make_client()
            collected = []
            async for delta in client.stream_complete(
                messages=[{"role": "user", "content": "hi"}]
            ):
                collected.append(delta)

        assert collected == ["Hello", " world", "!"]
        assert client.last_stream_response.content == "Hello world!"
        assert client.last_stream_response.tool_calls == []
        assert client.last_stream_response.input_tokens == 100

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self):
        chunks = _make_stream_chunks(
            text_deltas=[],
            tool_call_chunks=[
                {"index": 0, "id": "tc_1", "name": "search", "arguments": '{"q":'},
                {"index": 0, "id": None, "name": None, "arguments": ' "test"}'},
            ],
        )

        async def mock_acompletion(**kwargs):
            return _async_iter(chunks)

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            client = _make_client()
            collected = []
            async for delta in client.stream_complete(
                messages=[{"role": "user", "content": "search"}]
            ):
                collected.append(delta)

        assert collected == []  # no text deltas
        resp = client.last_stream_response
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "search"
        assert resp.tool_calls[0]["arguments"] == {"q": "test"}
        assert resp.tool_calls[0]["id"] == "tc_1"

    @pytest.mark.asyncio
    async def test_streaming_with_usage(self):
        chunks = _make_stream_chunks(
            text_deltas=["ok"],
            usage={"prompt_tokens": 200, "completion_tokens": 80},
        )

        async def mock_acompletion(**kwargs):
            return _async_iter(chunks)

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            client = _make_client()
            async for _ in client.stream_complete(
                messages=[{"role": "user", "content": "hi"}]
            ):
                pass

        assert client.last_stream_response.input_tokens == 200
        assert client.last_stream_response.output_tokens == 80

    @pytest.mark.asyncio
    async def test_stream_passes_kwargs(self):
        chunks = _make_stream_chunks(text_deltas=["ok"])

        async def mock_acompletion(**kwargs):
            assert kwargs["stream"] is True
            assert "stream_options" in kwargs
            return _async_iter(chunks)

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            client = _make_client()
            async for _ in client.stream_complete(
                messages=[{"role": "user", "content": "hi"}]
            ):
                pass


# ---------------------------------------------------------------------------
# LLMLoop.run_streaming() tests
# ---------------------------------------------------------------------------


def _text_response(content="Hello", input_tokens=100, output_tokens=50, cached=0):
    return LLMResponse(
        content=content,
        tool_calls=[],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        model="test-model",
    )


def _tool_response(tool_calls, content="", input_tokens=100, output_tokens=50, cached=0):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        model="test-model",
    )


class TestLoopRunStreaming:
    def _make_loop(self, stream_responses, tool_results=None, max_iterations=25):
        """Create an LLMLoop with a mocked stream_complete."""
        mock_llm = AsyncMock()

        call_count = 0

        async def mock_stream_complete(**kwargs):
            nonlocal call_count
            resp = stream_responses[call_count]
            call_count += 1
            # Yield text content character by character
            if resp.content:
                for ch in resp.content:
                    yield ch
            mock_llm.last_stream_response = resp

        mock_llm.stream_complete = mock_stream_complete

        mock_executor = AsyncMock()
        if tool_results:
            mock_executor.execute = AsyncMock(side_effect=tool_results)
        else:
            mock_executor.execute = AsyncMock(return_value="tool result")

        loop = LLMLoop(
            llm_client=mock_llm,
            tool_executor=mock_executor,
            max_iterations=max_iterations,
        )
        return loop, mock_llm, mock_executor

    @pytest.mark.asyncio
    async def test_text_only_streaming(self):
        loop, _, _ = self._make_loop([_text_response("Done")])
        events = []

        async def callback(event):
            events.append(event)

        result = await loop.run_streaming(
            [{"role": "user", "content": "hi"}],
            stream_callback=callback,
        )

        assert result.stopped_reason == "complete"
        assert result.response_text == "Done"
        # Should have text deltas + stream end
        text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        assert len(text_events) == 4  # D, o, n, e
        assert len(end_events) == 1
        assert end_events[0].full_text == "Done"

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls_and_status(self):
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {"q": "test"}}]
        responses = [
            _tool_response(tool_calls, content=""),
            _text_response("Found it!"),
        ]
        loop, _, executor = self._make_loop(responses, ["search result"])
        events = []

        async def callback(event):
            events.append(event)

        stream_content = StreamContentConfig(tool_status=True, tool_results=False)
        result = await loop.run_streaming(
            [{"role": "user", "content": "find test"}],
            stream_callback=callback,
            stream_content=stream_content,
        )

        assert result.stopped_reason == "complete"
        assert result.iterations == 2
        assert len(result.tool_calls) == 1

        tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "search"

        # tool_results=False means no ToolResultEvent
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 0

    @pytest.mark.asyncio
    async def test_streaming_with_tool_results(self):
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {"q": "test"}}]
        responses = [
            _tool_response(tool_calls),
            _text_response("Done"),
        ]
        loop, _, _ = self._make_loop(responses, ["found it"])
        events = []

        async def callback(event):
            events.append(event)

        stream_content = StreamContentConfig(tool_status=True, tool_results=True)
        result = await loop.run_streaming(
            [{"role": "user", "content": "go"}],
            stream_callback=callback,
            stream_content=stream_content,
        )

        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].result == "found it"
        assert tool_results[0].success is True

    @pytest.mark.asyncio
    async def test_streaming_returns_complete_loop_result(self):
        tool_calls = [{"id": "tc_1", "name": "t", "arguments": {}}]
        responses = [
            _tool_response(tool_calls, input_tokens=200, output_tokens=100, cached=50),
            _text_response("done", input_tokens=300, output_tokens=150, cached=100),
        ]
        loop, _, _ = self._make_loop(responses, ["ok"])

        result = await loop.run_streaming(
            [{"role": "user", "content": "go"}],
            stream_callback=AsyncMock(),
        )

        assert result.total_input_tokens == 500
        assert result.total_output_tokens == 250
        assert result.cached_tokens == 150
        assert result.iterations == 2

    @pytest.mark.asyncio
    async def test_streaming_error_sends_stream_end(self):
        mock_llm = AsyncMock()

        async def failing_stream(**kwargs):
            raise RuntimeError("API down")
            yield  # make it a generator  # noqa: E501

        mock_llm.stream_complete = failing_stream
        mock_executor = AsyncMock()

        loop = LLMLoop(llm_client=mock_llm, tool_executor=mock_executor)
        events = []

        async def callback(event):
            events.append(event)

        result = await loop.run_streaming(
            [{"role": "user", "content": "hi"}],
            stream_callback=callback,
        )

        assert result.stopped_reason == "error"
        assert "API down" in result.response_text
        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        assert len(end_events) == 1

    @pytest.mark.asyncio
    async def test_streaming_max_iterations(self):
        tool_calls = [{"id": "tc_1", "name": "loop_tool", "arguments": {}}]
        responses = [_tool_response(tool_calls) for _ in range(3)]
        loop, _, _ = self._make_loop(responses, max_iterations=3)
        events = []

        async def callback(event):
            events.append(event)

        result = await loop.run_streaming(
            [{"role": "user", "content": "go"}],
            stream_callback=callback,
        )

        assert result.stopped_reason == "max_iterations"
        assert result.iterations == 3
        end_events = [e for e in events if isinstance(e, StreamEndEvent)]
        assert len(end_events) == 1

    @pytest.mark.asyncio
    async def test_no_stream_content_skips_tool_events(self):
        """When stream_content is None, no tool events are pushed."""
        tool_calls = [{"id": "tc_1", "name": "search", "arguments": {}}]
        responses = [
            _tool_response(tool_calls),
            _text_response("Done"),
        ]
        loop, _, _ = self._make_loop(responses, ["ok"])
        events = []

        async def callback(event):
            events.append(event)

        result = await loop.run_streaming(
            [{"role": "user", "content": "go"}],
            stream_callback=callback,
            stream_content=None,
        )

        tool_events = [
            e for e in events if isinstance(e, (ToolStartEvent, ToolResultEvent))
        ]
        assert len(tool_events) == 0
