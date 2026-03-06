"""Tests for runtime LLM client wrapper."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from runtime.config import LLMModelConfig
from runtime.llm import LLMClient, LLMResponse


def _make_client(**overrides):
    """Create LLMClient with default LLMModelConfig objects."""
    from runtime.config import LLMModelConfig
    defaults = {
        "model": LLMModelConfig(name="claude-sonnet-4-20250514"),
        "fast_model": LLMModelConfig(name="claude-haiku-4-5-20251001", max_tokens=1000),
        "embedding": LLMModelConfig(name="text-embedding-3-small"),
    }
    defaults.update(overrides)
    return LLMClient(**defaults)


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_has_tool_calls_true(self):
        resp = LLMResponse(
            content="",
            tool_calls=[{"id": "1", "name": "search", "arguments": {}}],
            input_tokens=100,
            output_tokens=50,
        )
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false(self):
        resp = LLMResponse(content="Hello", input_tokens=100, output_tokens=50)
        assert resp.has_tool_calls is False

    def test_cache_hit_rate_computed(self):
        resp = LLMResponse(input_tokens=1000, cached_tokens=750)
        assert resp.cache_hit_rate == 0.75

    def test_cache_hit_rate_zero_input(self):
        resp = LLMResponse(input_tokens=0, cached_tokens=0)
        assert resp.cache_hit_rate == 0.0


def _make_mock_response(
    content="Hello",
    tool_calls=None,
    prompt_tokens=100,
    completion_tokens=50,
    cached_tokens_anthropic=None,
    cached_tokens_openai=None,
):
    """Build a mock LiteLLM response object."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage_attrs = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if cached_tokens_anthropic is not None:
        usage_attrs["cache_read_input_tokens"] = cached_tokens_anthropic
    if cached_tokens_openai is not None:
        usage_attrs["prompt_tokens_details"] = SimpleNamespace(
            cached_tokens=cached_tokens_openai
        )
    usage = SimpleNamespace(**usage_attrs)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestLLMClient:
    """Tests for the LLMClient wrapper."""

    @pytest.mark.asyncio
    async def test_complete_returns_response_with_content(self):
        mock_resp = _make_mock_response(content="Test response")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            client = _make_client()
            result = await client.complete(
                messages=[{"role": "user", "content": "Hi"}]
            )
        assert isinstance(result, LLMResponse)
        assert result.content == "Test response"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self):
        tool_call = SimpleNamespace(
            id="call_123",
            function=SimpleNamespace(
                name="search",
                arguments='{"query": "test"}',
            ),
        )
        mock_resp = _make_mock_response(content="", tool_calls=[tool_call])
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            client = _make_client()
            result = await client.complete(
                messages=[{"role": "user", "content": "search for test"}]
            )
        assert result.has_tool_calls is True
        assert result.tool_calls[0]["name"] == "search"
        assert result.tool_calls[0]["arguments"] == {"query": "test"}
        assert result.tool_calls[0]["id"] == "call_123"

    @pytest.mark.asyncio
    async def test_complete_without_tool_calls(self):
        mock_resp = _make_mock_response(content="Just text")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            client = _make_client()
            result = await client.complete(
                messages=[{"role": "user", "content": "hello"}]
            )
        assert result.has_tool_calls is False

    @pytest.mark.asyncio
    async def test_fast_complete_uses_fast_model(self):
        mock_resp = _make_mock_response(content="Fast response")
        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_call:
            client = _make_client(
                model=LLMModelConfig(name="claude-sonnet-4-20250514"),
                fast_model=LLMModelConfig(name="claude-haiku-4-5-20251001", max_tokens=1000),
            )
            result = await client.fast_complete("You are helpful.", "Summarize this.")
        assert result == "Fast response"
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_tool_call_malformed_json_handled(self):
        tool_call = SimpleNamespace(
            id="call_bad",
            function=SimpleNamespace(
                name="broken_tool",
                arguments="not valid json{{{",
            ),
        )
        mock_resp = _make_mock_response(content="", tool_calls=[tool_call])
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            client = _make_client()
            result = await client.complete(
                messages=[{"role": "user", "content": "test"}]
            )
        assert result.has_tool_calls is True
        assert result.tool_calls[0]["arguments"] == {}

    def test_extract_cached_tokens_anthropic_format(self):
        client = _make_client()
        usage = SimpleNamespace(cache_read_input_tokens=500)
        assert client._extract_cached_tokens(usage) == 500

    def test_extract_cached_tokens_openai_format(self):
        client = _make_client()
        usage = SimpleNamespace(
            prompt_tokens_details=SimpleNamespace(cached_tokens=300)
        )
        assert client._extract_cached_tokens(usage) == 300

    def test_extract_cached_tokens_missing_returns_zero(self):
        client = _make_client()
        usage = SimpleNamespace(prompt_tokens=100)
        assert client._extract_cached_tokens(usage) == 0


class TestTryParseToolCall:
    """Tests for _try_parse_tool_call with tool name validation."""

    def test_valid_tool_call_accepted(self):
        result = LLMClient._try_parse_tool_call(
            '{"name": "search_repos", "arguments": {"query": "test"}}',
            available_tools={"search_repos", "get_file"},
        )
        assert result is not None
        assert result["name"] == "search_repos"
        assert result["arguments"] == {"query": "test"}

    def test_hallucinated_tool_rejected(self):
        result = LLMClient._try_parse_tool_call(
            '{"name": "list_repositories", "arguments": {}}',
            available_tools={"search_repositories", "get_file_contents"},
        )
        assert result is None

    def test_no_available_tools_accepts_any(self):
        result = LLMClient._try_parse_tool_call(
            '{"name": "anything", "arguments": {}}',
            available_tools=None,
        )
        assert result is not None
        assert result["name"] == "anything"

    def test_tool_call_wrapper_accepted(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_call>{"name": "search", "arguments": {}}</tool_call>',
            available_tools={"search"},
        )
        assert result is not None
        assert result["name"] == "search"

    def test_plain_text_returns_none(self):
        result = LLMClient._try_parse_tool_call(
            "This is just regular text, not a tool call",
            available_tools={"search"},
        )
        assert result is None

    def test_invalid_json_returns_none(self):
        result = LLMClient._try_parse_tool_call(
            "{'name': 'search'}",  # Python repr, not JSON
            available_tools={"search"},
        )
        assert result is None


class TestToolCodeParsing:
    """Tests for <tool_code>function_name(args)</tool_code> parsing."""

    def test_tool_code_with_string_arg(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_code>manage_tasks("List all tasks")</tool_code>',
            available_tools={"manage_tasks"},
        )
        assert result is not None
        assert result["name"] == "manage_tasks"
        assert result["arguments"] == {"input": "List all tasks"}

    def test_tool_code_embedded_in_text(self):
        """Model outputs text before the tool_code block (common pattern)."""
        content = (
            "Let me check the current state of the Galaxy Map task board.\n\n"
            "<tool_code>\nmanage_tasks(\"List all tasks\")\n</tool_code>"
        )
        result = LLMClient._try_parse_tool_call(content, available_tools={"manage_tasks"})
        assert result is not None
        assert result["name"] == "manage_tasks"

    def test_tool_code_with_kwargs(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_code>search(query="galaxy map", limit=10)</tool_code>',
            available_tools={"search"},
        )
        assert result is not None
        assert result["name"] == "search"
        assert result["arguments"]["query"] == "galaxy map"

    def test_tool_code_with_json_arg(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_code>search({"query": "galaxy"})</tool_code>',
            available_tools={"search"},
        )
        assert result is not None
        assert result["arguments"] == {"query": "galaxy"}

    def test_tool_code_no_args(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_code>list_tasks()</tool_code>',
            available_tools={"list_tasks"},
        )
        assert result is not None
        assert result["name"] == "list_tasks"
        assert result["arguments"] == {}

    def test_tool_code_hallucinated_name_rejected(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_code>nonexistent_tool("test")</tool_code>',
            available_tools={"search", "manage_tasks"},
        )
        assert result is None

    def test_tool_code_no_available_tools_accepts_any(self):
        result = LLMClient._try_parse_tool_call(
            '<tool_code>any_tool("test")</tool_code>',
            available_tools=None,
        )
        assert result is not None
        assert result["name"] == "any_tool"

    def test_looks_like_tool_call_detects_tool_code(self):
        assert LLMClient._looks_like_tool_call(
            'Some text\n<tool_code>func("arg")</tool_code>'
        ) is True

    def test_looks_like_tool_call_plain_text_false(self):
        assert LLMClient._looks_like_tool_call("Just regular text") is False
