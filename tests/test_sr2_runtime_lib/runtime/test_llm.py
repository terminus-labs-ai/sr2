"""Tests for SR2Runtime LLM client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from sr2.runtime.config import ModelConfig
from sr2.runtime.llm import LLMResponse, RuntimeLLMClient


# ---------------------------------------------------------------------------
# Helpers — lightweight mock objects matching LiteLLM response shape
# ---------------------------------------------------------------------------

def _make_response(
    content: str | None = "Hello",
    tool_calls: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
):
    """Build a mock LiteLLM response object."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tool_call(
    tc_id: str = "call_1",
    name: str = "get_weather",
    arguments: str = '{"city": "London"}',
):
    """Build a mock tool call object."""
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=tc_id, function=func)


# ---------------------------------------------------------------------------
# Model string construction
# ---------------------------------------------------------------------------

class TestBuildModelString:
    def test_ollama_uses_ollama_chat_prefix(self):
        cfg = ModelConfig(provider="ollama", name="llama3")
        client = RuntimeLLMClient(cfg)
        assert client._model_string == "ollama_chat/llama3"

    def test_litellm_passes_name_through(self):
        cfg = ModelConfig(provider="litellm", name="gpt-4o", base_url="")
        client = RuntimeLLMClient(cfg)
        assert client._model_string == "gpt-4o"

    def test_openai_compat_uses_openai_prefix(self):
        cfg = ModelConfig(
            provider="openai-compat", name="my-model", base_url="http://x:8080"
        )
        client = RuntimeLLMClient(cfg)
        assert client._model_string == "openai/my-model"

    def test_anthropic_uses_provider_prefix(self):
        cfg = ModelConfig(provider="anthropic", name="claude-3-opus", base_url="")
        client = RuntimeLLMClient(cfg)
        assert client._model_string == "anthropic/claude-3-opus"

    def test_provider_case_insensitive(self):
        cfg = ModelConfig(provider="Ollama", name="llama3")
        client = RuntimeLLMClient(cfg)
        assert client._model_string == "ollama_chat/llama3"


# ---------------------------------------------------------------------------
# LLMResponse.from_litellm — content-only
# ---------------------------------------------------------------------------

class TestFromLitellmContentOnly:
    def test_parses_content(self):
        resp = _make_response(content="Hello world")
        parsed = LLMResponse.from_litellm(resp)
        assert parsed.content == "Hello world"
        assert parsed.tool_calls == []

    def test_none_content(self):
        resp = _make_response(content=None)
        parsed = LLMResponse.from_litellm(resp)
        assert parsed.content is None


# ---------------------------------------------------------------------------
# LLMResponse.from_litellm — tool calls
# ---------------------------------------------------------------------------

class TestFromLitellmToolCalls:
    def test_parses_single_tool_call(self):
        tc = _make_tool_call()
        resp = _make_response(content=None, tool_calls=[tc])
        parsed = LLMResponse.from_litellm(resp)

        assert len(parsed.tool_calls) == 1
        assert parsed.tool_calls[0]["id"] == "call_1"
        assert parsed.tool_calls[0]["name"] == "get_weather"
        assert parsed.tool_calls[0]["arguments"] == {"city": "London"}

    def test_parses_multiple_tool_calls(self):
        tc1 = _make_tool_call(tc_id="c1", name="func_a", arguments='{"x": 1}')
        tc2 = _make_tool_call(tc_id="c2", name="func_b", arguments='{"y": 2}')
        resp = _make_response(content=None, tool_calls=[tc1, tc2])
        parsed = LLMResponse.from_litellm(resp)

        assert len(parsed.tool_calls) == 2
        assert parsed.tool_calls[0]["name"] == "func_a"
        assert parsed.tool_calls[1]["name"] == "func_b"

    def test_malformed_arguments_default_to_empty_dict(self):
        tc = _make_tool_call(arguments="not valid json {{{")
        resp = _make_response(content=None, tool_calls=[tc])
        parsed = LLMResponse.from_litellm(resp)

        assert parsed.tool_calls[0]["arguments"] == {}

    def test_none_arguments_attribute_defaults_to_empty_dict(self):
        """Covers the AttributeError branch."""
        func = SimpleNamespace(name="do_thing", arguments=None)
        tc = SimpleNamespace(id="c1", function=func)
        # json.loads(None) raises TypeError, caught by except clause
        resp = _make_response(content=None, tool_calls=[tc])
        parsed = LLMResponse.from_litellm(resp)

        assert parsed.tool_calls[0]["arguments"] == {}


# ---------------------------------------------------------------------------
# LLMResponse.from_litellm — token usage
# ---------------------------------------------------------------------------

class TestFromLitellmUsage:
    def test_extracts_token_counts(self):
        resp = _make_response(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        parsed = LLMResponse.from_litellm(resp)
        assert parsed.prompt_tokens == 100
        assert parsed.completion_tokens == 50
        assert parsed.total_tokens == 150

    def test_missing_usage_fields_default_to_zero(self):
        """Provider returns None for some usage fields."""
        msg = SimpleNamespace(content="hi", tool_calls=None)
        choice = SimpleNamespace(message=msg)
        usage = SimpleNamespace(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
        resp = SimpleNamespace(choices=[choice], usage=usage)
        parsed = LLMResponse.from_litellm(resp)

        assert parsed.prompt_tokens == 0
        assert parsed.completion_tokens == 0
        assert parsed.total_tokens == 0


# ---------------------------------------------------------------------------
# RuntimeLLMClient.complete — verify kwargs passed to litellm
# ---------------------------------------------------------------------------

class TestComplete:
    @pytest.fixture
    def client(self):
        cfg = ModelConfig(
            provider="ollama",
            name="llama3",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2048,
        )
        return RuntimeLLMClient(cfg)

    @pytest.mark.asyncio
    async def test_basic_call_kwargs(self, client):
        mock_resp = _make_response()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_resp
            messages = [{"role": "user", "content": "hi"}]
            result = await client.complete(messages)

            mock_ac.assert_called_once()
            kwargs = mock_ac.call_args.kwargs
            assert kwargs["model"] == "ollama_chat/llama3"
            assert kwargs["messages"] == messages
            assert kwargs["temperature"] == 0.7
            assert kwargs["max_tokens"] == 2048
            assert kwargs["api_base"] == "http://localhost:11434"
            assert "tools" not in kwargs

            assert isinstance(result, LLMResponse)
            assert result.content == "Hello"

    @pytest.mark.asyncio
    async def test_tools_are_wrapped(self, client):
        mock_resp = _make_response()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_resp
            tools = [{"name": "search", "parameters": {"type": "object"}}]
            await client.complete(
                messages=[{"role": "user", "content": "find"}],
                tools=tools,
            )

            kwargs = mock_ac.call_args.kwargs
            assert kwargs["tools"] == [
                {"type": "function", "function": tools[0]}
            ]

    @pytest.mark.asyncio
    async def test_tool_choice_passed(self, client):
        mock_resp = _make_response()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_resp
            await client.complete(
                messages=[{"role": "user", "content": "x"}],
                tool_choice="auto",
            )

            kwargs = mock_ac.call_args.kwargs
            assert kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_no_api_base_when_empty(self):
        cfg = ModelConfig(
            provider="litellm", name="gpt-4o", base_url=""
        )
        client = RuntimeLLMClient(cfg)
        mock_resp = _make_response()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_resp
            await client.complete(
                messages=[{"role": "user", "content": "x"}],
            )

            kwargs = mock_ac.call_args.kwargs
            assert "api_base" not in kwargs

    @pytest.mark.asyncio
    async def test_extra_config_forwarded(self):
        cfg = ModelConfig(
            provider="litellm",
            name="gpt-4o",
            base_url="",
            extra={"top_p": 0.9, "seed": 42},
        )
        client = RuntimeLLMClient(cfg)
        mock_resp = _make_response()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_resp
            await client.complete(
                messages=[{"role": "user", "content": "x"}],
            )

            kwargs = mock_ac.call_args.kwargs
            assert kwargs["top_p"] == 0.9
            assert kwargs["seed"] == 42

    @pytest.mark.asyncio
    async def test_returns_parsed_response(self, client):
        tc = _make_tool_call()
        mock_resp = _make_response(
            content="partial",
            tool_calls=[tc],
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_ac:
            mock_ac.return_value = mock_resp
            result = await client.complete(
                messages=[{"role": "user", "content": "x"}],
            )

            assert result.content == "partial"
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0]["name"] == "get_weather"
            assert result.prompt_tokens == 20
            assert result.completion_tokens == 10
            assert result.total_tokens == 30
