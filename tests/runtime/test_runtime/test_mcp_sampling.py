"""Tests for MCP sampling callback and rate limiting."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2_runtime.config import MCPSamplingConfig, MCPServerConfig
from sr2_runtime.mcp.client import MCPManager


class TestSamplingCallback:
    """Tests for MCPManager._build_sampling_callback."""

    def test_returns_none_when_disabled(self):
        mgr = MCPManager()
        config = MCPServerConfig(
            name="test", url="test",
            sampling=MCPSamplingConfig(enabled=False),
        )
        assert mgr._build_sampling_callback(config) is None

    def test_returns_none_when_deny_policy(self):
        mgr = MCPManager()
        config = MCPServerConfig(
            name="test", url="test",
            sampling=MCPSamplingConfig(enabled=True, policy="deny"),
        )
        assert mgr._build_sampling_callback(config) is None

    def test_returns_callable_when_enabled(self):
        mgr = MCPManager()
        config = MCPServerConfig(
            name="test", url="test",
            sampling=MCPSamplingConfig(enabled=True, policy="auto_approve"),
        )
        cb = mgr._build_sampling_callback(config)
        assert cb is not None
        assert callable(cb)

    @pytest.mark.asyncio
    async def test_sampling_calls_llm_client(self):
        mgr = MCPManager()

        # Mock LLM client
        llm_response = MagicMock()
        llm_response.content = "Generated response"
        llm_response.model = "test-model"
        llm_client = MagicMock()
        llm_client.complete = AsyncMock(return_value=llm_response)
        mgr.set_llm_client(llm_client)

        config = MCPServerConfig(
            name="test", url="test",
            sampling=MCPSamplingConfig(enabled=True, policy="auto_approve", max_tokens=512),
        )
        cb = mgr._build_sampling_callback(config)

        # Build mock params
        msg_content = MagicMock()
        msg_content.text = "Hello"
        msg = MagicMock()
        msg.role = "user"
        msg.content = msg_content

        params = MagicMock()
        params.systemPrompt = "You are helpful."
        params.messages = [msg]
        params.maxTokens = 1024

        context = MagicMock()
        result = await cb(context, params)

        # Should have called LLM with capped max_tokens
        llm_client.complete.assert_called_once()
        call_kwargs = llm_client.complete.call_args
        assert call_kwargs[1]["max_tokens"] == 512  # capped to config max

        # Check result structure
        assert result.content.text == "Generated response"
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_sampling_no_llm_client_returns_error(self):
        mgr = MCPManager()
        # Don't set LLM client

        config = MCPServerConfig(
            name="test", url="test",
            sampling=MCPSamplingConfig(enabled=True, policy="auto_approve"),
        )
        cb = mgr._build_sampling_callback(config)

        msg_content = MagicMock()
        msg_content.text = "Hello"
        msg = MagicMock()
        msg.role = "user"
        msg.content = msg_content

        params = MagicMock()
        params.systemPrompt = None
        params.messages = [msg]
        params.maxTokens = 100

        context = MagicMock()
        result = await cb(context, params)

        assert hasattr(result, "message")
        assert "No LLM client" in result.message


class TestRateLimit:
    """Tests for MCPManager._check_rate_limit."""

    def test_allows_within_limit(self):
        mgr = MCPManager()
        assert mgr._check_rate_limit("s1", 5) is True
        assert mgr._check_rate_limit("s1", 5) is True

    def test_blocks_over_limit(self):
        mgr = MCPManager()
        for _ in range(5):
            mgr._check_rate_limit("s1", 5)
        assert mgr._check_rate_limit("s1", 5) is False

    def test_separate_servers_independent(self):
        mgr = MCPManager()
        for _ in range(5):
            mgr._check_rate_limit("s1", 5)
        # s2 should still be allowed
        assert mgr._check_rate_limit("s2", 5) is True

    def test_old_timestamps_expire(self):
        mgr = MCPManager()
        # Inject old timestamps
        mgr._sampling_timestamps["s1"] = [time.time() - 120] * 10
        assert mgr._check_rate_limit("s1", 5) is True


class TestRootsCallback:
    """Tests for MCPManager._build_roots_callback."""

    def test_returns_none_when_no_roots(self):
        config = MCPServerConfig(name="test", url="test")
        cb = MCPManager._build_roots_callback(config)
        assert cb is None

    def test_returns_callable_when_roots_configured(self):
        config = MCPServerConfig(
            name="test", url="test",
            roots=["file:///home/user/project"],
        )
        cb = MCPManager._build_roots_callback(config)
        assert cb is not None
        assert callable(cb)

    @pytest.mark.asyncio
    async def test_roots_callback_returns_configured_uris(self):
        config = MCPServerConfig(
            name="test", url="test",
            roots=["file:///home/user/a", "file:///home/user/b"],
        )
        cb = MCPManager._build_roots_callback(config)

        context = MagicMock()
        result = await cb(context)

        assert len(result.roots) == 2
        uris = [str(r.uri) for r in result.roots]
        assert "file:///home/user/a" in uris
        assert "file:///home/user/b" in uris
