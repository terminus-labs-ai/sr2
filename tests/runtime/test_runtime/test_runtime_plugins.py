"""Tests for Agent runtime plugin discovery."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLoadRuntimePlugins:
    """Agent._load_runtime_plugins() discovers and initializes plugins via entry points."""

    @pytest.mark.asyncio
    async def test_no_plugins_installed(self):
        """When no entry points exist, returns empty list."""
        from sr2_runtime.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._load_runtime_plugins = Agent._load_runtime_plugins.__get__(agent)

        with patch("importlib.metadata.entry_points", return_value=[]):
            result = await agent._load_runtime_plugins()

        assert result == []

    @pytest.mark.asyncio
    async def test_plugin_returns_none_not_added(self):
        """If register() returns None (disabled), plugin is not added."""
        from sr2_runtime.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._load_runtime_plugins = Agent._load_runtime_plugins.__get__(agent)

        ep = MagicMock()
        ep.name = "test_plugin"
        ep.load.return_value = MagicMock(return_value=None)

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            result = await agent._load_runtime_plugins()

        assert result == []

    @pytest.mark.asyncio
    async def test_plugin_loaded_successfully(self):
        """Valid plugin is loaded and added to the list."""
        from sr2_runtime.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._load_runtime_plugins = Agent._load_runtime_plugins.__get__(agent)

        mock_plugin = MagicMock()
        ep = MagicMock()
        ep.name = "test_plugin"
        ep.load.return_value = MagicMock(return_value=mock_plugin)

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            result = await agent._load_runtime_plugins()

        assert len(result) == 1
        assert result[0] is mock_plugin

    @pytest.mark.asyncio
    async def test_plugin_failure_does_not_crash(self):
        """If a plugin's register() raises, it's skipped gracefully."""
        from sr2_runtime.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._load_runtime_plugins = Agent._load_runtime_plugins.__get__(agent)

        ep = MagicMock()
        ep.name = "broken_plugin"
        ep.load.return_value = MagicMock(side_effect=RuntimeError("license check failed"))

        with patch("importlib.metadata.entry_points", return_value=[ep]):
            result = await agent._load_runtime_plugins()

        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_plugins(self):
        """Multiple plugins can be loaded."""
        from sr2_runtime.agent import Agent

        agent = MagicMock(spec=Agent)
        agent._load_runtime_plugins = Agent._load_runtime_plugins.__get__(agent)

        plugin_a = MagicMock()
        plugin_b = MagicMock()

        ep_a = MagicMock()
        ep_a.name = "plugin_a"
        ep_a.load.return_value = MagicMock(return_value=plugin_a)

        ep_b = MagicMock()
        ep_b.name = "plugin_b"
        ep_b.load.return_value = MagicMock(return_value=plugin_b)

        with patch("importlib.metadata.entry_points", return_value=[ep_a, ep_b]):
            result = await agent._load_runtime_plugins()

        assert len(result) == 2
