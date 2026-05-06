"""Tests for PluginRegistry dependency injection.

Verifies that:
- PluginRegistry accepts an optional deps dict at init
- Deps are passed as **kwargs to factory functions during _load
- Backward compatibility: no deps = no kwargs passed
- clear_cache and list_available still work
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sr2.plugins.registry import PluginRegistry


class TestPluginRegistryInit:
    """PluginRegistry construction with and without deps."""

    def test_accepts_deps_dict(self):
        mock_llm = MagicMock()
        registry = PluginRegistry("sr2.test", deps={"llm": mock_llm})
        assert registry._deps == {"llm": mock_llm}

    def test_works_without_deps(self):
        registry = PluginRegistry("sr2.test")
        assert registry._deps == {}

    def test_none_deps_becomes_empty_dict(self):
        registry = PluginRegistry("sr2.test", deps=None)
        assert registry._deps == {}

    def test_empty_deps_dict(self):
        registry = PluginRegistry("sr2.test", deps={})
        assert registry._deps == {}


class TestDepsPassedToFactory:
    """Factory functions receive deps as **kwargs during _load."""

    def test_factory_function_receives_deps(self):
        """A factory function gets deps as keyword arguments."""
        received = {}

        def mock_factory(**kwargs):
            received.update(kwargs)
            return "plugin_instance"

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_factory

        mock_llm = MagicMock()
        registry = PluginRegistry("sr2.test", deps={"llm": mock_llm})

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = registry.get("test_plugin")

        assert result == "plugin_instance"
        assert "llm" in received
        assert received["llm"] is mock_llm

    def test_class_factory_receives_deps(self):
        """A class entry point gets deps as constructor kwargs."""
        received = {}

        class MockPlugin:
            def __init__(self, **kwargs):
                received.update(kwargs)

        mock_ep = MagicMock()
        mock_ep.load.return_value = MockPlugin

        mock_llm = MagicMock()
        registry = PluginRegistry("sr2.test", deps={"llm": mock_llm})

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = registry.get("test_plugin")

        assert isinstance(result, MockPlugin)
        assert "llm" in received
        assert received["llm"] is mock_llm

    def test_factory_no_deps_gets_no_kwargs(self):
        """When no deps are configured, factory is called with no kwargs."""
        received = {}

        def mock_factory(**kwargs):
            received.update(kwargs)
            return "instance"

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_factory

        registry = PluginRegistry("sr2.test")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = registry.get("test_plugin")

        assert result == "instance"
        assert received == {}

    def test_factory_ignores_unknown_deps(self):
        """Factory with **kwargs gracefully receives extra deps it doesn't use."""

        def mock_factory(llm=None, **kwargs):
            return {"llm": llm, "extra": kwargs}

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_factory

        mock_llm = MagicMock()
        registry = PluginRegistry(
            "sr2.test",
            deps={"llm": mock_llm, "tokenizer": "unused", "extra_thing": 42},
        )

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = registry.get("test_plugin")

        assert result["llm"] is mock_llm
        assert result["extra"] == {"tokenizer": "unused", "extra_thing": 42}

    def test_multiple_deps_all_passed(self):
        """All deps in the dict are forwarded to the factory."""
        received = {}

        def mock_factory(**kwargs):
            received.update(kwargs)
            return "instance"

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_factory

        deps = {"llm": "llm_val", "tokenizer": "tok_val", "store": "store_val"}
        registry = PluginRegistry("sr2.test", deps=deps)

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.get("test_plugin")

        assert received == deps


class TestCacheBehavior:
    """Cache and list_available still work correctly with deps."""

    def test_clear_cache(self):
        def mock_factory(**kwargs):
            return "instance"

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_factory

        registry = PluginRegistry("sr2.test", deps={"llm": "val"})

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.get("plugin_a")
            assert "plugin_a" in registry._cache

            registry.clear_cache()
            assert registry._cache == {}

    def test_cached_plugin_not_reloaded(self):
        """Second get() returns cached instance, doesn't call _load again."""
        call_count = 0

        def mock_factory(**kwargs):
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        mock_ep = MagicMock()
        mock_ep.load.return_value = mock_factory

        registry = PluginRegistry("sr2.test", deps={"llm": "val"})

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            first = registry.get("plugin_a")
            second = registry.get("plugin_a")

        assert first == second == "instance_1"
        assert call_count == 1

    def test_list_available(self):
        mock_ep1 = MagicMock()
        mock_ep1.name = "plugin_a"
        mock_ep2 = MagicMock()
        mock_ep2.name = "plugin_b"

        registry = PluginRegistry("sr2.test", deps={"llm": "val"})

        with patch("importlib.metadata.entry_points", return_value=[mock_ep1, mock_ep2]):
            available = registry.list_available()

        assert available == ["plugin_a", "plugin_b"]
