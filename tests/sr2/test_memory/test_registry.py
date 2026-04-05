"""Tests for the memory store registry."""

from unittest.mock import MagicMock, patch

import pytest

from sr2.memory.registry import (
    _reset_registry,
    get_store,
    list_stores,
    register_store,
)
from sr2.memory.store import InMemoryMemoryStore, SQLiteMemoryStore


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before and after each test."""
    _reset_registry()
    # Re-register built-ins so tests start from a known state
    register_store("memory", InMemoryMemoryStore)
    register_store("sqlite", SQLiteMemoryStore)
    yield
    _reset_registry()


class TestRegisterAndGetStore:
    def test_get_builtin_memory(self):
        assert get_store("memory") is InMemoryMemoryStore

    def test_get_builtin_sqlite(self):
        assert get_store("sqlite") is SQLiteMemoryStore

    def test_register_custom_store(self):
        class CustomStore:
            pass

        register_store("custom", CustomStore)
        assert get_store("custom") is CustomStore

    def test_get_nonexistent_raises_import_error(self):
        with pytest.raises(ImportError, match="Memory store 'nonexistent' is not available"):
            get_store("nonexistent")

    def test_error_message_includes_available_backends(self):
        with pytest.raises(ImportError, match="'memory'"):
            get_store("nonexistent")

    def test_error_message_includes_upgrade_hint(self):
        with pytest.raises(ImportError, match="pip install sr2-pro"):
            get_store("postgres")

    def test_register_overwrites_existing(self):
        class NewMemoryStore:
            pass

        register_store("memory", NewMemoryStore)
        assert get_store("memory") is NewMemoryStore

    def test_list_stores(self):
        stores = list_stores()
        assert "memory" in stores
        assert "sqlite" in stores


class TestEntryPointDiscovery:
    def test_entry_points_discovered_on_missing_store(self):
        """Entry point discovery triggers when requesting an unregistered store."""
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "test_store"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_store("test_store")
            mock_ep.load.assert_called_once()

    def test_entry_points_discovered_only_once(self):
        """Discovery runs at most once, even if called multiple times."""
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "test_store"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_store("missing1")
            with pytest.raises(ImportError):
                get_store("missing2")
            # entry_points() called only once
            mock_ep.load.assert_called_once()

    def test_broken_entry_point_does_not_crash(self):
        """A broken plugin shouldn't prevent other stores from loading."""
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = RuntimeError("plugin broken")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_store("anything")
            # Didn't crash — just logged and continued

    def test_entry_point_registers_store(self):
        """Entry point load() can call register_store to make a backend available."""
        _reset_registry()

        class PluginStore:
            pass

        def load_plugin():
            register_store("plugin", PluginStore)

        mock_ep = MagicMock()
        mock_ep.name = "plugin"
        mock_ep.load.return_value = load_plugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = get_store("plugin")
            assert result is PluginStore
