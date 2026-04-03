"""Tests for the degradation policy registry."""

from unittest.mock import MagicMock, patch

import pytest

from sr2.degradation.registry import (
    _reset_registry,
    get_policy,
    list_policies,
    register_policy,
)
from sr2.degradation.ladder import DegradationLadder


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before and after each test."""
    _reset_registry()
    register_policy("ladder", DegradationLadder)
    yield
    _reset_registry()


class TestRegisterAndGetPolicy:
    def test_get_builtin_ladder(self):
        assert get_policy("ladder") is DegradationLadder

    def test_register_custom_policy(self):
        class CustomPolicy:
            pass

        register_policy("custom", CustomPolicy)
        assert get_policy("custom") is CustomPolicy

    def test_get_nonexistent_raises_import_error(self):
        with pytest.raises(ImportError, match="Degradation policy 'nonexistent' is not available"):
            get_policy("nonexistent")

    def test_error_message_includes_upgrade_hint(self):
        with pytest.raises(ImportError, match="pip install sr2-pro"):
            get_policy("sla")

    def test_list_policies(self):
        policies = list_policies()
        assert "ladder" in policies


class TestEntryPointDiscovery:
    def test_entry_points_discovered_on_missing_policy(self):
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "test_policy"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_policy("test_policy")
            mock_ep.load.assert_called_once()

    def test_entry_points_discovered_only_once(self):
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "test"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_policy("missing1")
            with pytest.raises(ImportError):
                get_policy("missing2")
            mock_ep.load.assert_called_once()

    def test_broken_entry_point_does_not_crash(self):
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = RuntimeError("plugin broken")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_policy("anything")

    def test_entry_point_registers_policy(self):
        _reset_registry()

        class PluginPolicy:
            pass

        def load_plugin():
            register_policy("plugin", PluginPolicy)

        mock_ep = MagicMock()
        mock_ep.name = "plugin"
        mock_ep.load.side_effect = load_plugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = get_policy("plugin")
            assert result is PluginPolicy
