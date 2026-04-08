"""Tests for the unified PluginRegistry."""

from unittest.mock import MagicMock, patch

import pytest

from sr2.plugins.registry import PluginRegistry
from sr2.plugins.errors import PluginLicenseError, PluginNotFoundError


@pytest.fixture
def registry():
    return PluginRegistry("sr2.test_group", install_hint="pip install sr2-test")


class TestRegisterAndGet:
    def test_register_and_get(self, registry):
        class MyPlugin:
            pass

        registry.register("my_plugin", MyPlugin)
        assert registry.get("my_plugin") is MyPlugin

    def test_get_nonexistent_raises_not_found(self, registry):
        with pytest.raises(PluginNotFoundError, match="not available"):
            registry.get("nonexistent")

    def test_error_includes_group_name(self, registry):
        with pytest.raises(PluginNotFoundError, match="sr2.test_group"):
            registry.get("missing")

    def test_error_includes_install_hint(self, registry):
        with pytest.raises(PluginNotFoundError, match="pip install sr2-test"):
            registry.get("missing")

    def test_error_includes_available_plugins(self, registry):
        class A:
            pass

        registry.register("available_one", A)
        with pytest.raises(PluginNotFoundError, match="available_one"):
            registry.get("missing")

    def test_overwrite_existing(self, registry):
        class V1:
            pass

        class V2:
            pass

        registry.register("plugin", V1)
        registry.register("plugin", V2)
        assert registry.get("plugin") is V2

    def test_not_found_is_import_error(self, registry):
        """PluginNotFoundError is a subclass of ImportError for backward compat."""
        with pytest.raises(ImportError):
            registry.get("missing")


class TestListAvailable:
    def test_empty_registry(self, registry):
        assert registry.list_available() == []

    def test_lists_registered_plugins(self, registry):
        class A:
            pass

        class B:
            pass

        registry.register("beta", B)
        registry.register("alpha", A)
        assert registry.list_available() == ["alpha", "beta"]


class TestEntryPointDiscovery:
    def test_discovery_triggers_on_miss(self, registry):
        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"

        def _mock_eps(group=""):
            if group == "sr2.test_group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            with pytest.raises(PluginNotFoundError):
                registry.get("test_plugin")
            mock_ep.load.assert_called_once()

    def test_discovery_runs_only_once(self, registry):
        mock_ep = MagicMock()
        mock_ep.name = "test"

        def _mock_eps(group=""):
            if group == "sr2.test_group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            with pytest.raises(PluginNotFoundError):
                registry.get("miss1")
            with pytest.raises(PluginNotFoundError):
                registry.get("miss2")
            mock_ep.load.assert_called_once()

    def test_entry_point_registers_plugin(self, registry):
        class PluginCls:
            pass

        def load_plugin():
            registry.register("discovered", PluginCls)

        mock_ep = MagicMock()
        mock_ep.name = "discovered"
        mock_ep.load.return_value = load_plugin

        def _mock_eps(group=""):
            if group == "sr2.test_group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            result = registry.get("discovered")
            assert result is PluginCls

    def test_broken_entry_point_does_not_crash(self, registry):
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = RuntimeError("plugin broken")

        def _mock_eps(group=""):
            if group == "sr2.test_group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            with pytest.raises(PluginNotFoundError):
                registry.get("anything")


class TestLicenseValidation:
    def test_license_error_stored_separately(self, registry):
        mock_ep = MagicMock()
        mock_ep.name = "premium"

        def raise_license():
            raise PluginLicenseError("License required")

        mock_ep.load.return_value = raise_license

        def _mock_eps(group=""):
            if group == "sr2.test_group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            with pytest.raises(PluginLicenseError, match="requires a valid license"):
                registry.get("premium")

    def test_license_error_is_import_error(self, registry):
        """PluginLicenseError is a subclass of ImportError for backward compat."""
        mock_ep = MagicMock()
        mock_ep.name = "premium"

        def raise_license():
            raise PluginLicenseError("License required")

        mock_ep.load.return_value = raise_license

        def _mock_eps(group=""):
            if group == "sr2.test_group":
                return [mock_ep]
            return []

        with patch("importlib.metadata.entry_points", side_effect=_mock_eps):
            with pytest.raises(ImportError):
                registry.get("premium")


class TestReset:
    def test_reset_clears_everything(self, registry):
        class A:
            pass

        registry.register("a", A)
        assert "a" in registry._registry

        registry._reset()
        assert registry._registry == {}
        assert registry._license_errors == {}
        assert not registry._discovered
