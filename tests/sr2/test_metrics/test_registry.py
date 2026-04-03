"""Tests for the metrics exporter registry."""

from unittest.mock import MagicMock, patch

import pytest

from sr2.metrics.registry import (
    _reset_registry,
    get_exporter,
    list_exporters,
    register_exporter,
)
from sr2.metrics.exporter import PrometheusExporter


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before and after each test."""
    _reset_registry()
    register_exporter("prometheus", PrometheusExporter)
    yield
    _reset_registry()


class TestRegisterAndGetExporter:
    def test_get_builtin_prometheus(self):
        assert get_exporter("prometheus") is PrometheusExporter

    def test_register_custom_exporter(self):
        class CustomExporter:
            pass

        register_exporter("custom", CustomExporter)
        assert get_exporter("custom") is CustomExporter

    def test_get_nonexistent_raises_import_error(self):
        with pytest.raises(ImportError, match="Metric exporter 'nonexistent' is not available"):
            get_exporter("nonexistent")

    def test_error_message_includes_upgrade_hint(self):
        with pytest.raises(ImportError, match="pip install sr2-pro"):
            get_exporter("otel_custom")

    def test_list_exporters(self):
        exporters = list_exporters()
        assert "prometheus" in exporters


class TestEntryPointDiscovery:
    def test_entry_points_discovered_on_missing_exporter(self):
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "test_exporter"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_exporter("test_exporter")
            mock_ep.load.assert_called_once()

    def test_entry_points_discovered_only_once(self):
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "test"

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_exporter("missing1")
            with pytest.raises(ImportError):
                get_exporter("missing2")
            mock_ep.load.assert_called_once()

    def test_broken_entry_point_does_not_crash(self):
        _reset_registry()

        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = RuntimeError("plugin broken")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            with pytest.raises(ImportError):
                get_exporter("anything")

    def test_entry_point_registers_exporter(self):
        _reset_registry()

        class PluginExporter:
            pass

        def load_plugin():
            register_exporter("plugin", PluginExporter)

        mock_ep = MagicMock()
        mock_ep.name = "plugin"
        mock_ep.load.side_effect = load_plugin

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            result = get_exporter("plugin")
            assert result is PluginExporter
