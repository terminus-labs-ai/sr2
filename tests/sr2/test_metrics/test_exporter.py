"""Tests for Prometheus metric exporter."""

from sr2.metrics.collector import MetricCollector
from sr2.metrics.definitions import MetricNames
from sr2.metrics.exporter import PrometheusExporter, create_metrics_endpoint
from sr2.pipeline.result import PipelineResult


def _collector_with_snapshot() -> MetricCollector:
    collector = MetricCollector(agent_name="test-agent")
    result = PipelineResult(
        total_tokens=150,
        total_duration_ms=42.5,
        cache_hit_rate=0.75,
    )
    collector.collect(result, "user_message")
    return collector


class TestPrometheusExporter:

    def test_export_no_snapshots(self):
        """Export with no snapshots -> minimal output."""
        collector = MetricCollector(agent_name="agent")
        exporter = PrometheusExporter(collector)
        output = exporter.export()

        # No gauge metrics, no averages (all None)
        assert output.strip() == ""

    def test_export_with_snapshot(self):
        """Export with one snapshot -> includes metrics."""
        collector = _collector_with_snapshot()
        exporter = PrometheusExporter(collector)
        output = exporter.export()

        assert MetricNames.PIPELINE_TOTAL_TOKENS in output
        assert "150" in output

    def test_labels_formatted_correctly(self):
        """Labels formatted correctly: {agent="test",interface="user_message"}."""
        collector = _collector_with_snapshot()
        exporter = PrometheusExporter(collector)
        output = exporter.export()

        assert 'agent="test-agent"' in output
        assert 'interface="user_message"' in output

    def test_averages_computed(self):
        """Averages computed over recent history."""
        collector = MetricCollector(agent_name="agent")
        for tokens in [100, 200, 300]:
            result = PipelineResult(total_tokens=tokens, total_duration_ms=10.0)
            collector.collect(result, "api")

        exporter = PrometheusExporter(collector)
        output = exporter.export()

        assert f"{MetricNames.PIPELINE_TOTAL_TOKENS}_avg100" in output

    def test_format_labels_empty(self):
        """_format_labels() with empty labels -> empty string."""
        collector = MetricCollector(agent_name="agent")
        exporter = PrometheusExporter(collector)
        assert exporter._format_labels({}) == ""

    def test_help_and_type_lines(self):
        """Each metric has # HELP and # TYPE lines."""
        collector = _collector_with_snapshot()
        exporter = PrometheusExporter(collector)
        output = exporter.export()

        assert "# HELP" in output
        assert "# TYPE" in output

    def test_create_metrics_endpoint_returns_callable(self):
        """create_metrics_endpoint() returns a callable."""
        collector = MetricCollector(agent_name="agent")
        endpoint = create_metrics_endpoint(collector)
        assert callable(endpoint)
