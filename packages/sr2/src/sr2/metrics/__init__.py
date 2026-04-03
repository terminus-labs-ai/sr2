"""Metrics subsystem — collection, export, and alerting."""

from sr2.metrics.registry import register_exporter
from sr2.metrics.collector import MetricCollector as MetricCollector
from sr2.metrics.exporter import PrometheusExporter

# Register built-in metric exporters.
register_exporter("prometheus", PrometheusExporter)

# OTel exporter is registered only if opentelemetry is installed.
try:
    from sr2.metrics.otel_exporter import OTelExporter

    register_exporter("otel", OTelExporter)
except ImportError:
    pass
