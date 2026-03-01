"""Metric exporter — Prometheus text exposition format."""

from sr2.metrics.collector import MetricCollector
from sr2.metrics.definitions import MetricNames


class PrometheusExporter:
    """Exports metrics in Prometheus text exposition format.

    Designed to be called from a /metrics endpoint.
    """

    def __init__(self, collector: MetricCollector):
        self._collector = collector

    def export(self) -> str:
        """Export current metric state as Prometheus text format.

        For each metric, exports the latest value as a gauge.
        For rate metrics, computes from recent history.
        """
        lines = []

        # Gauge metrics from latest snapshot
        latest = self._collector.get_latest(1)
        if latest:
            snapshot = latest[0]
            seen: set[str] = set()
            for m in snapshot.metrics:
                metric_key = self._make_key(m.name, m.labels)
                if metric_key in seen:
                    continue
                seen.add(metric_key)

                # HELP and TYPE lines
                lines.append(f"# HELP {m.name} Pipeline metric")
                lines.append(f"# TYPE {m.name} gauge")

                label_str = self._format_labels(m.labels)
                lines.append(f"{m.name}{label_str} {m.value}")

        # Computed averages over recent history
        avg_metrics = [
            MetricNames.CACHE_HIT_RATE,
            MetricNames.PIPELINE_TOTAL_TOKENS,
            MetricNames.PIPELINE_TOTAL_DURATION_MS,
        ]
        for name in avg_metrics:
            avg = self._collector.get_average(name, 100)
            if avg is not None:
                avg_name = f"{name}_avg100"
                lines.append(
                    f"# HELP {avg_name} Average over last 100 invocations"
                )
                lines.append(f"# TYPE {avg_name} gauge")
                lines.append(f"{avg_name} {avg:.4f}")

        return "\n".join(lines) + "\n"

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels as Prometheus label string."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def _make_key(self, name: str, labels: dict) -> str:
        """Make a unique key for deduplication."""
        return f"{name}:{sorted(labels.items())}"


def create_metrics_endpoint(collector: MetricCollector):
    """Create a FastAPI route function for /metrics."""
    exporter = PrometheusExporter(collector)

    async def metrics_endpoint():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(
            content=exporter.export(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return metrics_endpoint
