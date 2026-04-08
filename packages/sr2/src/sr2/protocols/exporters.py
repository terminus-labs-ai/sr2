"""Protocols for metric exporters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2.metrics.collector import MetricCollector


@runtime_checkable
class PushExporter(Protocol):
    """Registers on MetricCollector, fires on every collect cycle (e.g., OTel)."""

    def register(self, collector: MetricCollector) -> None: ...


@runtime_checkable
class PullExporter(Protocol):
    """Returns metrics on demand (e.g., Prometheus /metrics endpoint)."""

    def export(self) -> str: ...
