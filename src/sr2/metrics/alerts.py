"""Alert rule engine — check metrics against thresholds, generate alerts."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sr2.metrics.definitions import (
    DEFAULT_THRESHOLDS,
    MetricSnapshot,
    MetricThreshold,
)


@dataclass
class Alert:
    """A generated alert."""

    metric_name: str
    actual_value: float
    threshold_value: float
    condition: str
    severity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)
    message: str = ""


class AlertRuleEngine:
    """Checks metrics against thresholds and generates alerts."""

    def __init__(
        self,
        thresholds: list[MetricThreshold] | None = None,
        alert_callback=None,
    ):
        """
        Args:
            thresholds: list of alert thresholds. Defaults to DEFAULT_THRESHOLDS.
            alert_callback: optional async function(alert: Alert) -> None
                           Called when an alert fires.
        """
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._callback = alert_callback
        self._alert_history: list[Alert] = []
        self._suppressed: dict[str, datetime] = {}  # metric_name -> last alert time
        self._suppression_seconds: float = 300.0  # Don't re-alert within 5 minutes

    async def check(self, snapshot: MetricSnapshot) -> list[Alert]:
        """Check a snapshot against all thresholds. Returns triggered alerts."""
        alerts = []
        for threshold in self._thresholds:
            metric = snapshot.get(threshold.metric_name)
            if metric is None:
                continue

            if not threshold.is_triggered(metric.value):
                continue

            # Suppression check
            if self._is_suppressed(threshold.metric_name):
                continue

            alert = Alert(
                metric_name=threshold.metric_name,
                actual_value=metric.value,
                threshold_value=threshold.value,
                condition=threshold.condition,
                severity=threshold.severity,
                labels=metric.labels,
                message=(
                    f"{threshold.metric_name} {threshold.condition} {threshold.value}: "
                    f"actual={metric.value:.4f}"
                ),
            )

            alerts.append(alert)
            self._alert_history.append(alert)
            self._suppressed[threshold.metric_name] = datetime.now(UTC)

            if self._callback:
                await self._callback(alert)

        return alerts

    def _is_suppressed(self, metric_name: str) -> bool:
        """Check if alerts for this metric are suppressed."""
        if metric_name not in self._suppressed:
            return False
        elapsed = (datetime.now(UTC) - self._suppressed[metric_name]).total_seconds()
        return elapsed < self._suppression_seconds

    @property
    def alert_history(self) -> list[Alert]:
        return list(self._alert_history)

    def clear_history(self) -> None:
        self._alert_history.clear()

    def clear_suppressions(self) -> None:
        self._suppressed.clear()
