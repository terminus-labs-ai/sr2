"""Protocol for alert engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2.metrics.definitions import MetricSnapshot


@dataclass
class Alert:
    """A triggered alert from metric evaluation."""

    metric_name: str
    actual_value: float
    threshold_value: float
    condition: str
    severity: str
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    message: str = ""


@runtime_checkable
class AlertEngine(Protocol):
    """Evaluates metric snapshots against configured rules."""

    async def evaluate(self, snapshot: MetricSnapshot) -> list[Alert]: ...

    def configure(self, rules: list[dict]) -> None: ...
