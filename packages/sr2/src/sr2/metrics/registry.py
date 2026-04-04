"""Metric exporter registry with entry-point discovery."""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class MetricExporter(Protocol):
    """Protocol for metric exporters (Prometheus, OTel, etc.)."""

    def export(self, snapshot: Any) -> None: ...


_EXPORTER_REGISTRY: dict[str, type] = {}
_discovered = False


def register_exporter(name: str, cls: type) -> None:
    """Register a metric exporter by name."""
    _EXPORTER_REGISTRY[name] = cls


def get_exporter(name: str) -> type:
    """Resolve an exporter by config name.

    Raises ImportError with upgrade hint if not found.
    """
    if name not in _EXPORTER_REGISTRY:
        _discover_entry_points()
    if name not in _EXPORTER_REGISTRY:
        available = sorted(_EXPORTER_REGISTRY.keys())
        raise ImportError(
            f"Metric exporter '{name}' is not available. "
            f"Installed exporters: {available}. "
            f"For OTel/Prometheus support: pip install sr2-pro"
        )
    return _EXPORTER_REGISTRY[name]


def list_exporters() -> list[str]:
    """Return names of all registered exporters."""
    _discover_entry_points()
    return sorted(_EXPORTER_REGISTRY.keys())


def _discover_entry_points() -> None:
    """Lazy-discover plugins registered via entry points."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    from importlib.metadata import entry_points

    for ep in entry_points(group="sr2.exporters"):
        try:
            ep.load()()
        except Exception:
            logger.warning("Failed to load exporter plugin: %s", ep.name, exc_info=True)


def _reset_registry() -> None:
    """Reset registry state. For testing only."""
    global _discovered
    _EXPORTER_REGISTRY.clear()
    _discovered = False
