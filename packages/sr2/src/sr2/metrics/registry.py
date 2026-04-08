"""Metric exporter registry with entry-point discovery."""

from __future__ import annotations

import warnings
from typing import Any, Protocol

from sr2.plugins.registry import PluginRegistry


class MetricExporter(Protocol):
    """Protocol for metric exporters (Prometheus, OTel, etc.).

    .. deprecated:: Use PushExporter or PullExporter from sr2.protocols instead.
    """

    def export(self, snapshot: Any) -> None: ...


# --- Push exporter registry (OTel-style: registers callbacks on collector) ---

_push_registry: PluginRegistry = PluginRegistry(
    "sr2.push_exporters", install_hint="For OTel support: pip install sr2-pro"
)


def register_push_exporter(name: str, cls: type) -> None:
    """Register a push-style metric exporter by name."""
    _push_registry.register(name, cls)


def get_push_exporter(name: str) -> type:
    """Resolve a push exporter by name."""
    return _push_registry.get(name)


def list_push_exporters() -> list[str]:
    """Return names of all registered push exporters."""
    return _push_registry.list_available()


# --- Pull exporter registry (Prometheus-style: export() -> str) ---

_pull_registry: PluginRegistry = PluginRegistry(
    "sr2.pull_exporters", install_hint="For Prometheus support: pip install sr2-pro"
)


def register_pull_exporter(name: str, cls: type) -> None:
    """Register a pull-style metric exporter by name."""
    _pull_registry.register(name, cls)


def get_pull_exporter(name: str) -> type:
    """Resolve a pull exporter by name."""
    return _pull_registry.get(name)


def list_pull_exporters() -> list[str]:
    """Return names of all registered pull exporters."""
    return _pull_registry.list_available()


# --- Deprecated unified registry (backward compat) ---

_legacy_registry: PluginRegistry = PluginRegistry("sr2.exporters")


def register_exporter(name: str, cls: type) -> None:
    """Register a metric exporter by name.

    .. deprecated:: Use register_push_exporter or register_pull_exporter instead.
    """
    warnings.warn(
        "register_exporter() is deprecated. Use register_push_exporter() "
        "or register_pull_exporter() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _legacy_registry.register(name, cls)


def get_exporter(name: str) -> type:
    """Resolve an exporter by config name.

    .. deprecated:: Use get_push_exporter or get_pull_exporter instead.

    Falls back to pull exporter registry, then push, then legacy.
    """
    # Check legacy registry first (direct access to avoid double-discovery)
    if name in _legacy_registry._registry:
        return _legacy_registry._registry[name]

    # Try pull registry (most common use case: Prometheus)
    try:
        return get_pull_exporter(name)
    except ImportError:
        pass

    # Try push registry
    try:
        return get_push_exporter(name)
    except ImportError:
        pass

    # Discover legacy entry points
    try:
        return _legacy_registry.get(name)
    except ImportError:
        pass

    available = sorted(
        set(_legacy_registry._registry.keys())
        | set(_push_registry._registry.keys())
        | set(_pull_registry._registry.keys())
    )
    raise ImportError(
        f"Metric exporter '{name}' is not available. "
        f"Installed exporters: {available}. "
        f"For OTel/Prometheus support: pip install sr2-pro"
    )


def list_exporters() -> list[str]:
    """Return names of all registered exporters (all registries)."""
    return sorted(
        set(_legacy_registry.list_available())
        | set(_push_registry.list_available())
        | set(_pull_registry.list_available())
    )


def _reset_registry() -> None:
    """Reset all registry state. For testing only."""
    _legacy_registry._reset()
    _push_registry._reset()
    _pull_registry._reset()
