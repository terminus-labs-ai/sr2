"""Degradation policy registry with entry-point discovery."""

from __future__ import annotations

from typing import Any, Protocol

from sr2.plugins.registry import PluginRegistry


class DegradationPolicy(Protocol):
    """Protocol for degradation policies."""

    def should_degrade(self, metrics: Any) -> bool: ...

    def should_recover(self, metrics: Any) -> bool: ...


_registry: PluginRegistry = PluginRegistry(
    "sr2.degradation_policies", install_hint="For SLA policies: pip install sr2-pro"
)


def register_policy(name: str, cls: type) -> None:
    """Register a degradation policy by name."""
    _registry.register(name, cls)


def get_policy(name: str) -> type:
    """Resolve a degradation policy by config name.

    Raises ImportError with upgrade hint if not found.
    """
    return _registry.get(name)


def list_policies() -> list[str]:
    """Return names of all registered degradation policies."""
    return _registry.list_available()


def _reset_registry() -> None:
    """Reset registry state. For testing only."""
    _registry._reset()
