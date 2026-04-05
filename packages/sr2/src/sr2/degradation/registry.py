"""Degradation policy registry with entry-point discovery."""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class DegradationPolicy(Protocol):
    """Protocol for degradation policies."""

    def should_degrade(self, metrics: Any) -> bool: ...

    def should_recover(self, metrics: Any) -> bool: ...


_POLICY_REGISTRY: dict[str, type] = {}
_discovered = False


def register_policy(name: str, cls: type) -> None:
    """Register a degradation policy by name."""
    _POLICY_REGISTRY[name] = cls


def get_policy(name: str) -> type:
    """Resolve a degradation policy by config name.

    Raises ImportError with upgrade hint if not found.
    """
    if name not in _POLICY_REGISTRY:
        _discover_entry_points()
    if name not in _POLICY_REGISTRY:
        available = sorted(_POLICY_REGISTRY.keys())
        raise ImportError(
            f"Degradation policy '{name}' is not available. "
            f"Installed policies: {available}. "
            f"For SLA policies: pip install sr2-pro"
        )
    return _POLICY_REGISTRY[name]


def list_policies() -> list[str]:
    """Return names of all registered degradation policies."""
    _discover_entry_points()
    return sorted(_POLICY_REGISTRY.keys())


def _discover_entry_points() -> None:
    """Lazy-discover plugins registered via entry points."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    from importlib.metadata import entry_points

    for ep in entry_points(group="sr2.degradation_policies"):
        try:
            ep.load()()
        except Exception:
            logger.warning("Failed to load degradation policy plugin: %s", ep.name, exc_info=True)


def _reset_registry() -> None:
    """Reset registry state. For testing only."""
    global _discovered
    _POLICY_REGISTRY.clear()
    _discovered = False
