"""Memory store backend registry with entry-point discovery."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sr2.memory.store import MemoryStore

logger = logging.getLogger(__name__)

_STORE_REGISTRY: dict[str, type[MemoryStore]] = {}
_discovered = False


def register_store(name: str, cls: type[MemoryStore]) -> None:
    """Register a memory store backend by name."""
    _STORE_REGISTRY[name] = cls


def get_store(name: str) -> type[MemoryStore]:
    """Resolve a store by config name.

    Raises ImportError with upgrade hint if not found.
    """
    if name not in _STORE_REGISTRY:
        _discover_entry_points()
    if name not in _STORE_REGISTRY:
        available = sorted(_STORE_REGISTRY.keys())
        raise ImportError(
            f"Memory store '{name}' is not available. "
            f"Installed backends: {available}. "
            f"For PostgreSQL support: pip install sr2-pro"
        )
    return _STORE_REGISTRY[name]


def list_stores() -> list[str]:
    """Return names of all registered store backends."""
    _discover_entry_points()
    return sorted(_STORE_REGISTRY.keys())


def _discover_entry_points() -> None:
    """Lazy-discover plugins registered via entry points."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    from importlib.metadata import entry_points

    for ep in entry_points(group="sr2.stores"):
        try:
            ep.load()()
        except Exception:
            logger.warning("Failed to load store plugin: %s", ep.name, exc_info=True)

    # Fallback: try direct import if entry points aren't available
    # (uv workspace editable installs don't register entry points)
    if "postgres" not in _STORE_REGISTRY:
        try:
            from sr2_pro.memory import register_stores

            register_stores()
        except ImportError:
            pass
        except Exception:
            logger.warning("Failed to load sr2-pro postgres store via fallback", exc_info=True)


def _reset_registry() -> None:
    """Reset registry state. For testing only."""
    global _discovered
    _STORE_REGISTRY.clear()
    _discovered = False
