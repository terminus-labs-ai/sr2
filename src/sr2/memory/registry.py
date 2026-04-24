"""Memory store backend registry with entry-point discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sr2.plugins.registry import PluginRegistry

if TYPE_CHECKING:
    from sr2.memory.store import MemoryStore

_registry: PluginRegistry[MemoryStore] = PluginRegistry(
    "sr2.stores", install_hint="For PostgreSQL support: pip install sr2-pro"
)


def register_store(name: str, cls: type[MemoryStore]) -> None:
    """Register a memory store backend by name."""
    _registry.register(name, cls)


def get_store(name: str) -> type[MemoryStore]:
    """Resolve a store by config name.

    Raises ImportError with upgrade hint if not found.
    """
    return _registry.get(name)


def list_stores() -> list[str]:
    """Return names of all registered store backends."""
    return _registry.list_available()


def _reset_registry() -> None:
    """Reset registry state. For testing only."""
    _registry._reset()
