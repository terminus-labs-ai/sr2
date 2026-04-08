"""Generic plugin registry with lazy entry-point discovery."""

from __future__ import annotations

import logging
from typing import Generic, TypeVar

from sr2.plugins.errors import PluginLicenseError, PluginNotFoundError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Generic plugin registry with lazy entry-point discovery.

    Replaces the boilerplate pattern used in memory, metrics, and
    degradation registries. Each instance manages one entry-point group
    (e.g. ``sr2.stores``, ``sr2.push_exporters``).

    Usage::

        _registry = PluginRegistry[MemoryStore]("sr2.stores", install_hint="pip install sr2-pro")
        _registry.register("memory", InMemoryMemoryStore)
        store_cls = _registry.get("memory")
    """

    def __init__(self, group: str, *, install_hint: str = "") -> None:
        self._group = group
        self._install_hint = install_hint
        self._registry: dict[str, type[T]] = {}
        self._license_errors: dict[str, Exception] = {}
        self._discovered = False

    def register(self, name: str, cls: type[T]) -> None:
        """Register a plugin by name."""
        self._registry[name] = cls

    def get(self, name: str) -> type[T]:
        """Resolve a plugin by name.

        Triggers lazy entry-point discovery on first miss.

        Raises:
            PluginNotFoundError: Plugin not registered or discoverable.
            PluginLicenseError: Plugin found but blocked by license validation.
        """
        if name in self._registry:
            return self._registry[name]

        if not self._discovered:
            self._discover()

        if name in self._registry:
            return self._registry[name]

        if name in self._license_errors:
            raise PluginLicenseError(
                f"Plugin '{name}' requires a valid license. "
                f"{self._install_hint}".strip()
            ) from self._license_errors[name]

        available = sorted(self._registry.keys())
        parts = [
            f"Plugin '{name}' is not available (group: {self._group}).",
            f"Installed plugins: {available}.",
        ]
        if self._install_hint:
            parts.append(self._install_hint)
        raise PluginNotFoundError(" ".join(parts))

    def list_available(self) -> list[str]:
        """Return names of all registered plugins."""
        if not self._discovered:
            self._discover()
        return sorted(self._registry.keys())

    def _discover(self) -> None:
        """Lazy-discover plugins registered via entry points."""
        self._discovered = True
        from importlib.metadata import entry_points

        for ep in entry_points(group=self._group):
            try:
                result = ep.load()
                # Entry points either return a callable that registers,
                # or are the class itself. Convention: callable that registers.
                if callable(result):
                    result()
            except PluginLicenseError as exc:
                self._license_errors[ep.name] = exc
                logger.debug("Plugin %s blocked by license: %s", ep.name, exc)
            except Exception:
                logger.warning(
                    "Failed to load plugin: %s (group: %s)",
                    ep.name,
                    self._group,
                    exc_info=True,
                )

    def _reset(self) -> None:
        """Reset registry state. For testing only."""
        self._registry.clear()
        self._license_errors.clear()
        self._discovered = False
