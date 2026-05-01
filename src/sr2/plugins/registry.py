"""Plugin registry with lazy entry-point discovery and license gating.

SR2 never imports plugin code unless the YAML config references it.
Discovery is lazy — plugins are loaded on first use, not at startup.

Design principles:
- OCP: New plugins add via entry points, never modify registry code.
- SRP: Registry only handles discovery and activation — not plugin logic.
- DRY: Generic typing means one registry serves all extension points.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Generic, TypeVar

from sr2.core.errors import PluginLicenseError, PluginNotFoundError

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """Lazy-discovered plugin registry.

    Plugins register via Python packaging entry points. The registry
    discovers them on first reference, validates licenses for paid
    plugins, and caches the resolved instances.

    Usage:
        store_registry = PluginRegistry["MemoryStore"]("sr2.stores")
        store = store_registry.get("postgres")  # loads lazily
    """

    def __init__(self, entry_point_group: str) -> None:
        self.entry_point_group = entry_point_group
        self._cache: dict[str, T] = {}

    def get(self, name: str) -> T:
        """Get a plugin by name, loading it lazily if not cached.

        Args:
            name: Plugin identifier (e.g., "sqlite", "postgres").

        Returns:
            The plugin instance implementing the protocol.

        Raises:
            PluginNotFoundError: If no entry point matches the name.
            PluginLicenseError: If a paid plugin lacks a valid license.
        """
        if name in self._cache:
            return self._cache[name]

        plugin = self._load(name)
        self._cache[name] = plugin
        return plugin

    def _load(self, name: str) -> T:
        """Load and instantiate a plugin from its entry point."""
        entry_points = list(importlib.metadata.entry_points(group=self.entry_point_group, name=name))

        if not entry_points:
            raise PluginNotFoundError(
                f'Plugin "{name}" not found in group "{self.entry_point_group}"'
            )

        entry_point = entry_points[0]
        factory = entry_point.load()

        # If the entry point is a class, instantiate it; otherwise call as factory
        if isinstance(factory, type):
            instance = factory()
        else:
            instance = factory()

        # License validation happens inside the plugin's register/init
        # via require_license() — core never knows about licensing details
        return instance  # type: ignore[return-value]

    def list_available(self) -> list[str]:
        """List all discovered plugin names for this group."""
        entry_points = list(importlib.metadata.entry_points(group=self.entry_point_group))
        return [ep.name for ep in entry_points]

    def clear_cache(self) -> None:
        """Clear the instance cache. Useful for testing."""
        self._cache.clear()


def require_license(key: str) -> None:
    """Validate a plugin license key.

    Paid plugins call this during initialization. Core provides the
    validation mechanism but never imports paid code directly.

    Args:
        key: The HMAC-SHA256 license key.

    Raises:
        PluginLicenseError: If the key is invalid or expired.
    """
    # TODO: Implement HMAC-SHA256 license validation
    # For now, any non-empty key passes
    if not key or len(key) < 8:
        raise PluginLicenseError(
            "Invalid license key. "
            "Purchase a license at https://sr2.dev or use a free plugin."
        )
