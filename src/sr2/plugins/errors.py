"""Plugin error hierarchy for SR2 plugin registry."""

from __future__ import annotations


class PluginError(ImportError):
    """Base class for all plugin-related errors."""


class PluginNotFoundError(PluginError):
    """Raised when a plugin name is not found in the registry.

    The error message includes the requested name and the list of available names.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        available_str = ", ".join(sorted(available)) if available else "(none)"
        super().__init__(
            f"Plugin {name!r} not found in registry. "
            f"Available plugins: [{available_str}]"
        )
        self.name = name
        self.available = available


class PluginCollisionError(PluginError):
    """Raised when two entry points register the same plugin name.

    The error message includes the conflicting name and both distribution names.
    """

    def __init__(self, name: str, dist_names: list[str]) -> None:
        dists_str = ", ".join(repr(d) for d in dist_names)
        super().__init__(
            f"Plugin name {name!r} is registered by multiple distributions: "
            f"{dists_str}. Remove or rename the conflicting entry point."
        )
        self.name = name
        self.dist_names = dist_names
