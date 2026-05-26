"""SR2 plugin system — registry and error hierarchy."""

from sr2.plugins.errors import PluginCollisionError, PluginError, PluginNotFoundError
from sr2.plugins.registry import PluginRegistry

__all__ = [
    "PluginRegistry",
    "PluginError",
    "PluginNotFoundError",
    "PluginCollisionError",
]
