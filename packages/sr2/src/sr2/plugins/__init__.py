"""SR2 plugin system."""

from sr2.plugins.errors import PluginLicenseError, PluginNotFoundError
from sr2.plugins.registry import PluginRegistry

__all__ = [
    "PluginLicenseError",
    "PluginNotFoundError",
    "PluginRegistry",
]
