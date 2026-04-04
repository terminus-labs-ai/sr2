"""Interface plugins for the SR2 Runtime."""

from sr2_runtime.plugins.base import InterfacePlugin, TriggerContext
from sr2_runtime.plugins.registry import PluginRegistry, create_default_registry

__all__ = [
    "InterfacePlugin",
    "TriggerContext",
    "PluginRegistry",
    "create_default_registry",
]
