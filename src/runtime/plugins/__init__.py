"""Interface plugins for the SR2 Runtime."""

from runtime.plugins.base import InterfacePlugin, TriggerContext
from runtime.plugins.registry import PluginRegistry, create_default_registry

__all__ = [
    "InterfacePlugin",
    "TriggerContext",
    "PluginRegistry",
    "create_default_registry",
]
