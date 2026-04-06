"""Protocol adapters for different LLM API formats."""

from __future__ import annotations

from typing import Any

from sr2_bridge.adapters.base import BridgeAdapter, ExecutionAdapter
from sr2_bridge.adapters.anthropic import AnthropicAdapter

# Lazy registry — avoids importing heavy modules at package load time.
_ADAPTER_FACTORIES: dict[str, str] = {
    "claude_code": "sr2_bridge.adapters.claude_code.ClaudeCodeAdapter",
}

_CONFIG_FACTORIES: dict[str, str] = {
    "claude_code": "sr2_bridge.adapters.claude_code_config.ClaudeCodeAdapterConfig",
}


def get_execution_adapter(name: str, config: dict[str, Any]) -> ExecutionAdapter:
    """Instantiate an execution adapter by name.

    Args:
        name: Adapter name (e.g. ``"claude_code"``).
        config: Raw config dict passed to the adapter's config model.

    Returns:
        An initialized :class:`ExecutionAdapter`.

    Raises:
        ValueError: If the adapter name is not registered.
        ImportError: If the adapter module cannot be imported.
    """
    if name not in _ADAPTER_FACTORIES:
        available = sorted(_ADAPTER_FACTORIES)
        raise ValueError(f"Unknown execution adapter '{name}'. Available: {available}")

    import importlib

    # Resolve config class
    config_path = _CONFIG_FACTORIES[name]
    config_mod, config_cls_name = config_path.rsplit(".", 1)
    config_cls = getattr(importlib.import_module(config_mod), config_cls_name)

    # Resolve adapter class
    adapter_path = _ADAPTER_FACTORIES[name]
    adapter_mod, adapter_cls_name = adapter_path.rsplit(".", 1)
    adapter_cls = getattr(importlib.import_module(adapter_mod), adapter_cls_name)

    return adapter_cls(config_cls(**config))


__all__ = [
    "AnthropicAdapter",
    "BridgeAdapter",
    "ExecutionAdapter",
    "get_execution_adapter",
]
