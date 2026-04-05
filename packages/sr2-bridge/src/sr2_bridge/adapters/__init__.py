"""Protocol adapters for different LLM API formats."""

from sr2_bridge.adapters.base import BridgeAdapter, ExecutionAdapter
from sr2_bridge.adapters.anthropic import AnthropicAdapter
from sr2_bridge.adapters.claude_code import ClaudeCodeAdapter
from sr2_bridge.adapters.claude_code_config import ClaudeCodeAdapterConfig

__all__ = [
    "AnthropicAdapter",
    "BridgeAdapter",
    "ClaudeCodeAdapter",
    "ClaudeCodeAdapterConfig",
    "ExecutionAdapter",
]
