"""SR2 Runtime — config-driven single-agent runtime."""

from sr2.runtime.agent import SR2Runtime
from sr2.runtime.config import AgentConfig
from sr2.runtime.result import RuntimeResult, RuntimeMetrics

__all__ = ["SR2Runtime", "AgentConfig", "RuntimeResult", "RuntimeMetrics"]
