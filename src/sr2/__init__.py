"""SR2 v2 — Context Engineering Library for AI Agents.

Manages the full lifecycle of what goes into an LLM's context window:
compaction, caching, summarization, graceful degradation, and KV-cache optimization.

Public API:
    from sr2 import SR2
    from sr2.core.models import TurnResult, TokenUsage, ToolCall
    from sr2.config.models import PipelineConfig, LayerConfig
    from sr2.pipeline.result import CompiledContext, PostProcessResult

Everything else is internal. The harness should only import from this module.
"""

from sr2.core.models import (
    Memory,
    MemoryScope,
    MemoryStatus,
    MemoryType,
    ToolCall,
    TokenUsage,
    TurnResult,
)
from sr2.pipeline.result import (
    CompiledContext,
    PostProcessResult,
    PipelineMetrics,
)
from sr2.config.models import (
    LayerConfig,
    PipelineConfig,
)
from sr2.sr2 import SR2

__version__ = "2.0.0-alpha.0"

__all__ = [
    # Facade
    "SR2",
    # Core models
    "TurnResult",
    "ToolCall",
    "TokenUsage",
    "Memory",
    "MemoryScope",
    "MemoryStatus",
    "MemoryType",
    # Config
    "PipelineConfig",
    "LayerConfig",
    # Results
    "CompiledContext",
    "PostProcessResult",
    "PipelineMetrics",
]
