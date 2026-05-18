"""Data models for the SR2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from sr2.models import ContentBlock, Message
from sr2.pipeline.events import Event


# ---------------------------------------------------------------------------
# Compilation target
# ---------------------------------------------------------------------------


class CompilationTarget(Enum):
    """Where a resolved layer's content should be placed in the final LLM call."""

    SYSTEM = "system"
    MESSAGES = "messages"
    TOOLS = "tools"


def infer_compilation_target(
    layer_name: str,
    explicit_target: Optional[str] = None,
) -> CompilationTarget:
    """Infer the compilation target from a layer name, with optional explicit override.

    Rules:
      1. If explicit_target is not None, look up the corresponding CompilationTarget.
      2. Else if the layer name contains "system" -> SYSTEM.
      3. Else if the layer name contains "tool" -> TOOLS.
      4. Else -> MESSAGES.
    """
    if explicit_target is not None:
        return CompilationTarget(explicit_target)

    name = layer_name.lower()
    if "system" in name:
        return CompilationTarget.SYSTEM
    if "tool" in name:
        return CompilationTarget.TOOLS
    return CompilationTarget.MESSAGES

# ---------------------------------------------------------------------------
# Handler output
# ---------------------------------------------------------------------------


@dataclass
class ResolvedContent:
  """Output of a Resolver: content blocks with resolver identity and token count."""

  resolver_name: str
  source_layer: str
  content: list[ContentBlock | Message]
  token_count: int = 0


@dataclass
class TransformationResult:
  """Output of a Transformer: transformed content with optional emitted events."""

  transformer_name: str
  source_layer: str
  content: list[ContentBlock]
  events: list[Event] | None = None
  tokens_before: int = 0
  tokens_after: int = 0
  tokens_saved: int = 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class LayerMetrics:
  """Per-layer metrics collected after a pipeline run."""

  tokens_used: int = 0
  token_budget: Optional[int] = None
  budget_remaining: Optional[int] = None
  force_truncated: bool = False
  resolver_executions: Dict[str, int] = field(default_factory=dict)
  transformer_executions: Dict[str, int] = field(default_factory=dict)


@dataclass
class PipelineMetrics:
  """Aggregate metrics for a complete pipeline run."""

  layers: Dict[str, LayerMetrics] = field(default_factory=dict)
  total_tokens: int = 0
  warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
  """Final output of a pipeline run: a CompletionRequest plus metrics."""

  request: Any = None
  metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
