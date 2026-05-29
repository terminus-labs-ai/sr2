"""Data models for the SR2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from sr2.models import ContentBlock, Message
from sr2.pipeline.events import Event
from sr2.pipeline.provenance import Entry


# ---------------------------------------------------------------------------
# Compilation target
# ---------------------------------------------------------------------------


class CompilationTarget(Enum):
    """Where a resolved layer's content should be placed in the final LLM call."""

    SYSTEM = "system"
    MESSAGES = "messages"
    TOOLS = "tools"


# ---------------------------------------------------------------------------
# Handler output
# ---------------------------------------------------------------------------


@dataclass
class ResolvedContent:
  """Output of a Resolver: content blocks with resolver identity and token count."""

  resolver_name: str
  source_layer: str
  content: list[ContentBlock | Message] = field(default_factory=list)
  token_count: int = 0
  entries: list[Entry] = field(default_factory=list)
  events: list[Event] = field(default_factory=list)


@dataclass
class TransformationResult:
  """Output of a Transformer: transformed content with optional emitted events."""

  transformer_name: str
  source_layer: str
  content: list[ContentBlock] | None = None
  events: list[Event] | None = None
  tokens_before: int = 0
  tokens_after: int = 0
  tokens_saved: int = 0
  entries: list[Entry] = field(default_factory=list)


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
  bus_errors: List[str] = field(default_factory=list)
  """Handler exceptions and drain-abort events collected by EventBus.

  Each entry is a human-readable string describing the error.  The bus does
  not re-raise these; this field is the only way callers can inspect them.
  """


@dataclass
class PipelineResult:
  """Final output of a pipeline run: a CompletionRequest plus metrics."""

  request: Any = None
  metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
