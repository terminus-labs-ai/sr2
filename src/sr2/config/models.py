from typing import Annotated, Literal

from pydantic import BaseModel
from pydantic.functional_validators import PlainValidator


class ConfigError(Exception):
    """Raised when a pipeline configuration is invalid."""


class ToolLoopLimitError(Exception):
    """Raised when the tool iteration limit is exceeded within a single turn."""

# A dict field that Pydantic does not copy — preserves the original object's
# identity so that callers who mutate the dict after construction see the
# change at resolve/transform time (hot-reload, AC15).
_LiveDict = Annotated[dict, PlainValidator(lambda v: v)]


class EventSubscriptionConfig(BaseModel):
  event: str
  phase: str | None = None


class TransformerConfig(BaseModel):
  type: str
  subscriptions: list[EventSubscriptionConfig] = []
  config: _LiveDict = {}
  max_executions: int = 1


class ResolverConfig(BaseModel):
  type: str
  config: _LiveDict = {}
  subscriptions: list[EventSubscriptionConfig] = []
  max_executions: int = 1


class ToolProviderConfig(BaseModel):
  type: str
  config: _LiveDict = {}
  subscriptions: list[EventSubscriptionConfig] = []
  max_executions: int = 1


class LayerConfig(BaseModel):
  name: str
  cache: Literal["static", "ephemeral", "append_only"] | None = None
  token_budget: int | None = None
  token_threshold_pct: float | None = None
  resolvers: list[ResolverConfig]
  transformers: list[TransformerConfig] | None = None
  tool_providers: list[ToolProviderConfig] | None = None
  target: str
  position: str = "append"


class PipelineConfig(BaseModel):
  layers: list[LayerConfig]
  token_budget: int = 200_000
  enable_overflow_detection: bool = True
  max_tool_iterations: int = 25
  max_parallel_tools: int | None = None
