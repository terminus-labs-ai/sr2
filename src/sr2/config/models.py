from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class TransformTriggers(str, Enum):
  TURN_START = "turn_start"
  TURN_END = "turn_end"
  OVERFLOW = "overflow"
  INTERVAL = "interval"


class TransformerConfig(BaseModel):
  type: str
  triggers: list[TransformTriggers]
  config: dict[str, Any] = {}


class ResolverConfig(BaseModel):
  type: str
  config: dict[str, Any] = {}


class LayerConfig(BaseModel):
  name: str
  cache: Literal["static", "ephemeral", "append_only"] | None = None
  token_budget: int | None = None
  resolvers: list[ResolverConfig]
  transformers: list[TransformerConfig] | None = None


class ContextConfig(BaseModel):
  layers: list[LayerConfig]


class PipelineConfig(BaseModel):
  context: ContextConfig
