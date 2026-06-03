from typing import Annotated, Literal

from pydantic import BaseModel, model_validator
from pydantic.functional_validators import PlainValidator


class ConfigError(Exception):
    """Raised when a pipeline configuration is invalid."""


class ToolLoopLimitError(Exception):
    """Raised when the tool iteration limit is exceeded within a single turn."""

# A dict field that Pydantic does not copy — preserves the original object's
# identity so that callers who mutate the dict after construction see the
# change at resolve/transform time (hot-reload, AC15).
_LiveDict = Annotated[dict, PlainValidator(lambda v: v)]

# ---------------------------------------------------------------------------
# Degradation config (FR1 — sr2-81)
# ---------------------------------------------------------------------------


class DegradationLevelConfig(BaseModel):
    """One step on the degradation ladder: a name and the categories it keeps."""

    name: str
    keep_categories: list[str]


class DegradationTriggerConfig(BaseModel):
    """A trigger that can fire degradation step-downs.

    v1 trigger set: ``overflow`` (budget pressure, pre-LLM) and
    ``context_limit`` (LLM rejected the request as too long).
    """

    type: Literal["overflow", "context_limit"]
    threshold: int | float | None = None


class DegradationConfig(BaseModel):
    """Full degradation configuration for a pipeline.

    When absent (``None`` on ``PipelineConfig.degradation``), degradation is
    fully disabled and the pipeline runs identically to the pre-change baseline.
    """

    levels: list[DegradationLevelConfig]
    triggers: list[DegradationTriggerConfig] = []

    @model_validator(mode="after")
    def _validate_levels_not_empty(self) -> "DegradationConfig":
        if not self.levels:
            raise ConfigError("degradation requires at least one level")
        return self

    @model_validator(mode="after")
    def _validate_unique_level_names(self) -> "DegradationConfig":
        seen: set[str] = set()
        for level in self.levels:
            if level.name in seen:
                raise ConfigError(
                    f"duplicate degradation level name: {level.name!r}"
                )
            seen.add(level.name)
        return self


class EventSubscriptionConfig(BaseModel):
  event: str
  phase: str | None = None


class TransformerConfig(BaseModel):
  type: str
  name: str | None = None
  subscriptions: list[EventSubscriptionConfig] = []
  config: _LiveDict = {}
  max_executions: int = 1


class ResolverConfig(BaseModel):
  type: str
  name: str | None = None
  config: _LiveDict = {}
  subscriptions: list[EventSubscriptionConfig] = []
  max_executions: int = 1


class ToolProviderConfig(BaseModel):
  type: str
  name: str | None = None
  config: _LiveDict = {}
  subscriptions: list[EventSubscriptionConfig] = []
  max_executions: int = 1


class LayerConfig(BaseModel):
  name: str
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
  max_tool_iterations: int = 25
  max_parallel_tools: int | None = None
  llm_timeout_seconds: float | None = None
  circuit_breaker_failure_threshold: int = 3
  circuit_breaker_recovery_timeout: float = 60.0
  degradation: DegradationConfig | None = None
