from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator
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
  degradation_category: str | None = None
  priority: int = 0
  condition: str | None = None
  """Run-context key required for this layer to participate.

  When set, the layer is active only when the current run context contains a
  non-empty string for this key. A ``None`` condition leaves the layer active.
  """


# ---------------------------------------------------------------------------
# Forced final answer config (sr2-121 — role-run stop rules, spec spc-959)
# ---------------------------------------------------------------------------

# Default message texts. Every text the forced final answer uses is a YAML
# field with one of these defaults (spec FR 22); none is hardcoded at the
# call site. ``instruction``, ``fallback`` and the reason text interpolate
# with str.format placeholders.

DEFAULT_FINAL_ANSWER_INSTRUCTION = (
  "The harness is ending this run: {reason}. Tools are no longer available. "
  "Write your final answer now in your contract's format: your result if you "
  "have one, otherwise an escalation that says what blocked you and what you tried."
)

DEFAULT_FINAL_ANSWER_FALLBACK = (
  "Run stopped by the harness: {reason}. The model produced no final answer."
)

DEFAULT_FINAL_ANSWER_NOT_RUN = "Not run: the run is ending."


# ---------------------------------------------------------------------------
# Repeat guard config (sr2-122 — role-run stop rules, spec spc-959)
# ---------------------------------------------------------------------------

# Default message texts (spec spc-959 § Interfaces). ``note`` interpolates
# ``{first_call}`` and ``{repeat}``; ``warning`` interpolates ``{repeat}``.

DEFAULT_REPEAT_NOTE = (
  "Result not shown: this call is identical to call {first_call} and "
  "returned the same result. Do not repeat it. Change your approach, or if "
  "you cannot make progress, stop and escalate in your contract's format."
)

DEFAULT_REPEAT_WARNING = (
  " This is repeat {repeat}; the next identical call ends the run with an "
  "escalation."
)


class FinalAnswerConfig(BaseModel):
  """Forced final answer at the tool-iteration limit (spec spc-959 FR 1-6).

  Off by default: a pipeline with no ``final_answer`` block ends its tool
  loop with ``ToolLoopLimitError`` exactly as before. When enabled, an
  iteration that brings tool calls at ``max_tool_iterations`` runs none of
  them and makes one final tool-less LLM call instead of raising.
  """

  enabled: bool = False
  instruction: str = DEFAULT_FINAL_ANSWER_INSTRUCTION
  """Appended as a user message before the forced call. ``{reason}`` filled."""

  fallback: str = DEFAULT_FINAL_ANSWER_FALLBACK
  """Final text when the forced reply is empty, whitespace-only or has a
  tool call. ``{reason}`` filled. There is no retry."""

  not_run: str = DEFAULT_FINAL_ANSWER_NOT_RUN
  """Result given to every tool call of the unexecuted iteration."""


class RepeatGuardConfig(BaseModel):
  """Consecutive repeat guard in the tool loop (spec spc-959 FR 7-18).

  Off by default: a pipeline with no ``repeat_guard`` block never has its
  tool results altered. When enabled, every executed call still runs; an
  identical call (same tool, same sorted-key JSON arguments) with the same
  result gets ``note`` instead of its real result, ``warning`` is appended
  on repeat ``notes_before_stop``, and the next same-result repeat — or
  ``max_identical_calls`` consecutive identical calls whatever the results —
  ends the run through the forced final answer with reason ``repeat``.
  """

  enabled: bool = False
  notes_before_stop: int = Field(2, ge=1)
  """Same-result repeats that get the note before the run stops."""

  max_identical_calls: int = Field(10, ge=2)
  """Consecutive identical calls that stop the run whatever the results."""

  note: str = DEFAULT_REPEAT_NOTE
  """Replaces the result of a same-result repeat. ``{first_call}``, ``{repeat}``."""

  warning: str = DEFAULT_REPEAT_WARNING
  """Appended to the note on repeat ``notes_before_stop``. ``{repeat}``."""


class PipelineConfig(BaseModel):
  layers: list[LayerConfig]
  token_budget: int = 200_000
  max_tool_iterations: int = 25
  max_parallel_tools: int | None = None
  replay_reasoning: bool = True
  llm_timeout_seconds: float | None = None
  circuit_breaker_failure_threshold: int = 3
  circuit_breaker_recovery_timeout: float = 60.0
  degradation: DegradationConfig | None = None
  final_answer: FinalAnswerConfig = FinalAnswerConfig()
  repeat_guard: RepeatGuardConfig = RepeatGuardConfig()
