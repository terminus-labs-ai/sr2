"""SR2 Facade — the public API for the context engineering library.

This is the ONLY entry point the harness touches. Everything else is internal.

Two entry points:
1. process(config, inputs) -> CompiledContext   — pre-LLM context compilation
2. post_process(turn_result) -> PostProcessResult — post-LLM memory/maintenance

The harness flow:
  1. Call process() to get compiled context
  2. Make the LLM call (provider-specific, harness concern)
  3. Translate response into SR2 TurnResult schema
  4. Call post_process(turn_result)

SR2 does NOT make the LLM call. It only manages context.

Design principles:
- SRP: Facade only orchestrates — delegates to pipeline engine, memory system.
- OCP: New capabilities added via plugins, never by modifying facade.
- Hard boundary: Harness gets a clean facade with defined inputs/outputs.
"""

from __future__ import annotations

from typing import Any

from sr2.config.loader import ConfigLoader
from sr2.config.models import PipelineConfig
from sr2.core.errors import SR2Error
from sr2.core.models import TurnResult
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.result import CompiledContext, PostProcessResult


class SR2:
    """Main facade for SR2 context engineering.

    This is the only class the harness imports and interacts with.
    Internal subsystems (pipeline, memory, compaction, degradation)
    are managed by SR2 — the harness doesn't know they exist.

    Usage:
        sr2 = SR2()

        # Pre-LLM: compile context
        context = await sr2.process(config, inputs)
        # Harness makes LLM call with context.to_text()

        # Post-LLM: process turn result
        result = await sr2.post_process(turn_result)
    """

    def __init__(self, config: str | PipelineConfig | dict[str, Any] | None = None) -> None:
        """Initialize SR2 with optional config.

        Args:
            config: YAML path, PipelineConfig instance, or dict.
                   Can also be set per-process() call.
        """
        self._default_config = None
        if config is not None:
            if isinstance(config, PipelineConfig):
                self._default_config = config
            else:
                self._default_config = ConfigLoader.load(config)

    async def process(
        self,
        config: str | PipelineConfig | dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
    ) -> CompiledContext:
        """Compile context for a single LLM call.

        Parses the config, resolves all layers and providers,
        applies budgets and caching, and returns compiled context.

        Args:
            config: Config override for this call. Uses default if not provided.
            inputs: Runtime inputs (session_id, user message, etc.)

        Returns:
            CompiledContext with assembled layers and metrics.

        Raises:
            SR2Error: If compilation fails.
        """
        pipeline_config = self._resolve_config(config)
        engine = PipelineEngine(pipeline_config)
        return await engine.compile()

    async def post_process(self, turn_result: TurnResult) -> PostProcessResult:
        """Process the result of an LLM turn.

        Handles memory extraction, compaction triggers, and conversation
        tracking. Maintenance is internal — SR2 decides when to run it.

        Args:
            turn_result: The LLM response translated into SR2's schema.

        Returns:
            PostProcessResult with extraction results and maintenance actions.
        """
        result = PostProcessResult()

        # TODO: Extract memories from turn_result.content
        # TODO: Check compaction/summarization triggers
        # TODO: Run maintenance if needed (staleness, merge, cleanup)

        result.metrics = {
            "content_length": len(turn_result.content),
            "tool_calls_count": len(turn_result.tool_calls),
            "prompt_tokens": turn_result.token_usage.prompt_tokens,
            "completion_tokens": turn_result.token_usage.completion_tokens,
        }

        return result

    def _resolve_config(
        self, config: str | PipelineConfig | dict[str, Any] | None
    ) -> PipelineConfig:
        """Resolve config from call-time arg or instance default."""
        if config is None:
            if self._default_config is None:
                return PipelineConfig()
            return self._default_config

        if isinstance(config, PipelineConfig):
            return config

        if isinstance(config, dict):
            return PipelineConfig.model_validate(config)

        # String = YAML path or raw YAML
        return ConfigLoader.load(config)
