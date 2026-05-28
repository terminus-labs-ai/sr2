"""SR2 — top-level orchestrator.

Wires PipelineConfig → PipelineEngine, drives the turn loop, streams LLM
responses, and emits the assistant_response event on the shared bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, Callable, TYPE_CHECKING

from ulid import ULID

from sr2.config.models import ConfigError, LayerConfig, PipelineConfig, ResolverConfig, ToolProviderConfig, TransformerConfig
from sr2.pipeline.provenance import ProvenanceStore

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer
    from sr2.memory.protocol import MemoryExtractor, MemoryStore
from sr2.models import Message, TextBlock, TokenUsage, ToolResultBlock, ToolUseBlock
from sr2.pipeline.compilation import AppendStrategy, PrefixStrategy
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.protocols import Resolver, TokenCounter, Transformer
from sr2.plugins.registry import PluginRegistry
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMCallable,
    StreamEvent,
)

# FR1: Type alias for the tool executor callable.
ToolExecutor = Callable[[ToolUseBlock], Awaitable[ToolResultBlock]]

# Plugin registries (entry-point based, lazy discovery).
# object is used as the protocol to skip isinstance class-level validation —
# the Resolver/Transformer protocols define instance attributes, so checking
# the class itself yields false negatives. Correctness is enforced at build time.
_RESOLVERS: PluginRegistry = PluginRegistry("sr2.resolvers", object)
_TRANSFORMERS: PluginRegistry = PluginRegistry("sr2.transformers", object)
_TOOL_PROVIDERS: PluginRegistry = PluginRegistry("sr2.tool_providers", object)


def _build_resolver(config: ResolverConfig, deps: Dependencies) -> Any:
    return _RESOLVERS.get(config.type).build(config, deps)


def _build_transformer(config: TransformerConfig, deps: Dependencies) -> Any:
    return _TRANSFORMERS.get(config.type).build(config, deps)


def _build_tool_provider(config: ToolProviderConfig, deps: Dependencies) -> Any:
    return _TOOL_PROVIDERS.get(config.type).build(config, deps)


def _build_layer(layer_config: LayerConfig, token_counter: TokenCounter, deps: Dependencies) -> Layer:
    target = CompilationTarget(layer_config.target)

    position_str = layer_config.position or "append"
    position = PrefixStrategy() if position_str == "prefix" else AppendStrategy()

    resolvers = [_build_resolver(r, deps) for r in layer_config.resolvers]
    transformers = [_build_transformer(t, deps) for t in (layer_config.transformers or [])]
    tool_providers = [_build_tool_provider(tp, deps) for tp in (layer_config.tool_providers or [])]

    # EventBus is a placeholder here — PipelineEngine replaces it after init.
    event_bus = EventBus()

    return Layer(
        name=layer_config.name,
        target=target,
        position=position,
        token_budget=layer_config.token_budget,
        token_threshold_pct=layer_config.token_threshold_pct,
        resolvers=resolvers,
        transformers=transformers,
        tool_providers=tool_providers,
        token_counter=token_counter,
        event_bus=event_bus,
    )


class SR2:
    """Top-level orchestrator: turns a PipelineConfig into a streaming turn loop."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        llm: "LLMCallable | dict[str, LLMCallable]",
        token_counter: TokenCounter,
        session_id: str | None = None,
        provenance_store: ProvenanceStore | None = None,
        extras: Mapping[str, Any] | None = None,
        tracer: "Tracer | None" = None,
        memory_store: "MemoryStore | None" = None,
        memory_extractor: "MemoryExtractor | None" = None,
        tool_executor: "ToolExecutor | None" = None,
    ) -> None:
        # Normalise: bare LLMCallable → single-entry dict under "default".
        # Dict form: no "default" key requirement — callers may use any key names.
        if isinstance(llm, dict):
            llm_dict: dict[str, LLMCallable] = llm
        else:
            llm_dict = {"default": llm}  # type: ignore[assignment]

        if not llm_dict:
            raise ValueError("llm must not be empty")

        # Driver: explicit "default" key takes precedence; otherwise use first value.
        self._llm = llm_dict.get("default") or next(iter(llm_dict.values()))
        self._token_counter = token_counter
        self._tracer = tracer
        self._tool_executor: ToolExecutor | None = tool_executor
        self.session_id = session_id if session_id is not None else str(ULID())

        deps = Dependencies(
            llm=llm_dict,
            memory_store=memory_store,
            memory_extractor=memory_extractor,
            extras=extras or {},
        )
        layers = [_build_layer(lc, token_counter, deps) for lc in pipeline_config.layers]
        self._engine = PipelineEngine(
            layers=layers,
            token_counter=token_counter,
            provenance_store=provenance_store,
            token_budget=pipeline_config.token_budget,
            tracer=tracer,
        )

    # ------------------------------------------------------------------
    # Session seeding
    # ------------------------------------------------------------------

    def seed_session(self, messages: list[Message]) -> None:
        """Pre-populate conversation history in all SessionResolver instances.

        Delegates to engine.seed(), which propagates through Layer.seed() to
        each SessionResolver.seed(). Overwrites any existing history.
        No-op if no SessionResolver instances are found.
        """
        self._engine.seed(messages)

    # ------------------------------------------------------------------
    # Core turn loop
    # ------------------------------------------------------------------

    async def turn(self, user_input: list) -> AsyncIterator[StreamEvent]:
        """Async generator: runs the pipeline and streams LLM events."""
        # Reset all pipeline components so turn 2+ re-fires them.
        self._engine.reset_execution_counts()

        result = await self._engine.run(user_input)

        accumulated_text: list[str] = []
        accumulated_tool_use: list[StreamEvent] = []
        accumulated_usage: TokenUsage | None = None

        async def _stream() -> AsyncIterator[StreamEvent]:
            nonlocal accumulated_usage
            async for event in self._llm.stream(result.request):
                if event.type == "text" and event.text:
                    accumulated_text.append(event.text)
                elif event.type == "tool_use":
                    if self._tool_executor is None:
                        raise ConfigError(
                            "tool_executor not configured — set tool_executor= on SR2 to handle tool calls"
                        )
                    accumulated_tool_use.append(event)
                elif event.type == "usage" and event.usage is not None:
                    accumulated_usage = event.usage
                yield event

        async for stream_event in _stream():
            yield stream_event

        # Build CompletionResponse from accumulated stream events
        full_text = "".join(accumulated_text)
        content: list = []
        if full_text:
            content.append(TextBlock(text=full_text))
        for tu in accumulated_tool_use:
            tool_block = ToolUseBlock(id=tu.tool_use_id, name=tu.tool_name, input=tu.tool_input)
            content.append(tool_block)
            if self._tool_executor is not None:
                result_block = await self._tool_executor(tool_block)
                content.append(result_block)

        stop_reason = "tool_use" if accumulated_tool_use else "end_turn"
        usage = accumulated_usage if accumulated_usage is not None else TokenUsage()

        response = CompletionResponse(
            id="turn-response",
            content=content,
            stop_reason=stop_reason,
            usage=usage,
        )

        # Queue assistant_response on the engine bus
        self._engine.bus.queue(
            Event(
                name="assistant_response",
                phase=EventPhase.COMPLETED,
                source_layer="orchestrator",
                data=response,
            )
        )

        # Fire-and-forget post-processing
        asyncio.create_task(self.post_process(response))

    # ------------------------------------------------------------------
    # Post-processing (MVP no-op)
    # ------------------------------------------------------------------

    async def post_process(self, response: CompletionResponse) -> None:
        """Post-turn processing hook. No-op in MVP."""
        return
