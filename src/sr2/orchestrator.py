"""SR2 — top-level orchestrator.

Wires PipelineConfig → PipelineEngine, drives the turn loop, streams LLM
responses, and emits the assistant_response event on the shared bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ulid import ULID

from sr2.config.models import ConfigError, LayerConfig, PipelineConfig, ResolverConfig, ToolProviderConfig, TransformerConfig
from sr2.pipeline.provenance import ProvenanceStore
from sr2.models import Message, TextBlock, TokenUsage
from sr2.pipeline.compilation import AppendStrategy, PrefixStrategy
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import infer_compilation_target
from sr2.pipeline.protocols import Resolver, TokenCounter, Transformer
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.plugins.registry import PluginRegistry
from sr2.protocols.llm import (
    CompletionRequest,
    CompletionResponse,
    LLMCallable,
    StreamEvent,
)

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
    target = infer_compilation_target(layer_config.name, layer_config.target)

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
        llm: dict[str, LLMCallable],
        token_counter: TokenCounter,
        session_id: str | None = None,
        provenance_store: ProvenanceStore | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> None:
        if "default" not in llm:
            raise ValueError(
                "llm dict must contain a 'default' key. "
                f"Got keys: {list(llm.keys())!r}"
            )

        self._llm = llm["default"]
        self._token_counter = token_counter
        self.session_id = session_id if session_id is not None else str(ULID())

        deps = Dependencies(llm=llm, extras=extras or {})
        layers = [_build_layer(lc, token_counter, deps) for lc in pipeline_config.layers]
        self._engine = PipelineEngine(
            layers=layers,
            token_counter=token_counter,
            provenance_store=provenance_store,
            token_budget=pipeline_config.token_budget,
        )

    # ------------------------------------------------------------------
    # Session seeding
    # ------------------------------------------------------------------

    def seed_session(self, messages: list[Message]) -> None:
        """Pre-populate conversation history in all SessionResolver instances.

        Walks engine layers and sets each SessionResolver's _history to an
        independent copy of *messages*. Overwrites any existing history.
        No-op if no SessionResolver instances are found.
        """
        for layer in self._engine._layers:
            for resolver in layer.resolvers:
                if isinstance(resolver, SessionResolver):
                    resolver._history = [m.model_copy() for m in messages]

    # ------------------------------------------------------------------
    # Core turn loop
    # ------------------------------------------------------------------

    async def turn(self, user_input: list) -> AsyncIterator[StreamEvent]:
        """Async generator: runs the pipeline and streams LLM events."""
        # Reset all pipeline components so turn 2+ re-fires them.
        for layer in self._engine._layers:
            for comp in (*layer.resolvers, *layer.tool_providers, *layer.transformers):
                comp.execution_count = 0

        result = await self._engine.run(user_input)

        accumulated: list[str] = []

        async def _stream() -> AsyncIterator[StreamEvent]:
            async for event in self._llm.stream(result.request):
                if event.type == "text" and event.text:
                    accumulated.append(event.text)
                yield event

        async for stream_event in _stream():
            yield stream_event

        # Build CompletionResponse from accumulated text
        full_text = "".join(accumulated)
        response = CompletionResponse(
            id="turn-response",
            content=[TextBlock(text=full_text)],
            stop_reason="end_turn",
            usage=TokenUsage(),
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
