"""SR2 — top-level orchestrator.

Wires PipelineConfig → PipelineEngine, drives the turn loop, streams LLM
responses, and emits the assistant_response event on the shared bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, Callable, TYPE_CHECKING

from ulid import ULID

from sr2.config.models import ConfigError, LayerConfig, PipelineConfig, ResolverConfig, ToolLoopLimitError, ToolProviderConfig, TransformerConfig
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
        self._max_tool_iterations = pipeline_config.max_tool_iterations
        self._max_parallel_tools = pipeline_config.max_parallel_tools
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
        """Async generator: runs the pipeline and streams LLM events.

        Implements the multi-iteration tool loop (FR3):
          start_turn → stream LLM → detect tool_use → execute tools →
          continue_turn → repeat until no tool_use → end_turn

        FR11: post_process is awaited once after the full loop (not per iteration).
        FR14: exactly one StreamEvent(type="end") is emitted to the caller.
        FR7:  raises ToolLoopLimitError if iterations exceed max_tool_iterations.
        """
        # Reset all pipeline components so turn 2+ re-fires them.
        self._engine.reset_execution_counts()

        # Start the turn via the new engine API.
        next_seq = self._engine._turn_seq + 1
        await self._engine.start_turn(turn_seq=next_seq)

        # Inject user_input as an event.
        if user_input:
            self._engine.bus.queue(
                Event(
                    name="user_input",
                    phase=EventPhase.COMPLETED,
                    source_layer="engine",
                    data=user_input,
                )
            )
            await self._engine._run_loop()

        # Compile the initial request after user_input is settled.
        compiled = self._engine._compile_request()

        # ----------------------------------------------------------------
        # Multi-iteration tool loop
        # ----------------------------------------------------------------
        iteration_seq = 0
        final_response: CompletionResponse | None = None

        # Build the mutable request we'll feed to the LLM each iteration.
        # Between iterations we'll add assistant + tool_result messages directly.
        current_request = compiled

        while True:
            # Accumulate stream events for this iteration.
            iter_text: list[str] = []
            iter_tool_use: list[StreamEvent] = []
            iter_usage: TokenUsage | None = None

            async for event in self._llm.stream(current_request):
                if event.type == "text" and event.text:
                    iter_text.append(event.text)
                    yield event
                elif event.type == "tool_use":
                    iter_tool_use.append(event)
                    # Do NOT yield tool_use events to the caller — they are
                    # internal to the loop; callers see only text and end.
                elif event.type == "usage" and event.usage is not None:
                    iter_usage = event.usage
                    yield event
                # Suppress intermediate "end" events (FR14 — single end at finish).
                # They will be replaced by the single end event we yield below.

            full_iter_text = "".join(iter_text)
            usage = iter_usage if iter_usage is not None else TokenUsage()

            if not iter_tool_use:
                # No tool calls — this is the final LLM response. Build the
                # CompletionResponse and exit the loop.
                content: list = []
                if full_iter_text:
                    content.append(TextBlock(text=full_iter_text))
                stop_reason = "end_turn"
                final_response = CompletionResponse(
                    id="turn-response",
                    content=content,
                    stop_reason=stop_reason,
                    usage=usage,
                )
                break

            # There are tool_use blocks — check the iteration limit before executing.
            if iteration_seq >= self._max_tool_iterations:
                raise ToolLoopLimitError(
                    f"Tool loop limit exceeded: {iteration_seq} iterations reached "
                    f"(max_tool_iterations={self._max_tool_iterations}). "
                    f"The LLM kept requesting tools without converging."
                )

            # Execute tools: wrap errors as ToolResultBlock(is_error=True).
            # CancelledError is never wrapped — it propagates immediately.
            assistant_content: list = []
            if full_iter_text:
                assistant_content.append(TextBlock(text=full_iter_text))

            if self._tool_executor is None:
                raise ConfigError(
                    "tool_executor not configured — set tool_executor= on SR2 to handle tool calls"
                )

            # Build all ToolUseBlocks in order (preserves mapping for result ordering).
            tool_use_blocks: list[ToolUseBlock] = []
            for tu_event in iter_tool_use:
                tool_block = ToolUseBlock(
                    id=tu_event.tool_use_id,
                    name=tu_event.tool_name,
                    input=tu_event.tool_input,
                )
                assistant_content.append(tool_block)
                tool_use_blocks.append(tool_block)

            # FR4+FR15: execute all tool blocks concurrently via asyncio.gather.
            # Optional semaphore caps concurrency when max_parallel_tools is set.
            sem = (
                asyncio.Semaphore(self._max_parallel_tools)
                if self._max_parallel_tools is not None
                else None
            )

            import inspect as _inspect

            async def _run_one(block: ToolUseBlock) -> ToolResultBlock:
                coro = self._tool_executor(block)  # type: ignore[misc]
                # If the caller passed a sync callable by mistake, the returned
                # value is not awaitable — that is a configuration error, not a
                # tool error, so we propagate TypeError immediately (not wrapped).
                if not _inspect.isawaitable(coro):
                    raise TypeError(
                        f"tool_executor must be an async callable (coroutine function); "
                        f"got non-awaitable result {type(coro)!r}"
                    )
                try:
                    if sem is not None:
                        async with sem:
                            return await coro
                    else:
                        return await coro
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return ToolResultBlock(
                        tool_use_id=block.id,
                        content=str(exc),
                        is_error=True,
                    )

            # asyncio.gather preserves input order: results[i] maps to tool_use_blocks[i].
            tool_result_blocks: list[ToolResultBlock] = list(
                await asyncio.gather(*(_run_one(b) for b in tool_use_blocks))
            )

            # Emit tool_use_emitted after all tool blocks are collected.
            yield StreamEvent(type="tool_use_emitted", tool_uses=tool_use_blocks)

            # Emit tool_result_received after all executor results are collected.
            yield StreamEvent(type="tool_result_received", tool_results=tool_result_blocks)

            # Append assistant message (with tool_use blocks) and tool_result
            # user message to the conversation for the next LLM call.
            from sr2.models import Message as _Message
            assistant_msg = _Message(role="assistant", content=assistant_content)
            tool_result_msg = _Message(
                role="user",
                content=list(tool_result_blocks),  # type: ignore[arg-type]
            )

            # Build the next request by extending messages.
            next_messages = list(current_request.messages) + [assistant_msg, tool_result_msg]
            current_request = CompletionRequest(
                system=current_request.system,
                messages=next_messages,
                tools=current_request.tools,
            )

            # Inform the engine about the tool results so session/state resolvers
            # can capture them for future turns.
            tool_result_events = [
                Event(
                    name="tool_result",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=rb,
                )
                for rb in tool_result_blocks
            ]
            await self._engine.continue_turn(tool_result_events, iteration_seq)

            # FR13: emit iteration_complete to signal the end of this tool iteration.
            yield StreamEvent(type="iteration_complete", iteration=iteration_seq)

            iteration_seq += 1

        # ----------------------------------------------------------------
        # Turn complete — emit a single end event (FR14).
        # ----------------------------------------------------------------
        if final_response is None:
            # Defensive: build an empty response if the loop exited unexpectedly.
            final_response = CompletionResponse(
                id="turn-response",
                content=[],
                stop_reason="end_turn",
                usage=TokenUsage(),
            )

        # Queue assistant_response on the engine bus so session resolver captures it.
        self._engine.bus.queue(
            Event(
                name="assistant_response",
                phase=EventPhase.COMPLETED,
                source_layer="orchestrator",
                data=final_response,
            )
        )

        yield StreamEvent(type="end")

        # FR11: await post_process (NOT fire-and-forget) before the generator exits.
        await self.post_process(final_response)

    # ------------------------------------------------------------------
    # Post-processing (MVP no-op)
    # ------------------------------------------------------------------

    async def post_process(self, response: CompletionResponse) -> None:
        """Post-turn processing hook. No-op in MVP."""
        return
