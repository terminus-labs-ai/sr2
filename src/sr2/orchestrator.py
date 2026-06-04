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
from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from sr2.pipeline.provenance import InMemoryProvenanceStore, ProvenanceStore

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
from sr2.pipeline.protocols import Resolver, TokenCounter, ToolSource, Transformer
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
_RESOLVERS: PluginRegistry = PluginRegistry("sr2.resolvers")
_TRANSFORMERS: PluginRegistry = PluginRegistry("sr2.transformers")
_TOOL_PROVIDERS: PluginRegistry = PluginRegistry("sr2.tool_providers")


def _build_resolver(config: ResolverConfig, deps: Dependencies) -> Any:
    return _RESOLVERS.get(config.type).build(config, deps)


def _build_transformer(config: TransformerConfig, deps: Dependencies) -> Any:
    return _TRANSFORMERS.get(config.type).build(config, deps)


def _build_tool_provider(config: ToolProviderConfig, deps: Dependencies) -> Any:
    return _TOOL_PROVIDERS.get(config.type).build(config, deps)


def reset_discovery() -> None:
    """Reset all plugin registry discovery state.

    Clears the cached entry-point results so the next call to ``get()`` or
    ``names()`` on any registry re-scans entry points.

    Primarily intended as a test hook to isolate tests that rely on
    entry-point discovery from one another or from the live environment.
    """
    _RESOLVERS.reset_discovery()
    _TRANSFORMERS.reset_discovery()
    _TOOL_PROVIDERS.reset_discovery()


def _build_layer(
    layer_config: LayerConfig,
    token_counter: TokenCounter,
    deps: Dependencies,
    bus: EventBus | None = None,
    provenance_store: "ProvenanceStore | None" = None,
) -> Layer:
    target = CompilationTarget(layer_config.target)

    position_str = layer_config.position or "append"
    position = PrefixStrategy() if position_str == "prefix" else AppendStrategy()

    resolvers = [_build_resolver(r, deps) for r in layer_config.resolvers]
    transformers = [_build_transformer(t, deps) for t in (layer_config.transformers or [])]
    tool_providers = [_build_tool_provider(tp, deps) for tp in (layer_config.tool_providers or [])]

    event_bus = bus if bus is not None else EventBus()

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
        provenance_store=provenance_store,
        degradation_category=layer_config.degradation_category,
        priority=layer_config.priority,
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
        tracer: "Tracer | None" = None,
        memory_store: "MemoryStore | None" = None,
        memory_extractor: "MemoryExtractor | None" = None,
        tool_executor: "ToolExecutor | None" = None,
        active_frame_provider: Callable[[str], str | None] | None = None,
        tool_source: "ToolSource | None" = None,
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
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=pipeline_config.circuit_breaker_failure_threshold,
            recovery_timeout=pipeline_config.circuit_breaker_recovery_timeout,
        )
        self._llm_timeout_seconds = pipeline_config.llm_timeout_seconds
        self._pp_task: asyncio.Task[None] | None = None

        deps = Dependencies(
            llm=llm_dict,
            memory_store=memory_store,
            memory_extractor=memory_extractor,
            session_id=self.session_id,
            active_frame_provider=active_frame_provider,
            tool_source=tool_source,
        )
        self._active_frame_provider = deps.active_frame_provider

        # Resolve shared infrastructure before building layers so layers receive
        # the real bus and provenance store — not throwaway placeholders.
        shared_bus = EventBus()
        resolved_provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )

        layers = [
            _build_layer(lc, token_counter, deps, bus=shared_bus, provenance_store=resolved_provenance_store)
            for lc in pipeline_config.layers
        ]

        # FR5: Build the degradation ladder from config (or None if absent)
        ladder = None
        if pipeline_config.degradation is not None:
            from sr2.degradation.ladder import DegradationLadder

            ladder = DegradationLadder.from_config(pipeline_config.degradation)

        self._engine = PipelineEngine(
            layers=layers,
            token_counter=token_counter,
            provenance_store=resolved_provenance_store,
            token_budget=pipeline_config.token_budget,
            tracer=tracer,
            bus=shared_bus,
            ladder=ladder,
        )

    # ------------------------------------------------------------------
    # Block stamping
    # ------------------------------------------------------------------

    def _stamp_block(self, block: Any, origin: str | None = None) -> None:
        """Stamp ``meta["frame"]`` on *block* when an active-frame provider is set.

        The *origin* parameter identifies the transport/source of the current
        turn.  The provider uses it to resolve the correct frame
        (work-frame if open on that origin, else ambient frame).
        """
        if self._active_frame_provider is None:
            return
        frame = self._active_frame_provider(origin or "")
        if frame is not None:
            block.meta["frame"] = frame

    # ------------------------------------------------------------------
    # Public inspection helpers
    # ------------------------------------------------------------------

    @property
    def provenance_store(self) -> "ProvenanceStore":
        """Expose the active provenance store for testing and inspection."""
        return self._engine.provenance_store

    @property
    def bus(self) -> "EventBus":
        """Expose the shared event bus for tool-level event emission."""
        return self._engine.bus

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

    async def turn(self, user_input: list, *, origin: str = "") -> AsyncIterator[StreamEvent]:
        """Async generator: runs the pipeline and streams LLM events.

        Implements the multi-iteration tool loop (FR3):
          start_turn → stream LLM → detect tool_use → execute tools →
          continue_turn → repeat until no tool_use → end_turn

        FR11: post_process is awaited once after the full loop (not per iteration).
        FR14: exactly one StreamEvent(type="end") is emitted to the caller.
        FR7:  raises ToolLoopLimitError if iterations exceed max_tool_iterations.

        sr2-80: Finalization (assistant_response + end_turn + post_process) runs
        BEFORE yielding any events in the final iteration, so an early break
        by the consumer does not silently skip session-history capture or
        memory extraction.
        """
        # FR4+FR5+FR6: Await any pending deferred post_process task from the
        # previous turn before starting a new one. This ensures turn N's
        # post_process completes before turn N+1 begins.
        # FR8: Surface deferred-task errors as StreamEvent(type="error") early
        # in the stream. CancelledError propagates unchanged.
        if self._pp_task is not None:
            try:
                await self._pp_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield StreamEvent(type="error", errors=[f"deferred post_process error: {exc}"])
            self._pp_task = None

        # Reset all pipeline components so turn 2+ re-fires them.
        self._engine.reset_execution_counts()

        # FR7: Accumulate bus errors across in-band drains.
        in_band_errors: list[str] = []

        # Start the turn via the new engine API.
        next_seq = self._engine._turn_seq + 1
        await self._engine.start_turn(turn_seq=next_seq)
        in_band_errors.extend(self._engine.bus.get_errors())

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
            in_band_errors.extend(self._engine.bus.get_errors())

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

            # Circuit breaker check — reject immediately if open.
            if not self._circuit_breaker.allow_request():
                raise CircuitBreakerOpenError("LLM circuit breaker is open")

            # Collect all stream events, applying optional per-call timeout.
            async def _collect_stream() -> list[StreamEvent]:
                collected: list[StreamEvent] = []
                async for event in self._llm.stream(current_request):
                    collected.append(event)
                return collected

            try:
                if self._llm_timeout_seconds is not None:
                    raw_events = await asyncio.wait_for(
                        _collect_stream(), timeout=self._llm_timeout_seconds
                    )
                else:
                    raw_events = await _collect_stream()
                self._circuit_breaker.record_success()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._circuit_breaker.record_failure()
                raise

            # Classify events from this LLM call.
            for event in raw_events:
                if event.type == "text" and event.text:
                    iter_text.append(event.text)
                elif event.type == "tool_use":
                    iter_tool_use.append(event)
                elif event.type == "usage" and event.usage is not None:
                    iter_usage = event.usage

            full_iter_text = "".join(iter_text)
            usage = iter_usage if iter_usage is not None else TokenUsage()

            if not iter_tool_use:
                # No tool calls — this is the final LLM response. Build the
                # CompletionResponse.
                content: list = []
                if full_iter_text:
                    block = TextBlock(text=full_iter_text)
                    self._stamp_block(block, origin)
                    content.append(block)
                stop_reason = "end_turn"
                final_response = CompletionResponse(
                    id="turn-response",
                    content=content,
                    stop_reason=stop_reason,
                    usage=usage,
                )

                # sr2-80: Finalize BEFORE yielding any events so an early
                # consumer break does not skip session-history capture.
                await self._finalize_turn(final_response)

                # Stream text/usage events to the caller (they were already
                # collected; yielding now preserves streaming semantics).
                for event in raw_events:
                    if event.type == "text" and event.text:
                        yield event
                    elif event.type == "usage" and event.usage is not None:
                        yield event

                # FR7: Surface accumulated in-band bus errors before the final end event.
                if in_band_errors:
                    yield StreamEvent(type="error", errors=in_band_errors)

                yield StreamEvent(type="end")
                return

            # ----------------------------------------------------------------
            # Tool-use iteration — execute tools and loop back
            # ----------------------------------------------------------------

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
                text_block = TextBlock(text=full_iter_text)
                self._stamp_block(text_block, origin)
                assistant_content.append(text_block)

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
                self._stamp_block(tool_block, origin)
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
            for result_block in tool_result_blocks:
                self._stamp_block(result_block, origin)

            # FR9: Queue tool_use_emitted on the engine bus (internal subscribers).
            self._engine.bus.queue(
                Event(
                    name="tool_use_emitted",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=tool_use_blocks,
                    iteration_seq=iteration_seq,
                )
            )

            # FR9: Queue tool_result_received on the engine bus (internal subscribers).
            self._engine.bus.queue(
                Event(
                    name="tool_result_received",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=tool_result_blocks,
                    iteration_seq=iteration_seq,
                )
            )

            # FR10: Queue assistant_iteration_response — the intermediate CompletionResponse
            # from this tool-use iteration (NOT the final turn response).
            intermediate_response = CompletionResponse(
                id="iter-response",
                content=list(assistant_content),
                stop_reason="tool_use",
                usage=usage,
            )
            self._engine.bus.queue(
                Event(
                    name="assistant_iteration_response",
                    phase=EventPhase.COMPLETED,
                    source_layer="orchestrator",
                    data=intermediate_response,
                    iteration_seq=iteration_seq,
                )
            )

            # Stream text events from this iteration to the caller.
            for event in raw_events:
                if event.type == "text" and event.text:
                    yield event
                elif event.type == "usage" and event.usage is not None:
                    yield event

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
            in_band_errors.extend(self._engine.bus.get_errors())

            # FR13: emit iteration_complete to signal the end of this tool iteration.
            yield StreamEvent(type="iteration_complete", iteration=iteration_seq)

            iteration_seq += 1

    async def _finalize_turn(self, final_response: CompletionResponse) -> None:
        """Queue assistant_response, drain via end_turn, schedule post_process.

        Called BEFORE yielding the final end event (sr2-80) so that an early
        consumer break does not silently skip session-history capture or
        memory extraction.
        """
        # Queue assistant_response on the engine bus so session resolver captures it.
        self._engine.bus.queue(
            Event(
                name="assistant_response",
                phase=EventPhase.COMPLETED,
                source_layer="orchestrator",
                data=final_response,
            )
        )

        # Finalize the turn: fire turn_end, drain the bus (processes
        # assistant_response subscribers: SessionResolver, MemoryExtractionTransformer),
        # compile, and return PipelineResult.
        await self._engine.end_turn()

        # FR4+FR5+FR6: Schedule post_process as a deferred task. The client
        # is freed immediately; the next turn() will await this task before
        # proceeding. Capture final_response by argument.
        self._pp_task = asyncio.create_task(
            self._finalize_and_post_process(final_response)
        )

    # ------------------------------------------------------------------
    # Post-processing (MVP no-op)
    # ------------------------------------------------------------------

    async def _finalize_and_post_process(self, final_response: CompletionResponse) -> None:
        """Deferred post-processing: called by the scheduled task after end_turn.

        Captures final_response by argument to avoid closure issues.
        """
        await self.post_process(final_response)

    async def aclose(self) -> None:
        """Explicit shutdown: await any pending deferred post_process task.

        Safe when no task is pending (no-op). Idempotent — clears _pp_task
        so subsequent calls are also no-ops.

        FR9: Surfaces errors from the deferred task rather than swallowing them.
        """
        if self._pp_task is None:
            return

        try:
            await self._pp_task
        finally:
            self._pp_task = None

    async def post_process(self, response: CompletionResponse) -> None:
        """Post-turn processing hook. No-op in MVP."""
        return
