"""Engine that runs the SR2 pipeline — turn loop, event emission, and metrics.

The engine is the orchestration layer. It wires layers to a shared bus, emits
lifecycle events (turn_start, user_input, turn_end), and drives the
drain-process loop until all events settle. Finally it compiles each layer's
content into a CompletionRequest and collects per-layer metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from typing import TYPE_CHECKING

from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
from sr2.pipeline.compilation import get_compilation_targets
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import (
    CompilationTarget,
    LayerMetrics,
    PipelineMetrics,
    PipelineResult,
)
from sr2.pipeline.protocols import TokenCounter
from sr2.pipeline.provenance import InMemoryProvenanceStore, ProvenanceStore
from sr2.protocols.llm import CompletionRequest

if TYPE_CHECKING:
    from sr2.degradation.ladder import DegradationLadder
    from sr2.pipeline.tracing import Tracer


class PipelineEngine:
    """Orchestration layer that runs the SR2 pipeline.

    Responsibilities:
    - Wire layers to a shared event bus
    - Emit lifecycle events (turn_start, user_input, turn_end)
    - Run the drain-process loop until all events settle
    - Compile layers into a CompletionRequest
    - Collect per-layer and aggregate metrics
    """

    def __init__(
        self,
        layers: List[Layer],
        token_counter: TokenCounter,
        provenance_store: ProvenanceStore | None = None,
        max_cycles: int = 50,
        token_budget: int | None = None,
        tracer: "Tracer | None" = None,
        bus: EventBus | None = None,
        ladder: "DegradationLadder | None" = None,
    ) -> None:
        self.token_counter = token_counter
        self._max_cycles = max_cycles
        self._token_budget = token_budget
        self._tracer = tracer
        self._turn_seq: int = -1
        self._firing_seq: int = -1
        self._bus = bus if bus is not None else EventBus()
        self._layers = layers
        self._provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )
        # FR5: Degradation ladder — None means degradation is fully disabled
        self._ladder: "DegradationLadder | None" = ladder

        self._wire_layers()

    def _next_firing_seq(self) -> int:
        """Increment and return the firing sequence counter."""
        self._firing_seq += 1
        return self._firing_seq

    def _wire_layers(self) -> None:
        """Wire all layers to the engine's shared bus, provenance store, and tracer,
        then register event subscriptions."""
        for layer in self._layers:
            layer.wire(self._bus, self._provenance_store, self._tracer)
            # handle_event only buffers events; resolvers are filtered
            # individually in process_pending. Register it at most once per
            # event name — otherwise an event subscribed by multiple resolvers
            # in the same layer is buffered once per subscription and duplicated
            # (obsidian-75d).
            subscribed_names: set[str] = set()
            for subscription in layer.subscriptions:
                if subscription.event_name in subscribed_names:
                    continue
                subscribed_names.add(subscription.event_name)
                self._bus.subscribe(subscription, layer.handle_event)

    async def start_turn(self, turn_seq: int) -> None:
        """Begin a new turn: reset state, emit turn_start, and drain the bus.

        Args:
            turn_seq: The sequence number for this turn (assigned directly).
        """
        # FR7: Reset the ladder to FULL at the start of every turn
        if self._ladder is not None:
            self._ladder.reset()

        self._turn_seq = turn_seq
        self._firing_seq = -1
        self._bus.reset()
        for layer in self._layers:
            layer._turn_seq = self._turn_seq
            layer._next_firing_seq = self._next_firing_seq
        for layer in self._layers:
            layer.set_content([])
            layer._pending_events = []
            if layer.tool_providers:
                layer.reset_tools()

        self._bus.queue(
            Event(
                name="turn_start",
                phase=EventPhase.COMPLETED,
                source_layer="engine",
            )
        )
        await self._run_loop()

    async def continue_turn(self, events: List[Event], iteration_seq: int) -> None:
        """Inject mid-turn events (e.g. tool results) and drain the bus.

        Does NOT reset the bus — existing subscriptions and state are preserved.

        Args:
            events: Events to inject into the bus for this iteration.
            iteration_seq: Iteration number — stamped onto each event so that
                FiringRecords produced during this iteration carry the correct
                iteration_seq for grouping in render_trace.
        """
        for event in events:
            event.iteration_seq = iteration_seq
            self._bus.queue(event)
        await self._run_loop()

    async def end_turn(self) -> "PipelineResult":
        """Finalise the turn: emit turn_end, drain, compile, and return result.

        Uses the degradation-aware compile loop (D5) when a ladder is set.

        Returns:
            PipelineResult with compiled CompletionRequest and metrics.
        """
        self._bus.queue(
            Event(
                name="turn_end",
                phase=EventPhase.COMPLETED,
                source_layer="engine",
            )
        )
        await self._run_loop()

        return await self._compile_with_degradation()

    async def _run_loop(self) -> None:
        """Drain the event bus and process layers until quiescent."""
        for _ in range(self._max_cycles):
            await self._bus.drain()
            changed = await self._process_layers()
            if not changed and self._bus.is_empty():
                break

    async def _process_layers(self) -> bool:
        """Process pending events in all layers. Returns True if any work done."""
        any_changed = False
        for layer in self._layers:
            changed = await layer.process_pending()
            if changed:
                any_changed = True
        return any_changed

    def _compile_request(self) -> CompletionRequest:
        """Compile all layers into a CompletionRequest.

        FR5: When a ladder is set, layers whose degradation_category is not in
        the ladder's active_categories() are excluded from compilation.
        Layers with no category (None) are always kept (structural).
        """
        system_blocks: List[TextBlock] = []
        messages: List[Message] = []
        tools: List[ToolDefinition] = []
        compilation_targets = get_compilation_targets()

        active_cats: set[str] | None = (
            self._ladder.active_categories() if self._ladder else None
        )

        for layer in self._layers:
            # FR5: Skip layers whose category is not active
            if active_cats is not None and layer.degradation_category is not None:
                if layer.degradation_category not in active_cats:
                    continue

            compiled = layer.compile()
            compilation_targets[layer.target].collect(
                compiled, system_blocks, messages, tools
            )

        return CompletionRequest(
            system=system_blocks or None,
            messages=messages,
            tools=tools or None,
        )

    def _count_request_tokens(self, request: CompletionRequest) -> int:
        """Count total tokens in a compiled CompletionRequest."""
        total = 0
        if request.system:
            total += self.token_counter.count(request.system)
        if request.messages:
            for msg in request.messages:
                total += self.token_counter.count(msg.content)
        return total

    async def _compile_with_degradation(self) -> "PipelineResult":
        """Compile with the D5 step-down/recompile loop.

        D5: compile → if over budget → step_down → recompile → until fits or
        the bottom level is reached.  Then build metrics and return result.

        When no ladder is set, delegates to the simple path
        (_compile_request + _build_metrics).
        """
        if self._ladder is None:
            request = self._compile_request()
            if self._tracer is not None:
                self._tracer.on_compile(request)
            metrics = self._build_metrics()
            return PipelineResult(request=request, metrics=metrics)

        # Ladder is set — run the step-down/recompile loop
        steps_taken = 0
        request = self._compile_request()
        total_tokens = self._count_request_tokens(request)

        while (
            self._token_budget is not None
            and total_tokens > self._token_budget
            and self._ladder.current_level < len(self._ladder._levels) - 1
        ):
            self._ladder.step_down()
            steps_taken += 1
            request = self._compile_request()
            total_tokens = self._count_request_tokens(request)

        if self._tracer is not None:
            self._tracer.on_compile(request)

        metrics = self._build_metrics()

        # Add degradation warning if step-down occurred
        if steps_taken > 0:
            level_idx = self._ladder.current_level
            level_name = (
                self._ladder._levels[level_idx][0]
                if self._ladder._legacy
                else f"level-{level_idx}"
            )
            metrics.warnings.append(
                f"Degradation: stepped down {steps_taken} level(s) "
                f"due to budget overflow (total={total_tokens}, budget={self._token_budget})"
            )

        return PipelineResult(request=request, metrics=metrics)

    def _build_metrics(self) -> PipelineMetrics:
        """Build per-layer and aggregate metrics."""
        layers: Dict[str, LayerMetrics] = {}
        total_tokens = 0
        warnings: List[str] = []

        for layer in self._layers:
            content = layer.get_content()
            tokens_used = self.token_counter.count(content)
            total_tokens += tokens_used

            force_truncated = False
            budget_remaining: Optional[int] = None

            if layer.token_budget is not None:
                budget_remaining = layer.token_budget - tokens_used
                if tokens_used > layer.token_budget:
                    force_truncated = True

            # Check if force truncation happened (content was truncated to fit)
            # Even if currently under budget, truncation may have occurred
            if layer.token_budget is not None and any(
                r.execution_count > 0 for r in layer.resolvers
            ):
                # Detect if content was truncated by comparing token count to budget
                # The layer already ran force_truncate in process_pending
                pass

            resolver_execs = {
                r.name: r.execution_count
                for r in layer.resolvers
                if r.execution_count > 0
            }
            transformer_execs = {
                t.name: t.execution_count
                for t in layer.transformers
                if t.execution_count > 0
            }

            lm = LayerMetrics(
                tokens_used=tokens_used,
                token_budget=layer.token_budget,
                budget_remaining=budget_remaining,
                force_truncated=layer._force_truncated,
                resolver_executions=resolver_execs,
                transformer_executions=transformer_execs,
            )
            layers[layer.name] = lm

            if layer._force_truncated:
                warnings.append(
                    f"Layer '{layer.name}': force-truncated to fit budget of "
                    f"{layer.token_budget} tokens"
                )

        if self._token_budget is not None and total_tokens > self._token_budget:
            warnings.append(
                f"Pipeline token budget exceeded: {total_tokens} tokens used, "
                f"budget is {self._token_budget}"
            )

        return PipelineMetrics(
            layers=layers,
            total_tokens=total_tokens,
            warnings=warnings,
            bus_errors=self._bus.get_errors(),
        )

    def seed(self, messages: List[Message]) -> None:
        """Pre-populate conversation history across all layers.

        Delegates to Layer.seed() on each layer. Layers without a SessionResolver
        treat this as a no-op.
        """
        for layer in self._layers:
            layer.seed(messages)

    def reset_execution_counts(self) -> None:
        """Reset execution_count to zero on all resolvers, transformers, and tool providers.

        Called by SR2.turn() before each turn so all components re-fire.
        """
        for layer in self._layers:
            for comp in (*layer.resolvers, *layer.tool_providers, *layer.transformers):
                comp.execution_count = 0

    @property
    def bus(self) -> EventBus:
        """Expose the bus for testing and inspection."""
        return self._bus

    @property
    def layers(self) -> list:
        """Read-only view of the engine's layer list (for inspection and testing)."""
        return list(self._layers)

    @property
    def provenance_store(self) -> "ProvenanceStore":
        """Expose the active provenance store (for inspection and testing)."""
        return self._provenance_store
