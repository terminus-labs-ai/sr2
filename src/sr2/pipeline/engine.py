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
    ) -> None:
        self.token_counter = token_counter
        self._max_cycles = max_cycles
        self._token_budget = token_budget
        self._tracer = tracer
        self._turn_seq: int = -1
        self._firing_seq: int = -1
        self._bus = EventBus()
        self._layers = layers
        self._provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )

        for layer in self._layers:
            layer._event_bus = self._bus
            layer._provenance_store = self._provenance_store

        self._setup_event_handlers()

        for layer in self._layers:
            layer._tracer = self._tracer

    def _next_firing_seq(self) -> int:
        """Increment and return the firing sequence counter."""
        self._firing_seq += 1
        return self._firing_seq

    def _setup_event_handlers(self) -> None:
        """Wire all layer component subscriptions to the shared bus."""
        for layer in self._layers:
            for subscription in layer.subscriptions:
                self._bus.subscribe(subscription, layer.handle_event)

    async def run(
        self,
        user_input: List[ContentBlock],
    ) -> PipelineResult:
        """Run the pipeline for a single turn."""
        self._turn_seq += 1
        self._firing_seq = -1
        self._bus.reset()
        for layer in self._layers:
            layer._turn_seq = self._turn_seq
            layer._next_firing_seq = self._next_firing_seq
        for layer in self._layers:
            layer.set_content([])
            if layer.tool_providers:
                layer.reset_tools()

        # --- Emit lifecycle events ---
        self._bus.queue(
            Event(
                name="turn_start",
                phase=EventPhase.COMPLETED,
                source_layer="engine",
            )
        )
        if user_input:
            self._bus.queue(
                Event(
                    name="user_input",
                    phase=EventPhase.COMPLETED,
                    source_layer="engine",
                    data=user_input,
                )
            )

        # --- Drain-process loop ---
        await self._run_loop()

        # --- Emit turn_end for post-processing transformers ---
        self._bus.queue(
            Event(
                name="turn_end",
                phase=EventPhase.COMPLETED,
                source_layer="engine",
            )
        )
        await self._run_loop()

        # --- Compile and collect metrics ---
        request = self._compile_request()
        metrics = self._build_metrics()

        return PipelineResult(request=request, metrics=metrics)

    async def _run_loop(self) -> None:
        """Drain the event bus and process layers until quiescent."""
        for _ in range(self._max_cycles):
            await self._bus._drain()
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
        """Compile all layers into a CompletionRequest."""
        system_blocks: List[TextBlock] = []
        messages: List[Message] = []
        tools: List[ToolDefinition] = []

        for layer in self._layers:
            compiled = layer.compile()
            if layer.target == CompilationTarget.SYSTEM:
                system_blocks.extend(compiled)
            elif layer.target == CompilationTarget.MESSAGES:
                messages.extend(compiled)
            elif layer.target == CompilationTarget.TOOLS:
                tools.extend(compiled)

        return CompletionRequest(
            system=system_blocks or None,
            messages=messages,
            tools=tools or None,
        )

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
        )

    @property
    def bus(self) -> EventBus:
        """Expose the bus for testing and inspection."""
        return self._bus
