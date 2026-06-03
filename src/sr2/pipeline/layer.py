"""Layer — the core unit of the SR2 pipeline.

Holds resolvers and transformers, accumulates content, enforces token
budgets, and compiles content for its compilation target.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
from sr2.pipeline.compilation import PositionStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.protocols import Resolver, TokenCounter, ToolProvider, Transformer
from sr2.pipeline.provenance import Entry, InMemoryProvenanceStore, ProvenanceStore

from sr2.pipeline.tracing import FiringRecord

if TYPE_CHECKING:
    from sr2.pipeline.tracing import Tracer


class Layer:
    """A single pipeline layer that accumulates, budgets, and compiles content."""

    def __init__(
        self,
        name: str,
        target: CompilationTarget,
        position: PositionStrategy,
        token_budget: int | None,
        resolvers: list[Resolver],
        transformers: list[Transformer],
        token_counter: TokenCounter,
        event_bus: EventBus | None = None,
        provenance_store: ProvenanceStore | None = None,
        token_threshold_pct: float | None = None,
        tool_providers: list | None = None,
        tracer: "Tracer | None" = None,
    ) -> None:
        self.name = name
        self.target = target
        self._position = position
        self.token_budget = token_budget
        self.token_threshold_pct = token_threshold_pct
        self.resolvers = resolvers
        self.transformers = transformers
        self.tool_providers = list(tool_providers) if tool_providers is not None else []
        self._token_counter = token_counter
        self._event_bus: EventBus | None = event_bus
        self._provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )

        self._tracer: "Tracer | None" = tracer

        self._turn_seq: int = 0
        self._next_firing_seq: Callable[[], int] = lambda: 0  # replaced by engine before each run

        self._content: list[ContentBlock | Message] = []
        self._tool_definitions: list[ToolDefinition] = []
        self._force_truncated: bool = False
        # Collected events from bus callbacks — processed by engine after drain
        self._pending_events: list[Event] = []
        # Entries buffered for store write — flushed in process_pending
        self._pending_writes: list[Entry] = []

    # -- wiring ---------------------------------------------------------------

    def wire(
        self,
        bus: EventBus,
        provenance_store: ProvenanceStore,
        tracer: "Tracer | None",
    ) -> None:
        """Wire this layer to the engine's shared bus, provenance store, and tracer.

        If the layer was previously wired to a different bus, this method
        removes the layer's handle_event callback from that bus before
        switching to the new one. This prevents the old bus from delivering
        events to this layer after rewiring.
        """
        # Unsubscribe handle_event from the old bus if it differs from the new one.
        # Bound methods can't be compared with `is` — compare __func__ and __self__
        # to detect that a stored callback is this layer's handle_event method.
        if self._event_bus is not None and self._event_bus is not bus:
            handle_event_func = self.handle_event.__func__  # type: ignore[attr-defined]
            self._event_bus._subs = [
                (name, cb, is_async)
                for name, cb, is_async in self._event_bus._subs
                if not (
                    hasattr(cb, "__func__")
                    and cb.__func__ is handle_event_func
                    and cb.__self__ is self
                )
            ]

        self._event_bus = bus
        self._provenance_store = provenance_store
        self._tracer = tracer

    # -- session seeding ------------------------------------------------------

    def seed(self, messages: list[Message]) -> None:
        """Seed conversation history on any SessionResolver in this layer.

        Calls seed() on each resolver that exposes the method. Resolvers that
        don't implement seed() (i.e. non-session resolvers) are silently skipped.
        """
        from sr2.pipeline.resolvers.session import SessionResolver

        for resolver in self.resolvers:
            if isinstance(resolver, SessionResolver):
                resolver.seed(messages)

    # -- subscriptions --------------------------------------------------------

    @property
    def subscriptions(self) -> list[EventSubscription]:
        """All event subscriptions from this layer's resolvers, transformers, and tool providers."""
        subs: list[EventSubscription] = []
        for comp in [*self.resolvers, *self.transformers, *self.tool_providers]:
            subs.extend(comp.subscriptions)
        return subs

    # -- done check -----------------------------------------------------------

    def is_done(self) -> bool:
        """True when all components are idle (never fired) or exhausted (hit max_executions)."""
        for comp in [*self.resolvers, *self.transformers, *self.tool_providers]:
            if comp.execution_count > 0 and comp.execution_count < comp.max_executions:
                return False
        return True

    # -- event handling (sync callback) ---------------------------------------

    def handle_event(self, event: Event) -> None:
        """Collect incoming events. Called as a bus callback during drain.

        This is intentionally sync — it just buffers events. The engine
        processes them asynchronously after drain completes.
        """
        self._pending_events.append(event)

    def _snapshot_after(
        self, kind: str, tokens_before: int
    ) -> tuple[list, int, int]:
        if kind == "tool_provider":
            return [td.name for td in self._tool_definitions], 0, 0
        content_after = list(self._content)
        tokens_after = self._token_counter.count(content_after)
        return content_after, tokens_after, tokens_after - tokens_before

    def _build_record(
        self,
        *,
        kind: str,
        comp: object,
        events: list[Event],
        content_before: list,
        tokens_before: int,
        content_after: list,
        tokens_after: int,
        tokens_delta: int,
        duration_ms: float,
        iter_seq: int,
        status: str,
        error: str | None = None,
    ) -> FiringRecord:
        return FiringRecord(
            turn_seq=self._turn_seq,
            firing_seq=self._next_firing_seq(),
            kind=kind,
            component_name=comp.name,  # type: ignore[union-attr]
            layer=self.name,
            trigger_events=[e.name for e in events],
            content_before=content_before,
            content_after=content_after,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_delta=tokens_delta,
            duration_ms=duration_ms,
            status=status,
            error=error,
            iteration_seq=iter_seq,
        )

    async def _fire_component(
        self,
        comp: object,
        kind: str,
        events: list[Event],
    ) -> object:
        """Shared try/except/FiringRecord logic for resolvers, transformers, and tool providers.

        Dispatches to the correct component method based on *kind*, snapshots
        content before/after when a tracer is attached, emits a FiringRecord on
        success or failure, and re-raises any exception after recording it.

        Returns the raw result from the component call so the caller can apply
        kind-specific post-processing (e.g. transformer execution_count guard,
        result.entries buffering, result.events queuing).

        Signature: _fire_component(comp, kind, events)
          - kind='resolver'      → calls comp.resolve(events), adds content via add_content()
          - kind='transformer'   → calls comp.transform(get_content(), events),
                                   applies set_content(result.content) if non-None so the
                                   content_after snapshot reflects the post-transform state;
                                   result.entries / result.events remain the caller's responsibility
          - kind='tool_provider' → calls comp.provide(events), adds defs via add_tool_definitions()
        """
        # Snapshot before
        if self._tracer is not None:
            if kind == "tool_provider":
                content_before: list = [td.name for td in self._tool_definitions]
                tokens_before = 0
            else:
                content_before = list(self._content)
                tokens_before = self._token_counter.count(content_before)
            t_start = time.perf_counter()

        try:
            if kind == "resolver":
                result = await comp.resolve(events)  # type: ignore[union-attr]
                self.add_content(result)
            elif kind == "transformer":
                result = await comp.transform(self.get_content(), events)  # type: ignore[union-attr]
                # Apply content replacement before snapshotting content_after
                if result.content is not None:
                    self.set_content(result.content)
            else:  # tool_provider
                result = await comp.provide(events)  # type: ignore[union-attr]
                self.add_tool_definitions(result)

            if self._tracer is not None:
                content_after, tokens_after, tokens_delta = self._snapshot_after(kind, tokens_before)
                duration_ms = (time.perf_counter() - t_start) * 1000
                iter_seq = events[0].iteration_seq if events else 1
                self._tracer.on_firing(self._build_record(
                    kind=kind, comp=comp, events=events,
                    content_before=content_before, tokens_before=tokens_before,
                    content_after=content_after, tokens_after=tokens_after,
                    tokens_delta=tokens_delta, duration_ms=duration_ms,
                    iter_seq=iter_seq, status="ok",
                ))

        except Exception as exc:
            if self._tracer is not None:
                content_after, tokens_after, tokens_delta = self._snapshot_after(kind, tokens_before)
                duration_ms = (time.perf_counter() - t_start) * 1000
                iter_seq = events[0].iteration_seq if events else 1
                self._tracer.on_firing(self._build_record(
                    kind=kind, comp=comp, events=events,
                    content_before=content_before, tokens_before=tokens_before,
                    content_after=content_after, tokens_after=tokens_after,
                    tokens_delta=tokens_delta, duration_ms=duration_ms,
                    iter_seq=iter_seq, status="failed", error=str(exc),
                ))
            raise

        return result

    async def process_pending(self) -> bool:
        """Process all pending collected events. Returns True if new events were generated.

        Drives resolvers and transformers against collected events, adds
        content, checks budget, and emits overflow events if needed.
        Also flushes any buffered provenance writes to the store.
        """
        changed = False

        if self._pending_events:
            events = self._pending_events
            self._pending_events = []

            # --- Resolvers ---
            for resolver in self.resolvers:
                if resolver.execution_count >= resolver.max_executions:
                    continue
                if not any(
                    s.matches(e)
                    for s in resolver.subscriptions
                    for e in events
                ):
                    continue
                await self._fire_component(comp=resolver, kind="resolver", events=events)

            # --- Transformers ---
            for transformer in self.transformers:
                if transformer.execution_count >= transformer.max_executions:
                    continue
                if not any(
                    s.matches(e)
                    for s in transformer.subscriptions
                    for e in events
                ):
                    continue
                _count_before = transformer.execution_count
                result = await self._fire_component(comp=transformer, kind="transformer", events=events)
                if transformer.execution_count == _count_before:
                    transformer.execution_count += 1
                # set_content already applied inside _fire_component
                if result.entries:
                    self._pending_writes.extend(result.entries)
                # Queue events emitted by transformer
                if result.events:
                    for ev in result.events:
                        self._event_bus.queue(ev)

            # --- Tool Providers ---
            for tp in self.tool_providers:
                if tp.execution_count >= tp.max_executions:
                    continue
                if not any(
                    s.matches(e)
                    for s in tp.subscriptions
                    for e in events
                ):
                    continue
                await self._fire_component(comp=tp, kind="tool_provider", events=events)

            # --- Budget check ---
            self.check_budget()

            # Force-truncate if over budget
            warning = self.force_truncate()
            if warning:
                self._event_bus.queue(
                    Event(
                        name="truncation",
                        phase=EventPhase.COMPLETED,
                        source_layer=self.name,
                        data=warning,
                    )
                )

            # True if any events were queued (overflow, truncation, etc.)
            changed = not self._event_bus.is_empty()

        # Flush pending provenance writes (regardless of whether events fired)
        if self._pending_writes:
            await self._provenance_store.write_batch(self._pending_writes)
            self._pending_writes = []

        return changed

    # -- content management --------------------------------------------------

    def get_content(self) -> list[ContentBlock | Message]:
        return list(self._content)

    def add_content(self, resolved: ResolvedContent) -> None:
        if resolved.entries:
            # New path: entries with provenance — extract content blocks and buffer for store
            entry_contents = [e.content for e in resolved.entries]
            self._content = self._position.place(self._content, entry_contents)
            self._pending_writes.extend(resolved.entries)
        elif resolved.content:
            # Old path: raw content blocks (backward compat) — no store write
            self._content = self._position.place(self._content, resolved.content)
        # Queue any events emitted by the resolver onto the bus
        if resolved.events and self._event_bus is not None:
            for ev in resolved.events:
                self._event_bus.queue(ev)

    def set_content(self, content: list[ContentBlock | Message]) -> None:
        self._content = list(content)
        if not content:
            self._force_truncated = False

    def reset_tools(self) -> None:
        self._tool_definitions = []

    def add_tool_definitions(self, defs: list[ToolDefinition]) -> None:
        self._tool_definitions.extend(defs)

    # -- budget --------------------------------------------------------------

    def check_budget(self) -> None:
        if self.token_budget is None:
            return
        used = self._token_counter.count(self._content)
        if self.token_threshold_pct is not None and used >= self.token_budget * self.token_threshold_pct:
            self._event_bus.queue(
                Event(
                    name="token_threshold",
                    phase=EventPhase.COMPLETED,
                    source_layer=self.name,
                    data={"used": used, "budget": self.token_budget, "pct": self.token_threshold_pct},
                )
            )
        if used > self.token_budget:
            self._event_bus.queue(
                Event(
                    name="overflow",
                    phase=EventPhase.COMPLETED,
                    source_layer=self.name,
                )
            )

    def force_truncate(self) -> str | None:
        if self.token_budget is None:
            return None

        used = self._token_counter.count(self._content)
        if used <= self.token_budget:
            return None

        self._force_truncated = True
        removed = 0
        while self._content and self._token_counter.count(self._content) > self.token_budget:
            self._content.pop(0)
            removed += 1

        return (
            f"Layer '{self.name}': force-truncated {removed} block(s) "
            f"to fit budget of {self.token_budget} tokens"
        )

    # -- compile --------------------------------------------------------------

    def compile(self) -> list[TextBlock] | list[Message] | list[ToolDefinition]:
        _COMPILE_DISPATCH: dict[CompilationTarget, Callable[[], list]] = {
            CompilationTarget.SYSTEM: self._compile_system,
            CompilationTarget.MESSAGES: self._compile_messages,
            CompilationTarget.TOOLS: self._compile_tools,
        }
        return _COMPILE_DISPATCH[self.target]()

    def _compile_system(self) -> list[TextBlock]:
        out: list[TextBlock] = []
        for block in self._content:
            if isinstance(block, TextBlock):
                out.append(block)
        return out

    def _compile_messages(self) -> list[Message]:
        if not self._content:
            return []

        messages: list[Message] = []
        raw_blocks: list[ContentBlock] = []

        for item in self._content:
            if isinstance(item, Message):
                # Flush any accumulated raw blocks first
                if raw_blocks:
                    messages.append(Message(role="user", content=raw_blocks))
                    raw_blocks = []
                messages.append(item)
            else:
                raw_blocks.append(item)

        # Flush trailing raw blocks
        if raw_blocks:
            messages.append(Message(role="user", content=raw_blocks))

        return messages

    def _compile_tools(self) -> list[ToolDefinition]:
        return list(self._tool_definitions)

    @property
    def blocks(self) -> list[ContentBlock | Message]:
        """Expose accumulated content blocks for engine to read."""
        return list(self._content)
