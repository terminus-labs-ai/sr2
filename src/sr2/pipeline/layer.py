"""Layer — the core unit of the SR2 pipeline.

Holds resolvers and transformers, accumulates content, enforces token
budgets, and compiles content for its compilation target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
from sr2.pipeline.compilation import PositionStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.protocols import Resolver, TokenCounter, ToolProvider, Transformer
from sr2.pipeline.provenance import Entry, InMemoryProvenanceStore, ProvenanceStore

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
        event_bus: EventBus,
        provenance_store: ProvenanceStore | None = None,
        token_threshold_pct: float | None = None,
        tool_providers: list = [],
        tracer: "Tracer | None" = None,
    ) -> None:
        self.name = name
        self.target = target
        self._position = position
        self.token_budget = token_budget
        self.token_threshold_pct = token_threshold_pct
        self.resolvers = resolvers
        self.transformers = transformers
        self.tool_providers = list(tool_providers)
        self._token_counter = token_counter
        self._event_bus = event_bus
        self._provenance_store: ProvenanceStore = (
            provenance_store if provenance_store is not None else InMemoryProvenanceStore()
        )

        self._tracer: "Tracer | None" = tracer

        self._content: list[ContentBlock | Message] = []
        self._tool_definitions: list[ToolDefinition] = []
        self._force_truncated: bool = False
        # Collected events from bus callbacks — processed by engine after drain
        self._pending_events: list[Event] = []
        # Entries buffered for store write — flushed in process_pending
        self._pending_writes: list[Entry] = []

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
                resolved = await resolver.resolve(events)
                self.add_content(resolved)

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
                result = await transformer.transform(self.get_content(), events)
                if transformer.execution_count == _count_before:
                    transformer.execution_count += 1
                if result.content is not None:
                    self.set_content(result.content)
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
                defs = await tp.provide(events)
                self.add_tool_definitions(defs)

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
        if self.target == CompilationTarget.SYSTEM:
            return self._compile_system()
        if self.target == CompilationTarget.MESSAGES:
            return self._compile_messages()
        if self.target == CompilationTarget.TOOLS:
            return self._compile_tools()
        return []  # pragma: no cover

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
