"""Layer — the core unit of the SR2 pipeline.

Holds resolvers and transformers, accumulates content, enforces token
budgets, and compiles content for its compilation target.
"""

from __future__ import annotations

from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
from sr2.pipeline.compilation import PositionStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.models import ResolvedContent, TransformationResult
from sr2.pipeline.protocols import Resolver, TokenCounter, Transformer


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
    ) -> None:
        self.name = name
        self.target = target
        self._position = position
        self.token_budget = token_budget
        self.resolvers = resolvers
        self.transformers = transformers
        self._token_counter = token_counter
        self._event_bus = event_bus

        self._content: list[ContentBlock | Message] = []
        self._tool_definitions: list[ToolDefinition] = []
        self._force_truncated: bool = False
        # Collected events from bus callbacks — processed by engine after drain
        self._pending_events: list[Event] = []

    # -- subscriptions --------------------------------------------------------

    @property
    def subscriptions(self) -> list[EventSubscription]:
        """All event subscriptions from this layer's resolvers and transformers."""
        subs: list[EventSubscription] = []
        for comp in [*self.resolvers, *self.transformers]:
            subs.extend(comp.subscriptions)
        return subs

    # -- done check -----------------------------------------------------------

    def is_done(self) -> bool:
        """True when all components are idle (never fired) or exhausted (hit max_executions)."""
        for comp in [*self.resolvers, *self.transformers]:
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
        """
        if not self._pending_events:
            return False

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
            result = await transformer.transform(self.get_content(), events)
            if result.content is not None:
                self.set_content(result.content)

            # Queue events emitted by transformer
            if result.events:
                for ev in result.events:
                    self._event_bus.queue(ev)

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

        # Return True if any events were queued (overflow, truncation, etc.)
        # The engine checks this to decide if another drain is needed.
        return not self._event_bus.is_empty()

    # -- content management --------------------------------------------------

    def get_content(self) -> list[ContentBlock | Message]:
        return list(self._content)

    def add_content(self, resolved: ResolvedContent) -> None:
        self._content = self._position.place(self._content, resolved.content)

    def set_content(self, content: list[ContentBlock | Message]) -> None:
        self._content = list(content)
        if not content:
            self._force_truncated = False

    def add_tool_definitions(self, defs: list[ToolDefinition]) -> None:
        self._tool_definitions.extend(defs)

    # -- budget --------------------------------------------------------------

    def check_budget(self) -> None:
        if self.token_budget is None:
            return
        used = self._token_counter.count(self._content)
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
