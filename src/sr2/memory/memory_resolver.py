"""MemoryResolver: injects relevant memories into the LLM context.

Subscribes to user_input events, uses the user's text as a search query
against the MemoryStore, and returns matching memories as context blocks.

Registered as a resolver type via the `sr2.resolvers` entry point.
"""

from __future__ import annotations

from sr2.config.models import ConfigError, ResolverConfig
from sr2.memory import MemoryScope, MemoryStore
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions, extract_user_input_text

_DEFAULT_SUBSCRIPTION = EventSubscription(
    event_name="user_input", phase=EventPhase.STARTING
)


class MemoryResolver:
    """Retrieve relevant memories for the current user input.

    On each resolve() call:
      1. Extract the user's text from the user_input event.
      2. Search the MemoryStore with that text.
      3. Return matching memories as TextBlocks in system context.

    Config options:
      - scope: MemoryScope filter (default: all scopes)
      - limit: max memories to inject (default: 5)
      - prefix: text prefix for injected memories (default: "Relevant context:\\n")
    """

    name: str = "memory"

    def __init__(self, config: ResolverConfig, store: MemoryStore) -> None:
        self._config = config
        self._store = store
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0

        cfg = config.config or {}
        scope_str = cfg.get("scope")
        self._scope = MemoryScope(scope_str.lower()) if scope_str else None
        self._limit = int(cfg.get("limit", 5))
        self._prefix = cfg.get("prefix", "Relevant context:\n")

        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> "MemoryResolver":
        """Build from config, pulling MemoryStore from the typed deps field.

        Reads ``deps.memory_store`` (typed field only).
        """
        store: MemoryStore | None = deps.memory_store

        if store is None:
            raise ConfigError(
                "resolver 'memory' requires a memory_store. "
                "Pass it as deps.memory_store."
            )

        return cls(config, store)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1

        # Find the user input text from events
        query = extract_user_input_text(events)
        if not query:
            return ResolvedContent(
                resolver_name=self.name,
                source_layer="memory",
                content=[],
            )

        # Search the store
        results = self._store.search(query, scope=self._scope, limit=self._limit)
        if not results:
            return ResolvedContent(
                resolver_name=self.name,
                source_layer="memory",
                content=[],
            )

        # Build context block
        memory_lines = [f"- {r.content}" for r in results]
        context_text = self._prefix + "\n".join(memory_lines)

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="memory",
            content=[TextBlock(text=context_text)],
        )

