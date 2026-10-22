"""WorkingSetResolver: bounded working set read from the durable archive.

FR6 (sr2-94): replaces unbounded full-history replay with a configurable
recent verbatim window plus lifecycle-pinned content. The compiled working
set stays within the configured budget regardless of archive age or turn
count — no manual reset required.

FR7 residency rules implemented here:

  - lifecycle-pinned entries (system/identity invariants) are never evicted;
  - the current user input is pinned for the turn and never dropped;
  - recent dialogue (``turn_archive`` entries) is retained newest-first
    until the recent-window budget evicts it — the window is bounded
    independently of archive size;
  - everything else (retrieved, superseded, dormant, suppressed records)
    is excluded from the working set.

The archive is read through the ``ProvenanceStore`` protocol
(``get_session``); the store and the session id reach the resolver via its
``Dependencies`` (see :meth:`WorkingSetResolver.build`).

Continuity-disabled configurations (no ``continuity`` section) keep using
the legacy :class:`~sr2.pipeline.resolvers.session.SessionResolver`; this
resolver is opt-in via the ``working_set`` resolver type.
"""

from __future__ import annotations

from sr2.config.models import ResolverConfig
from sr2.models import ContentBlock, ContentBlockBase, Message
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.provenance import (
    TURN_ARCHIVE_LAYER,
    ContextLifecycle,
    Entry,
    ProvenanceStore,
)
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

_DEFAULT_SUBSCRIPTIONS = [
    EventSubscription(event_name="user_input", phase=EventPhase.STARTING),
]

_RECENT_TOKEN_BUDGET_KEY = "recent_token_budget"
_HARD_TOKEN_BUDGET_KEY = "hard_token_budget"


class WorkingSetResolver:
    """Assembles a token-bounded working set per turn from the archive.

    Each ``resolve()`` firing (one per turn, driven by the ``user_input``
    event):

      1. Reads the session's ``turn_archive`` entries through the
         provenance store — statelessly, so boundedness comes from the
         selector itself and never needs a reset.
      2. Collects lifecycle-PINNED entries; they are never evicted.
      3. Walks the remaining archive newest-first, keeping as much recent
         dialogue as the recent-window budget allows. The current turn's
         input is served by its freshly archived entry when present;
         otherwise the live ``user_input`` event is materialized so the
         current input is always pinned into the output.
      4. Applies the optional hard token budget as a final cap (pinned
         entries first, then newest recent items), so total output stays
         within it modulo the pinned/live-content floor.
    """

    name: str = "working_set"

    def __init__(
        self,
        config: ResolverConfig,
        *,
        provenance_store: ProvenanceStore,
        session_id: str,
        recent_token_budget: int,
        hard_token_budget: int | None = None,
        token_counter: CharacterTokenCounter | None = None,
    ) -> None:
        if recent_token_budget <= 0:
            raise ValueError(
                f"{_RECENT_TOKEN_BUDGET_KEY} must be positive; "
                f"got {recent_token_budget}"
            )
        self._config = config
        self._store = provenance_store
        self._session_id = session_id
        self._recent_budget = recent_token_budget
        self._hard_budget = hard_token_budget
        self._counter = token_counter or CharacterTokenCounter()
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, _DEFAULT_SUBSCRIPTIONS
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> "WorkingSetResolver":
        """Construct from resolver config and the Dependencies container.

        Reads ``recent_token_budget`` (required, positive) and
        ``hard_token_budget`` (optional) from ``config.config``; the
        provenance store and session id are read off ``deps``.
        """
        cfg = dict(config.config or {})
        raw_recent = cfg.get(_RECENT_TOKEN_BUDGET_KEY)
        if raw_recent is None:
            raise ValueError(
                f"working_set resolver requires '{_RECENT_TOKEN_BUDGET_KEY}' "
                "in its config"
            )
        raw_hard = cfg.get(_HARD_TOKEN_BUDGET_KEY)
        return cls(
            config,
            provenance_store=deps.provenance_store,
            session_id=deps.session_id,
            recent_token_budget=int(raw_recent),
            hard_token_budget=int(raw_hard) if raw_hard is not None else None,
        )

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1

        entries = await self._store.get_session(
            self._session_id, layer=TURN_ARCHIVE_LAYER
        )
        entries.sort(key=self._order_key)

        pinned = [e for e in entries if e.lifecycle is ContextLifecycle.PINNED]
        window_pool = [
            e
            for e in entries
            if e.lifecycle is not ContextLifecycle.PINNED
            and e.lifecycle is not ContextLifecycle.RETRIEVED
        ]

        current_blocks = self._current_input_blocks(events)
        current_text = self._blocks_text(current_blocks)
        # Recent window: newest-first until the window budget evicts the rest.
        window: list[Entry] = []
        window_tokens = 0
        for entry in reversed(window_pool):
            cost = self._entry_tokens(entry)
            if window_tokens + cost > self._recent_budget:
                continue
            window.append(entry)
            window_tokens += cost

        # The current input is pinned: if the window already carries an
        # entry with the live text (e.g. the freshly archived current-turn
        # message), that entry serves as the live input; otherwise the live
        # event is materialized so the current input is never dropped.
        live_covered = bool(current_text) and any(
            e.content is not None and self._entry_text(e) == current_text
            for e in window
        )
        live_blocks = current_blocks if (current_text and not live_covered) else []
        live_cost = self._counter.count(live_blocks) if live_blocks else 0

        # Budget the working set. Total output — recent window plus the
        # live current input — stays within the recent-window budget; the
        # optional hard budget tightens it further. Pinned entries are
        # never dropped, so they reserve space ahead of the window.
        pinned_tokens = sum(self._entry_tokens(e) for e in pinned)
        cap = self._recent_budget - pinned_tokens - live_cost
        if self._hard_budget is not None:
            cap = min(cap, self._hard_budget - pinned_tokens - live_cost)
        kept: list[Entry] = []
        remaining = max(cap, 0)
        for entry in reversed(window):
            cost = self._entry_tokens(entry)
            if cost > remaining:
                continue
            kept.append(entry)
            remaining -= cost

        content: list[ContentBlock | Message] = [
            e.content for e in (*pinned, *reversed(kept))
        ]
        if live_blocks:
            content.append(Message(role="user", content=list(live_blocks)))

        return ResolvedContent(
            resolver_name=self.name,
            source_layer=TURN_ARCHIVE_LAYER,
            content=content,
            token_count=self._count_content(content),
        )

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _order_key(entry: Entry) -> tuple:
        """Archive order: created_at first, ULID id as deterministic
        tie-breaker (ULIDs are sortable)."""
        return (entry.created_at, entry.id)

    def _entry_tokens(self, entry: Entry) -> int:
        content = entry.content
        if isinstance(content, Message):
            return self._counter.count(list(content.content))
        return self._counter.count([content])

    def _entry_text(self, entry: Entry) -> str:
        content = entry.content
        blocks = list(content.content) if isinstance(content, Message) else [content]
        return "".join(
            b.text
            for b in blocks
            if isinstance(b, ContentBlockBase) and hasattr(b, "text")
        )

    @staticmethod
    def _current_input_blocks(events: list[Event]) -> list[ContentBlock]:
        for event in events:
            if event.name == "user_input" and isinstance(event.data, list):
                return [b for b in event.data if isinstance(b, ContentBlockBase)]
        return []

    @staticmethod
    def _blocks_text(blocks: list[ContentBlock]) -> str:
        return "".join(
            b.text
            for b in blocks
            if isinstance(b, ContentBlockBase) and hasattr(b, "text")
        )

    def _count_content(self, content: list[ContentBlock | Message]) -> int:
        total = 0
        for item in content:
            if isinstance(item, Message):
                total += self._counter.count(list(item.content))
            else:
                total += self._counter.count([item])
        return total
