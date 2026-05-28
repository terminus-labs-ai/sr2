"""EventPayloadResolver: surfaces event.data ContentBlocks as provenance entries."""

from __future__ import annotations

from datetime import datetime, timezone

from ulid import ULID

from sr2.config.models import ResolverConfig
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.provenance import Entry, EntryOrigin
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

_ORIGIN = EntryOrigin(kind="resolver", name="event_payload")


class EventPayloadResolver:
    """Returns ContentBlocks emitted in matching event payloads as provenance entries."""

    name: str = "event_payload"

    def __init__(self, config: ResolverConfig, session_id: str = "") -> None:
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self._session_id = session_id

        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, []
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "EventPayloadResolver":
        return cls(config, session_id=deps.session_id)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1

        subscribed_names = {sub.event_name for sub in self.subscriptions}
        entries: list[Entry] = []

        for event in events:
            if event.name not in subscribed_names:
                continue
            data = event.data
            if not isinstance(data, list) or not data:
                continue
            for block in data:
                entries.append(
                    Entry(
                        id=str(ULID()),
                        content=block,
                        sources=(),
                        origin=_ORIGIN,
                        layer="event_payload",
                        session_id=self._session_id,
                        created_at=datetime.now(tz=timezone.utc),
                    )
                )

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="event_payload",
            entries=entries,
        )
