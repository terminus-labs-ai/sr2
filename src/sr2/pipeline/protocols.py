"""Runtime-checkable protocols for pipeline components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sr2.models import ContentBlock
from sr2.pipeline.events import Event, EventSubscription
from sr2.pipeline.models import ResolvedContent, TransformationResult


@runtime_checkable
class Resolver(Protocol):
    """Fires when subscribed events arrive, returns ResolvedContent."""

    subscriptions: list[EventSubscription]
    max_executions: int

    async def resolve(self, events: list[Event]) -> ResolvedContent: ...


@runtime_checkable
class Transformer(Protocol):
    """Fires when subscribed events arrive, transforms layer content."""

    subscriptions: list[EventSubscription]
    max_executions: int

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult: ...


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a list of content blocks. Injected into the engine."""

    def count(self, content: list[ContentBlock]) -> int: ...
