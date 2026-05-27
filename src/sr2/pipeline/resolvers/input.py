"""InputResolver: wraps user_input event data into a Message."""

from __future__ import annotations

from sr2.config.models import ResolverConfig
from sr2.models import Message
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="user_input", phase=EventPhase.STARTING)


class InputResolver:
    """Wraps user_input event data (list[ContentBlock]) into a Message(role='user')."""

    name: str = "input"

    def __init__(self, config: ResolverConfig) -> None:
        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "InputResolver":
        return cls(config)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1

        # Find the first user_input event with non-empty data
        for event in events:
            if event.name == "user_input" and event.data:
                return ResolvedContent(
                    resolver_name=self.name,
                    source_layer="input",
                    content=[Message(role="user", content=list(event.data))],
                )

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="input",
            content=[],
        )
