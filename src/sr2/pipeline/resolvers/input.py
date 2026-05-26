"""InputResolver: wraps user_input event data into a Message."""

from __future__ import annotations

from sr2.config.models import ResolverConfig
from sr2.models import Message
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="user_input", phase=EventPhase.STARTING)

_PHASE_MAP: dict[str, EventPhase] = {
    "starting": EventPhase.STARTING,
    "completed": EventPhase.COMPLETED,
    "failed": EventPhase.FAILED,
}


class InputResolver:
    """Wraps user_input event data (list[ContentBlock]) into a Message(role='user')."""

    name: str = "input"

    def __init__(self, config: ResolverConfig) -> None:
        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0

        if config.subscriptions:
            self.subscriptions: list[EventSubscription] = [
                EventSubscription(
                    event_name=sub.event,
                    phase=_PHASE_MAP[sub.phase] if sub.phase is not None else None,
                )
                for sub in config.subscriptions
            ]
        else:
            self.subscriptions = [_DEFAULT_SUBSCRIPTION]

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
