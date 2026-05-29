"""StaticResolver: returns a fixed text block from config."""

from __future__ import annotations

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)


class StaticResolver:
    """Returns a fixed text block read from config at resolve-time."""

    name: str = "static"

    def __init__(self, config: ResolverConfig) -> None:
        if "text" not in config.config:
            raise ValueError("StaticResolver requires config['text'] to be set.")

        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "StaticResolver":
        return cls(config)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        text = self._config.config["text"]
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="static",
            content=[TextBlock(text=text)],
        )
