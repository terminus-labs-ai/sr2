"""SessionResolver: accumulates conversation history across turns.

Returns PRIOR history on each resolve() call, then captures new events
(user_input, assistant_response) for subsequent calls.
"""

from __future__ import annotations

from sr2.config.models import ResolverConfig
from sr2.models import ContentBlock, Message
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionResponse

_DEFAULT_SUBSCRIPTIONS = [
    EventSubscription(event_name="user_input", phase=EventPhase.STARTING),
    EventSubscription(event_name="assistant_response", phase=EventPhase.COMPLETED),
]


class SessionResolver:
    """Accumulates user/assistant messages across turns.

    Each resolve() call:
      1. Snapshots the current history as output (copy).
      2. Captures matching events into internal history for next turn.
      3. Increments execution_count.
    """

    name: str = "session"

    def __init__(self, config: ResolverConfig) -> None:
        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self._history: list[Message] = []
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, _DEFAULT_SUBSCRIPTIONS
        )

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "SessionResolver":
        return cls(config)

    def seed(self, messages: list[Message]) -> None:
        """Pre-populate conversation history.

        Replaces any existing history with independent copies of *messages*.
        Call before the first turn to inject prior context (e.g. restored sessions).
        """
        self._history = [m.model_copy() for m in messages]

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1

        # Snapshot prior history (copy so mutations don't corrupt state)
        output = list(self._history)

        # Capture new events into history for subsequent calls
        for event in events:
            if event.name == "user_input":
                data: list[ContentBlock] | None = event.data
                if data:
                    self._history.append(Message(role="user", content=list(data)))

            elif event.name == "assistant_response":
                response: CompletionResponse | None = event.data
                if response is not None:
                    self._history.append(
                        Message(role="assistant", content=list(response.content))
                    )

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="session",
            content=output,
        )
