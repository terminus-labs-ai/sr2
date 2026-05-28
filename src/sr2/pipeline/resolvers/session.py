"""SessionResolver: accumulates conversation history across turns.

Returns PRIOR history on each resolve() call, then captures new events
(user_input, assistant_response, tool_use_emitted, tool_result_received)
for subsequent calls.
"""

from __future__ import annotations

import sys

from sr2.config.models import ResolverConfig
from sr2.models import ContentBlock, Message, ToolResultBlock, ToolUseBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionResponse

_DEFAULT_SUBSCRIPTIONS = [
    EventSubscription(event_name="user_input", phase=EventPhase.STARTING),
    EventSubscription(event_name="assistant_response", phase=EventPhase.COMPLETED),
    EventSubscription(event_name="tool_use_emitted", phase=EventPhase.COMPLETED),
    EventSubscription(event_name="tool_result_received", phase=EventPhase.COMPLETED),
]


class SessionResolver:
    """Accumulates user/assistant messages across turns.

    Acts as an accumulator — max_executions is set to sys.maxsize so it fires
    on every matching event throughout the turn (including mid-turn tool loop
    iterations).

    Each resolve() call:
      1. Snapshots the current history as output (copy).
      2. Captures matching events into internal history for next turn.
      3. Increments execution_count.

    Supported events:
      - user_input: appends a user Message from event.data (list[ContentBlock])
      - assistant_response: appends an assistant Message from CompletionResponse.content
      - tool_use_emitted: appends an assistant Message containing tool_use blocks
      - tool_result_received: appends a user Message containing tool_result blocks
    """

    name: str = "session"

    def __init__(self, config: ResolverConfig) -> None:
        self._config = config
        self.max_executions: int = sys.maxsize
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

        # Determine if this firing is purely for tool-loop event capture.
        # Tool-loop fires (tool_use_emitted, tool_result_received) only append
        # to history — they must NOT re-emit the history snapshot, which would
        # duplicate content already placed in the layer on the initial fire.
        _tool_loop_names = {"tool_use_emitted", "tool_result_received"}
        event_names = {e.name for e in events}
        is_tool_loop_only = bool(events) and event_names.issubset(_tool_loop_names)

        # Snapshot prior history for output (copy so mutations don't corrupt state).
        # Tool-loop firings return empty — history was already placed in the layer
        # on the initial (user_input / turn_start) fire.
        output: list[Message] = [] if is_tool_loop_only else list(self._history)

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

            elif event.name == "tool_use_emitted":
                tool_blocks: list[ContentBlock] | None = event.data
                if tool_blocks:
                    self._history.append(
                        Message(role="assistant", content=list(tool_blocks))
                    )

            elif event.name == "tool_result_received":
                result_blocks: list[ToolResultBlock] | None = event.data
                if result_blocks:
                    self._history.append(
                        Message(role="user", content=list(result_blocks))
                    )

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="session",
            content=output,
        )
