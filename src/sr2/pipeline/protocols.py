"""Runtime-checkable protocols for pipeline components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from sr2.config.models import ResolverConfig, TransformerConfig
from sr2.models import ContentBlock, Message, ToolDefinition
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventSubscription
from sr2.pipeline.models import ResolvedContent, TransformationResult

if TYPE_CHECKING:
    from sr2.config.models import ToolProviderConfig


# ---------------------------------------------------------------------------
# Pipeline component protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Resolver(Protocol):
    """Fires when subscribed events arrive, returns ResolvedContent."""

    subscriptions: list[EventSubscription]
    max_executions: int
    execution_count: int

    async def resolve(self, events: list[Event]) -> ResolvedContent: ...

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> Self: ...


@runtime_checkable
class Transformer(Protocol):
    """Fires when subscribed events arrive, transforms layer content."""

    subscriptions: list[EventSubscription]
    max_executions: int
    execution_count: int

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult: ...

    @classmethod
    def build(cls, config: TransformerConfig, deps: Dependencies) -> Self: ...


@runtime_checkable
class ToolProvider(Protocol):
    """Fires when subscribed events arrive, returns ToolDefinitions."""

    subscriptions: list[EventSubscription]
    max_executions: int
    execution_count: int

    async def provide(self, events: list[Event]) -> list[ToolDefinition]: ...

    @classmethod
    def build(cls, config: "ToolProviderConfig", deps: Dependencies) -> Self: ...


@runtime_checkable
class ToolSource(Protocol):
    """Harness-provided source of tool definitions.

    A typed optional dependency (mirrors ``memory_store`` and
    ``active_frame_provider``): the harness injects a concrete tool runtime
    via ``SR2(tool_source=...)``, and a ToolProvider reads it from
    ``deps.tool_source`` to surface definitions into the pipeline. Core owns
    this contract; it never imports the harness's concrete registry.
    """

    def to_sr2_definitions(self) -> list[ToolDefinition]: ...


@runtime_checkable
class Seedable(Protocol):
    """Resolvers (or other components) that can be seeded with conversation history.

    Used by Layer.seed() to call seed() on any component that exposes it,
    without hardcoding a dependency on SessionResolver.
    """

    def seed(self, messages: list[Message]) -> None: ...


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens in a list of content blocks. Injected into the engine."""

    def count(self, content: list[ContentBlock]) -> int: ...
