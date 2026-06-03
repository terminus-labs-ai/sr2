"""Runtime-checkable protocols for pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from sr2.config.models import ResolverConfig, TransformerConfig
from sr2.models import ContentBlock, ToolDefinition
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventSubscription
from sr2.pipeline.models import ResolvedContent, TransformationResult

if TYPE_CHECKING:
    from sr2.config.models import ToolProviderConfig


# ---------------------------------------------------------------------------
# Component protocol — unified dispatch contract
# ---------------------------------------------------------------------------


@dataclass
class ComponentResult:
    """Return type for Component.run()."""

    component_name: str
    source_layer: str
    content: list | None = None


@runtime_checkable
class Component(Protocol):
    """Unified protocol for all pipeline components.

    ``isinstance(obj, Component)`` checks for the four shared attributes:
    ``name``, ``subscriptions``, ``max_executions``, ``execution_count``.

    Layer dispatches custom components (those in ``Layer.components``) by
    calling ``comp.run(layer_view, events)`` directly.  Built-in resolvers,
    transformers, and tool providers also satisfy this protocol via those four
    shared attributes and are dispatched through the existing
    ``_fire_component`` path for backward compatibility.

    Custom components placed in ``Layer.components`` MUST implement ``run()``.
    Layer calls it directly (not via isinstance dispatch) so it is not part of
    the protocol's runtime-checkable surface.
    """

    name: str
    subscriptions: list[EventSubscription]
    max_executions: int
    execution_count: int


# ---------------------------------------------------------------------------
# Built-in component protocols (unchanged from original)
# ---------------------------------------------------------------------------


@runtime_checkable
class Resolver(Protocol):
    """Fires when subscribed events arrive, returns ResolvedContent."""

    subscriptions: list[EventSubscription]
    max_executions: int

    async def resolve(self, events: list[Event]) -> ResolvedContent: ...

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> Self: ...


@runtime_checkable
class Transformer(Protocol):
    """Fires when subscribed events arrive, transforms layer content."""

    subscriptions: list[EventSubscription]
    max_executions: int

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
class TokenCounter(Protocol):
    """Counts tokens in a list of content blocks. Injected into the engine."""

    def count(self, content: list[ContentBlock]) -> int: ...
