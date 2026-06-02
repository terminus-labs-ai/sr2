"""Regression tests for obsidian-75d: duplicate user_input when a layer has
multiple resolvers subscribing to the same event.

The conversation layer wires both SessionResolver and InputResolver, which BOTH
subscribe to ``user_input``. Engine._wire_layers must register the layer's
``handle_event`` such that a single queued event is buffered ONCE — not once per
matching subscription. Otherwise the layer buffers N copies of every event and
accumulating resolvers (e.g. SessionResolver) capture the user turn N times,
duplicating it in the compiled request.

These are behavioural tests: they assert what reaches the compiled request /
the resolver, not how wiring is implemented.
"""

from __future__ import annotations

import pytest

from conftest import run_engine

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.engine import PipelineEngine
from sr2.pipeline.events import Event, EventSubscription
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.resolvers.input import InputResolver
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.pipeline.token_counting import CharacterTokenCounter


def _conversation_layer() -> Layer:
    """A layer mirroring the real conversation layer: session + input resolvers,
    both of which subscribe to ``user_input``."""
    session = SessionResolver(ResolverConfig(type="session"))
    inp = InputResolver(ResolverConfig(type="input"))
    return Layer(
        name="conversation",
        target=CompilationTarget.MESSAGES,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[session, inp],
        transformers=[],
        token_counter=CharacterTokenCounter(),
    )


def _count_user_text(request, needle: str) -> int:
    """Count user-role messages whose content contains ``needle``."""
    count = 0
    for msg in (getattr(request, "messages", None) or []):
        if getattr(msg, "role", None) != "user":
            continue
        for block in msg.content:
            if isinstance(block, TextBlock) and needle in block.text:
                count += 1
    return count


@pytest.mark.asyncio
async def test_single_user_input_appears_once_in_compiled_request():
    """A single user message must appear exactly once in the compiled request,
    even though both session and input resolvers subscribe to user_input."""
    layer = _conversation_layer()
    engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())

    result = await run_engine(engine, user_input=[TextBlock(text="ping")])

    assert _count_user_text(result.request, "ping") == 1, (
        "A single user message must appear exactly once in the compiled request. "
        "Duplication means the layer's handle_event was registered once per "
        "resolver subscription, so user_input was buffered multiple times."
    )


@pytest.mark.asyncio
async def test_prior_user_input_appears_once_on_next_turn():
    """The user-visible symptom: a user turn captured into session history must
    appear exactly once as prior history on the following turn. Double-buffered
    user_input makes SessionResolver accumulate N copies, which then surface in
    the next turn's compiled request."""
    layer = _conversation_layer()
    engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())

    await run_engine(engine, user_input=[TextBlock(text="first")])
    result2 = await run_engine(engine, user_input=[TextBlock(text="second")])

    assert _count_user_text(result2.request, "first") == 1, (
        "The first turn's user message must appear exactly once as prior history "
        "on the second turn. More than one copy means user_input was buffered "
        "multiple times into session history."
    )


class _CapturingResolver:
    """Records every event batch passed to resolve()."""

    def __init__(self, name: str, subscriptions: list[EventSubscription]):
        self.name = name
        self.subscriptions = subscriptions
        self.max_executions = 10
        self.execution_count = 0
        self.captured_events: list[list[Event]] = []

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        self.captured_events.append(list(events))
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="test",
            content=[TextBlock(text=f"resolved by {self.name}")],
        )


@pytest.mark.asyncio
async def test_event_buffered_once_when_two_resolvers_share_subscription():
    """When two resolvers in one layer subscribe to the same event, a single
    occurrence of that event must be delivered to each resolver exactly once —
    not once per subscription."""
    sub = EventSubscription(event_name="turn_start")
    r1 = _CapturingResolver("r1", [sub])
    r2 = _CapturingResolver("r2", [sub])

    layer = Layer(
        name="shared_sub_layer",
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[r1, r2],
        transformers=[],
        token_counter=CharacterTokenCounter(),
    )
    engine = PipelineEngine(layers=[layer], token_counter=CharacterTokenCounter())

    # start_turn emits exactly one turn_start event.
    await engine.start_turn(turn_seq=0)

    for resolver in (r1, r2):
        delivered = sum(
            1
            for batch in resolver.captured_events
            for ev in batch
            if ev.name == "turn_start"
        )
        assert delivered == 1, (
            f"{resolver.name} received turn_start {delivered} times; expected 1. "
            "The layer buffered the event once per subscription instead of once."
        )
