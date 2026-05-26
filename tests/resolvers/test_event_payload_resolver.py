"""Tests for EventPayloadResolver.

Covers:
  FR6:  EventPayloadResolver implements the Resolver protocol.
        - subscriptions: configurable via YAML (ResolverConfig.subscriptions).
        - resolve(events) -> ResolvedContent: for each matching event where
          event.data is a non-empty list of ContentBlocks, add those blocks
          to the returned ResolvedContent as provenance entries.
        - If event.data is None or empty, returns empty ResolvedContent.
  FR7:  EventPayloadResolver is registered in _RESOLVER_FACTORIES as "event_payload".

Acceptance criteria exercised:
  AC7:  EventPayloadResolver subscribed to summarization_complete: after the
        event fires, resolver adds the TextBlock from event.data to its layer.
  AC8:  EventPayloadResolver with event.data = None: returns empty
        ResolvedContent, adds nothing.
  AC5 (scenario): Future transformer emits compaction_complete with content in
        event.data. An EventPayloadResolver subscribed to compaction_complete
        picks it up with zero new code.
"""

import pytest

from sr2.config.models import EventSubscriptionConfig, ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.provenance import Entry, EntryOrigin
from sr2.pipeline.resolvers.event_payload import EventPayloadResolver
from sr2.orchestrator import _RESOLVERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    subscriptions: list[EventSubscriptionConfig] | None = None,
    **kwargs,
) -> ResolverConfig:
    """Build a minimal ResolverConfig for EventPayloadResolver."""
    return ResolverConfig(
        type="event_payload",
        subscriptions=subscriptions or [],
        **kwargs,
    )


def make_event(
    name: str,
    data=None,
    phase: EventPhase = EventPhase.COMPLETED,
    source_layer: str = "pipeline",
) -> Event:
    return Event(name=name, phase=phase, source_layer=source_layer, data=data)


def make_summarization_event(data=None) -> Event:
    return make_event("summarization_complete", data=data)


def make_compaction_event(data=None) -> Event:
    return make_event("compaction_complete", data=data)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


class TestEventPayloadResolverConstruction:
    def test_constructs_with_subscriptions(self):
        """FR6: Constructs without error when subscriptions are provided."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        assert resolver is not None

    def test_constructs_with_empty_subscriptions(self):
        """FR6: Constructs without error when subscriptions list is empty."""
        resolver = EventPayloadResolver(make_config())
        assert resolver is not None

    def test_execution_count_starts_at_zero(self):
        """FR6: Fresh resolver has execution_count == 0."""
        resolver = EventPayloadResolver(make_config())
        assert resolver.execution_count == 0

    def test_max_executions_default_from_config(self):
        """FR6: max_executions defaults to 1 from ResolverConfig."""
        resolver = EventPayloadResolver(make_config())
        assert resolver.max_executions == 1

    def test_max_executions_reads_from_config(self):
        """FR6: max_executions is read from ResolverConfig.max_executions."""
        cfg = make_config(max_executions=5)
        resolver = EventPayloadResolver(cfg)
        assert resolver.max_executions == 5


# ---------------------------------------------------------------------------
# 2. Protocol conformance (FR6)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverProtocolConformance:
    def test_isinstance_resolver(self):
        """FR6: EventPayloadResolver must satisfy the Resolver protocol."""
        resolver = EventPayloadResolver(make_config())
        assert isinstance(resolver, Resolver)

    def test_has_subscriptions_attribute(self):
        """FR6: Must expose a subscriptions list."""
        resolver = EventPayloadResolver(make_config())
        assert hasattr(resolver, "subscriptions")
        assert isinstance(resolver.subscriptions, list)

    def test_has_max_executions_attribute(self):
        """FR6: Must expose max_executions (int)."""
        resolver = EventPayloadResolver(make_config())
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_has_resolve_method(self):
        """FR6: Must expose an async resolve() method."""
        resolver = EventPayloadResolver(make_config())
        assert callable(getattr(resolver, "resolve", None))


# ---------------------------------------------------------------------------
# 3. Subscriptions from config (FR6)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverSubscriptions:
    def test_subscriptions_set_from_config(self):
        """FR6: subscriptions are driven by ResolverConfig.subscriptions."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="summarization_complete", phase="completed")
            ]
        )
        resolver = EventPayloadResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "summarization_complete" in names

    def test_subscription_phase_is_respected(self):
        """FR6: Phase from config subscription is preserved in EventSubscription."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="my_event", phase="completed")
            ]
        )
        resolver = EventPayloadResolver(cfg)
        matching = [s for s in resolver.subscriptions if s.event_name == "my_event"]
        assert matching
        assert matching[0].phase == EventPhase.COMPLETED

    def test_subscription_none_phase_is_preserved(self):
        """FR6: Phase=None in config is preserved (matches any phase)."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="my_event", phase=None)
            ]
        )
        resolver = EventPayloadResolver(cfg)
        matching = [s for s in resolver.subscriptions if s.event_name == "my_event"]
        assert matching
        assert matching[0].phase is None

    def test_multiple_subscriptions_all_registered(self):
        """FR6: Multiple subscriptions from config are all registered."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="summarization_complete"),
                EventSubscriptionConfig(event="compaction_complete"),
            ]
        )
        resolver = EventPayloadResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "summarization_complete" in names
        assert "compaction_complete" in names

    def test_empty_subscriptions_produces_empty_subscriptions_list(self):
        """FR6: No default subscriptions — empty config means empty list."""
        resolver = EventPayloadResolver(make_config())
        # No default subscriptions: the resolver simply never matches anything.
        assert resolver.subscriptions == []


# ---------------------------------------------------------------------------
# 4. resolve() — happy path (AC7)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverHappyPath:
    @pytest.mark.asyncio
    async def test_resolve_returns_resolved_content(self):
        """FR6 / AC7: resolve() returns a ResolvedContent instance."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="summary here")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert isinstance(result, ResolvedContent)

    @pytest.mark.asyncio
    async def test_resolve_adds_text_block_as_entry(self):
        """AC7: TextBlock from event.data appears in result.entries."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="summary content")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert len(result.entries) == 1
        assert result.entries[0].content == block

    @pytest.mark.asyncio
    async def test_entry_has_resolver_origin(self):
        """FR6 / AC7: Provenance entries have kind='resolver'."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="data")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert len(result.entries) == 1
        assert result.entries[0].origin.kind == "resolver"

    @pytest.mark.asyncio
    async def test_entry_origin_name_is_event_payload(self):
        """FR6 / AC7: Entry origin name is 'event_payload'."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="data")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert result.entries[0].origin.name == "event_payload"

    @pytest.mark.asyncio
    async def test_entry_sources_is_empty(self):
        """FR6 / AC7: Resolver-origin entries must have empty sources (genesis entries)."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="data")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert result.entries[0].sources == ()

    @pytest.mark.asyncio
    async def test_resolver_name_in_result(self):
        """FR6: ResolvedContent.resolver_name must be set (non-empty string)."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="hi")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert isinstance(result.resolver_name, str)
        assert result.resolver_name

    @pytest.mark.asyncio
    async def test_source_layer_in_result(self):
        """FR6: ResolvedContent.source_layer must be set (non-empty string)."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block = TextBlock(text="hi")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert isinstance(result.source_layer, str)
        assert result.source_layer


# ---------------------------------------------------------------------------
# 5. resolve() — empty / None data (AC8)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverEmptyData:
    @pytest.mark.asyncio
    async def test_none_data_returns_empty_resolved_content(self):
        """AC8: event.data = None => empty ResolvedContent (no entries)."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        event = make_summarization_event(data=None)
        result = await resolver.resolve([event])
        assert isinstance(result, ResolvedContent)
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_empty_list_data_returns_empty_resolved_content(self):
        """AC8: event.data = [] => empty ResolvedContent (no entries)."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        event = make_summarization_event(data=[])
        result = await resolver.resolve([event])
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_no_matching_events_returns_empty_resolved_content(self):
        """FR6: Events that don't match subscriptions produce no entries."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        # Pass an event that doesn't match the subscription
        unrelated = make_event("turn_start", data=[TextBlock(text="ignored")])
        result = await resolver.resolve([unrelated])
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_empty_events_list_returns_empty_resolved_content(self):
        """FR6: Empty events list => empty ResolvedContent."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        result = await resolver.resolve([])
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_no_subscriptions_always_returns_empty(self):
        """FR6: Resolver with no subscriptions never matches; always returns empty."""
        resolver = EventPayloadResolver(make_config())
        event = make_summarization_event(data=[TextBlock(text="ignored")])
        result = await resolver.resolve([event])
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_non_list_data_is_treated_as_no_op(self):
        """FR6: event.data that is not a list is ignored (treated as no-op)."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        # Scalar string is not a list[ContentBlock]
        event = make_summarization_event(data="not a list")
        result = await resolver.resolve([event])
        assert result.entries == []


# ---------------------------------------------------------------------------
# 6. Multiple events (FR6 — collect all matching blocks)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverMultipleEvents:
    @pytest.mark.asyncio
    async def test_multiple_matching_events_all_collected(self):
        """FR6: If multiple matching events fire, all their data blocks are collected."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block_a = TextBlock(text="first summary")
        block_b = TextBlock(text="second summary")
        events = [
            make_summarization_event(data=[block_a]),
            make_summarization_event(data=[block_b]),
        ]
        result = await resolver.resolve(events)
        assert len(result.entries) == 2
        entry_contents = [e.content for e in result.entries]
        assert block_a in entry_contents
        assert block_b in entry_contents

    @pytest.mark.asyncio
    async def test_multiple_blocks_in_single_event(self):
        """FR6: Multiple ContentBlocks within a single event.data are all added."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        block_a = TextBlock(text="block one")
        block_b = TextBlock(text="block two")
        event = make_summarization_event(data=[block_a, block_b])
        result = await resolver.resolve([event])
        assert len(result.entries) == 2

    @pytest.mark.asyncio
    async def test_mixed_matching_and_non_matching_events(self):
        """FR6: Only events matching subscriptions contribute entries."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        matching_block = TextBlock(text="from summary")
        events = [
            make_event("turn_start", data=[TextBlock(text="ignored")]),
            make_summarization_event(data=[matching_block]),
            make_event("some_other_event", data=[TextBlock(text="also ignored")]),
        ]
        result = await resolver.resolve(events)
        assert len(result.entries) == 1
        assert result.entries[0].content == matching_block

    @pytest.mark.asyncio
    async def test_mixed_none_and_valid_data_events(self):
        """FR6: None-data events are skipped; valid events still contribute."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        valid_block = TextBlock(text="valid content")
        events = [
            make_summarization_event(data=None),
            make_summarization_event(data=[valid_block]),
            make_summarization_event(data=[]),
        ]
        result = await resolver.resolve(events)
        assert len(result.entries) == 1
        assert result.entries[0].content == valid_block


# ---------------------------------------------------------------------------
# 7. AC5 scenario: compaction_complete (extensibility proof)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverCompactionScenario:
    @pytest.mark.asyncio
    async def test_compaction_complete_subscription_picks_up_data(self):
        """AC5: EventPayloadResolver subscribed to compaction_complete picks up
        data from that event with zero new code — just a different subscription."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="compaction_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        compacted_block = TextBlock(text="compacted content")
        event = make_compaction_event(data=[compacted_block])
        result = await resolver.resolve([event])
        assert len(result.entries) == 1
        assert result.entries[0].content == compacted_block

    @pytest.mark.asyncio
    async def test_compaction_resolver_does_not_pick_up_summarization_events(self):
        """AC5: A compaction-subscribed resolver ignores summarization events."""
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="compaction_complete")]
        )
        resolver = EventPayloadResolver(cfg)
        summary_event = make_summarization_event(data=[TextBlock(text="summary")])
        result = await resolver.resolve([summary_event])
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_multi_subscription_catches_both_event_types(self):
        """AC5: Subscribed to both compaction and summarization catches both."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="compaction_complete"),
                EventSubscriptionConfig(event="summarization_complete"),
            ]
        )
        resolver = EventPayloadResolver(cfg)
        compacted = TextBlock(text="compacted")
        summarized = TextBlock(text="summarized")
        events = [
            make_compaction_event(data=[compacted]),
            make_summarization_event(data=[summarized]),
        ]
        result = await resolver.resolve(events)
        assert len(result.entries) == 2


# ---------------------------------------------------------------------------
# 8. execution_count
# ---------------------------------------------------------------------------


class TestEventPayloadResolverExecutionCount:
    @pytest.mark.asyncio
    async def test_execution_count_increments_on_resolve(self):
        """FR6: execution_count increments once per resolve() call."""
        resolver = EventPayloadResolver(make_config())
        await resolver.resolve([])
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_accumulates(self):
        """FR6: Multiple calls accumulate in execution_count."""
        resolver = EventPayloadResolver(make_config())
        await resolver.resolve([])
        await resolver.resolve([])
        await resolver.resolve([])
        assert resolver.execution_count == 3

    @pytest.mark.asyncio
    async def test_execution_count_increments_even_with_no_matches(self):
        """FR6: execution_count increments regardless of whether events matched."""
        resolver = EventPayloadResolver(make_config())
        await resolver.resolve([make_summarization_event(data=None)])
        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 10. build() classmethod
# ---------------------------------------------------------------------------


class TestEventPayloadResolverBuild:
    def test_build_returns_event_payload_resolver_instance(self):
        """build() must return an EventPayloadResolver instance."""
        config = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        result = EventPayloadResolver.build(config, Dependencies())
        assert isinstance(result, EventPayloadResolver)

    def test_build_with_populated_deps_also_works(self):
        """build() must accept and ignore a non-empty Dependencies container."""
        config = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        deps = Dependencies(llm={"default": lambda *a, **kw: None})
        result = EventPayloadResolver.build(config, deps)
        assert isinstance(result, EventPayloadResolver)

    def test_build_result_satisfies_resolver_protocol(self):
        """Instance returned by build() must satisfy isinstance(x, Resolver)."""
        config = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        result = EventPayloadResolver.build(config, Dependencies())
        assert isinstance(result, Resolver)

    def test_build_state_matches_direct_construction(self):
        """build() must produce an instance with the same observable state
        as one constructed via EventPayloadResolver(config) directly."""
        config = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")],
            max_executions=4,
        )
        via_build = EventPayloadResolver.build(config, Dependencies())
        via_init = EventPayloadResolver(config)
        assert via_build.max_executions == via_init.max_executions
        assert via_build.execution_count == via_init.execution_count
        # Subscriptions should mirror what the config specified
        build_event_names = {s.event_name for s in via_build.subscriptions}
        init_event_names = {s.event_name for s in via_init.subscriptions}
        assert build_event_names == init_event_names


# ---------------------------------------------------------------------------
# 9. Registration in _RESOLVER_FACTORIES (FR7)
# ---------------------------------------------------------------------------


class TestEventPayloadResolverRegistration:
    def test_registered_as_event_payload(self):
        """FR7: 'event_payload' key must be resolvable via _RESOLVERS registry."""
        assert _RESOLVERS.get("event_payload") is not None

    def test_factory_produces_event_payload_resolver_instance(self):
        """FR7: Factory for 'event_payload' produces an EventPayloadResolver."""
        from sr2.pipeline.dependencies import Dependencies

        cls = _RESOLVERS.get("event_payload")
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        instance = cls.build(cfg, Dependencies())
        assert isinstance(instance, EventPayloadResolver)

    def test_factory_produced_instance_satisfies_resolver_protocol(self):
        """FR7: Instance from registry satisfies the Resolver protocol."""
        from sr2.pipeline.dependencies import Dependencies

        cls = _RESOLVERS.get("event_payload")
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        instance = cls.build(cfg, Dependencies())
        assert isinstance(instance, Resolver)

    @pytest.mark.asyncio
    async def test_factory_produced_instance_is_functional(self):
        """FR7: Instance from registry can resolve events end-to-end."""
        from sr2.pipeline.dependencies import Dependencies

        cls = _RESOLVERS.get("event_payload")
        cfg = make_config(
            subscriptions=[EventSubscriptionConfig(event="summarization_complete")]
        )
        resolver = cls.build(cfg, Dependencies())
        block = TextBlock(text="from factory")
        event = make_summarization_event(data=[block])
        result = await resolver.resolve([event])
        assert len(result.entries) == 1
        assert result.entries[0].content == block
