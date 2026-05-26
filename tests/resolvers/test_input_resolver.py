"""Tests for InputResolver.

Covers:
  - Construction from ResolverConfig with default subscriptions (user_input event)
  - Construction with custom subscriptions overriding defaults
  - resolve() wraps user_input event data into Message(role="user")
  - resolve() wraps multiple content blocks into a single Message
  - resolve() ignores non-user_input events (returns empty ResolvedContent)
  - execution_count increments on each resolve() call
  - Protocol conformance with Resolver protocol
"""

import pytest

from sr2.config.models import EventSubscriptionConfig, ResolverConfig
from sr2.models import Message, TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.resolvers.input import InputResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> ResolverConfig:
    """Build a minimal ResolverConfig for InputResolver."""
    return ResolverConfig(type="input", **kwargs)


def make_user_input_event(data=None) -> Event:
    """Create a user_input event with given data."""
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="core",
        data=data,
    )


def make_other_event(name: str = "turn_start") -> Event:
    """Create a non-user_input event."""
    return Event(
        name=name,
        phase=EventPhase.STARTING,
        source_layer="core",
    )


# ---------------------------------------------------------------------------
# 1. Construction — default subscriptions
# ---------------------------------------------------------------------------


class TestInputResolverConstruction:
    def test_constructs_with_valid_config(self):
        """InputResolver builds without error from a basic config."""
        resolver = InputResolver(make_config())
        assert resolver is not None

    def test_default_subscription_is_user_input(self):
        """Default subscription must be 'user_input'."""
        resolver = InputResolver(make_config())
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names

    def test_name_is_input(self):
        """Resolver name attribute must be 'input'."""
        resolver = InputResolver(make_config())
        assert resolver.name == "input"

    def test_default_max_executions_is_one(self):
        """Default max_executions from ResolverConfig is 1."""
        resolver = InputResolver(make_config())
        assert resolver.max_executions == 1

    def test_max_executions_reads_from_config(self):
        """max_executions is read from ResolverConfig, not hardcoded."""
        resolver = InputResolver(make_config(max_executions=10))
        assert resolver.max_executions == 10


# ---------------------------------------------------------------------------
# 2. Construction — custom subscriptions override
# ---------------------------------------------------------------------------


class TestInputResolverSubscriptionOverride:
    def test_custom_subscriptions_replace_defaults(self):
        """Non-empty config.subscriptions override the user_input default."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="custom_event", phase="completed"),
            ],
        )
        resolver = InputResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "custom_event" in names
        assert "user_input" not in names

    def test_empty_config_subscriptions_falls_back_to_default(self):
        """Empty config.subscriptions => use user_input default."""
        cfg = make_config(subscriptions=[])
        resolver = InputResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names


# ---------------------------------------------------------------------------
# 3. resolve() — wraps user input into Message
# ---------------------------------------------------------------------------


class TestInputResolverResolve:
    @pytest.mark.asyncio
    async def test_resolve_wraps_single_text_block_into_message(self):
        """Given a user_input event with [TextBlock], resolve returns
        ResolvedContent with a single Message(role='user')."""
        resolver = InputResolver(make_config())
        content_blocks = [TextBlock(text="hello")]
        event = make_user_input_event(data=content_blocks)

        result = await resolver.resolve([event])

        assert isinstance(result, ResolvedContent)
        assert len(result.content) == 1

        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "hello"

    @pytest.mark.asyncio
    async def test_resolve_wraps_multiple_content_blocks(self):
        """Multiple TextBlocks in user_input event data all go into one Message."""
        resolver = InputResolver(make_config())
        content_blocks = [
            TextBlock(text="first"),
            TextBlock(text="second"),
            TextBlock(text="third"),
        ]
        event = make_user_input_event(data=content_blocks)

        result = await resolver.resolve([event])

        assert len(result.content) == 1
        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert len(msg.content) == 3
        assert msg.content[0].text == "first"
        assert msg.content[1].text == "second"
        assert msg.content[2].text == "third"

    @pytest.mark.asyncio
    async def test_resolve_returns_resolved_content_type(self):
        """resolve() must return a ResolvedContent instance."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=[TextBlock(text="hi")])
        result = await resolver.resolve([event])
        assert isinstance(result, ResolvedContent)

    @pytest.mark.asyncio
    async def test_resolve_sets_resolver_name(self):
        """ResolvedContent.resolver_name must be 'input'."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=[TextBlock(text="hi")])
        result = await resolver.resolve([event])
        assert result.resolver_name == "input"

    @pytest.mark.asyncio
    async def test_resolve_sets_source_layer(self):
        """ResolvedContent.source_layer must be set to a non-empty string."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=[TextBlock(text="hi")])
        result = await resolver.resolve([event])
        assert isinstance(result.source_layer, str)
        assert result.source_layer  # non-empty


# ---------------------------------------------------------------------------
# 4. resolve() — ignores non-user_input events
# ---------------------------------------------------------------------------


class TestInputResolverIgnoresOtherEvents:
    @pytest.mark.asyncio
    async def test_resolve_with_no_user_input_events_returns_empty(self):
        """If only non-matching events are present, returns empty content."""
        resolver = InputResolver(make_config())
        events = [
            make_other_event("turn_start"),
            make_other_event("layer_ready"),
        ]
        result = await resolver.resolve(events)
        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_with_empty_events_list_returns_empty(self):
        """Empty events list produces empty content."""
        resolver = InputResolver(make_config())
        result = await resolver.resolve([])
        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_with_data_none_returns_empty(self):
        """user_input event with data=None produces empty content."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=None)
        result = await resolver.resolve([event])
        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_with_data_empty_list_returns_empty(self):
        """user_input event with data=[] produces empty content."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=[])
        result = await resolver.resolve([event])
        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_picks_user_input_among_mixed_events(self):
        """When events list has both user_input and other events,
        only user_input data is wrapped."""
        resolver = InputResolver(make_config())
        content_blocks = [TextBlock(text="the input")]
        events = [
            make_other_event("turn_start"),
            make_user_input_event(data=content_blocks),
            make_other_event("layer_ready"),
        ]
        result = await resolver.resolve(events)
        assert len(result.content) == 1
        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.content[0].text == "the input"


# ---------------------------------------------------------------------------
# 5. execution_count
# ---------------------------------------------------------------------------


class TestInputResolverExecutionCount:
    def test_execution_count_starts_at_zero(self):
        """Fresh resolver should have execution_count == 0."""
        resolver = InputResolver(make_config())
        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_execution_count_increments_after_resolve(self):
        """execution_count increments once per resolve() call."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=[TextBlock(text="hi")])
        await resolver.resolve([event])
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_each_call(self):
        """Multiple calls accumulate in execution_count."""
        resolver = InputResolver(make_config())
        event = make_user_input_event(data=[TextBlock(text="hi")])
        await resolver.resolve([event])
        await resolver.resolve([event])
        await resolver.resolve([event])
        assert resolver.execution_count == 3

    @pytest.mark.asyncio
    async def test_execution_count_increments_even_with_no_matching_events(self):
        """execution_count increments even when no user_input events found."""
        resolver = InputResolver(make_config())
        await resolver.resolve([make_other_event()])
        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 6. Protocol conformance
# ---------------------------------------------------------------------------


class TestInputResolverProtocolConformance:
    def test_isinstance_resolver(self):
        """InputResolver must satisfy the Resolver protocol."""
        resolver = InputResolver(make_config())
        assert isinstance(resolver, Resolver)

    def test_has_subscriptions_attribute(self):
        resolver = InputResolver(make_config())
        assert hasattr(resolver, "subscriptions")
        assert isinstance(resolver.subscriptions, list)

    def test_has_max_executions_attribute(self):
        resolver = InputResolver(make_config())
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_has_execution_count_attribute(self):
        resolver = InputResolver(make_config())
        assert hasattr(resolver, "execution_count")
        assert isinstance(resolver.execution_count, int)


# ---------------------------------------------------------------------------
# 7. build() classmethod
# ---------------------------------------------------------------------------


class TestInputResolverBuild:
    def test_build_returns_input_resolver_instance(self):
        """build() must return an InputResolver instance."""
        config = make_config()
        result = InputResolver.build(config, Dependencies())
        assert isinstance(result, InputResolver)

    def test_build_with_populated_deps_also_works(self):
        """build() must accept and ignore a non-empty Dependencies container."""
        config = make_config()
        deps = Dependencies(llm={"default": lambda *a, **kw: None})
        result = InputResolver.build(config, deps)
        assert isinstance(result, InputResolver)

    def test_build_result_satisfies_resolver_protocol(self):
        """Instance returned by build() must satisfy isinstance(x, Resolver)."""
        config = make_config()
        result = InputResolver.build(config, Dependencies())
        assert isinstance(result, Resolver)

    def test_build_state_matches_direct_construction(self):
        """build() must produce an instance with the same observable state
        as one constructed via InputResolver(config) directly."""
        config = make_config(max_executions=7)
        via_build = InputResolver.build(config, Dependencies())
        via_init = InputResolver(config)
        assert via_build.max_executions == via_init.max_executions
        assert via_build.name == via_init.name
        assert via_build.execution_count == via_init.execution_count
