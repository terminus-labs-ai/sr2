"""Tests for SessionResolver.

Covers:
  - Construction: default subscriptions (user_input + assistant_response), name, max_executions
  - Custom subscriptions override defaults
  - Protocol conformance with Resolver protocol
  - History capture: user_input events captured into internal history
  - History capture: assistant_response events captured into internal history
  - Resolve output: first call with no prior history returns empty content
  - Resolve output: current turn's events are captured but not included in output
  - Multi-turn accumulation: history grows across resolve() calls
  - Chronological ordering preserved
  - execution_count increments per call
  - Edge cases: no matching events, empty events list
"""

import pytest

from sr2.config.models import EventSubscriptionConfig, ResolverConfig
from sr2.models import Message, TextBlock, TokenUsage
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.resolvers.session import SessionResolver
from sr2.protocols.llm import CompletionResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> ResolverConfig:
    """Build a minimal ResolverConfig for SessionResolver."""
    return ResolverConfig(type="session", **kwargs)


def make_user_input_event(text: str = "hello") -> Event:
    """Create a user_input event with a single TextBlock."""
    return Event(
        name="user_input",
        phase=EventPhase.STARTING,
        source_layer="core",
        data=[TextBlock(text=text)],
    )


def make_assistant_response_event(text: str = "hi there") -> Event:
    """Create an assistant_response event with a CompletionResponse."""
    return Event(
        name="assistant_response",
        phase=EventPhase.COMPLETED,
        source_layer="core",
        data=CompletionResponse(
            id="resp-001",
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
    )


def make_other_event(name: str = "turn_start") -> Event:
    """Create a non-session event."""
    return Event(
        name=name,
        phase=EventPhase.STARTING,
        source_layer="core",
    )


# ---------------------------------------------------------------------------
# 1. Construction — defaults
# ---------------------------------------------------------------------------


class TestSessionResolverConstruction:
    def test_constructs_with_valid_config(self):
        """SessionResolver builds without error from a basic config."""
        resolver = SessionResolver(make_config())
        assert resolver is not None

    def test_name_is_session(self):
        """Resolver name attribute must be 'session'."""
        resolver = SessionResolver(make_config())
        assert resolver.name == "session"

    def test_default_subscriptions_include_user_input(self):
        """Default subscriptions must include 'user_input'."""
        resolver = SessionResolver(make_config())
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names

    def test_default_subscriptions_include_assistant_response(self):
        """Default subscriptions must include 'assistant_response'."""
        resolver = SessionResolver(make_config())
        names = [s.event_name for s in resolver.subscriptions]
        assert "assistant_response" in names

    def test_default_max_executions_is_one(self):
        """Default max_executions from ResolverConfig is 1."""
        resolver = SessionResolver(make_config())
        assert resolver.max_executions == 1

    def test_max_executions_reads_from_config(self):
        """max_executions is read from ResolverConfig, not hardcoded."""
        resolver = SessionResolver(make_config(max_executions=10))
        assert resolver.max_executions == 10


# ---------------------------------------------------------------------------
# 2. Custom subscriptions override
# ---------------------------------------------------------------------------


class TestSessionResolverSubscriptionOverride:
    def test_custom_subscriptions_replace_defaults(self):
        """Non-empty config.subscriptions override the defaults."""
        cfg = make_config(
            subscriptions=[
                EventSubscriptionConfig(event="custom_event", phase="completed"),
            ],
        )
        resolver = SessionResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "custom_event" in names
        assert "user_input" not in names
        assert "assistant_response" not in names

    def test_empty_config_subscriptions_falls_back_to_defaults(self):
        """Empty config.subscriptions => use default subscriptions."""
        cfg = make_config(subscriptions=[])
        resolver = SessionResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names
        assert "assistant_response" in names


# ---------------------------------------------------------------------------
# 3. Protocol conformance
# ---------------------------------------------------------------------------


class TestSessionResolverProtocolConformance:
    def test_isinstance_resolver(self):
        """SessionResolver must satisfy the Resolver protocol."""
        resolver = SessionResolver(make_config())
        assert isinstance(resolver, Resolver)

    def test_has_subscriptions_attribute(self):
        resolver = SessionResolver(make_config())
        assert hasattr(resolver, "subscriptions")
        assert isinstance(resolver.subscriptions, list)

    def test_has_max_executions_attribute(self):
        resolver = SessionResolver(make_config())
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_has_execution_count_attribute(self):
        resolver = SessionResolver(make_config())
        assert hasattr(resolver, "execution_count")
        assert isinstance(resolver.execution_count, int)


# ---------------------------------------------------------------------------
# 4. First turn — no prior history
# ---------------------------------------------------------------------------


class TestSessionResolverFirstTurn:
    @pytest.mark.asyncio
    async def test_first_resolve_with_user_input_returns_empty(self):
        """First call returns empty content — there's no prior history yet.
        The current user_input is captured for next turn, not returned."""
        resolver = SessionResolver(make_config())
        event = make_user_input_event("first message")
        result = await resolver.resolve([event])

        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_first_resolve_with_no_events_returns_empty(self):
        """First call with no events returns empty content."""
        resolver = SessionResolver(make_config())
        result = await resolver.resolve([])

        assert isinstance(result, ResolvedContent)
        assert result.content == []

    @pytest.mark.asyncio
    async def test_first_resolve_sets_resolver_name(self):
        """ResolvedContent.resolver_name must be 'session'."""
        resolver = SessionResolver(make_config())
        result = await resolver.resolve([])
        assert result.resolver_name == "session"

    @pytest.mark.asyncio
    async def test_first_resolve_sets_source_layer(self):
        """ResolvedContent.source_layer must be set to a non-empty string."""
        resolver = SessionResolver(make_config())
        result = await resolver.resolve([])
        assert isinstance(result.source_layer, str)
        assert result.source_layer


# ---------------------------------------------------------------------------
# 5. History capture — user_input events
# ---------------------------------------------------------------------------


class TestSessionResolverUserInputCapture:
    @pytest.mark.asyncio
    async def test_user_input_captured_into_history(self):
        """After resolving with a user_input event, the user message
        is captured and appears in the NEXT resolve call's output."""
        resolver = SessionResolver(make_config(max_executions=100))
        user_event = make_user_input_event("hello")

        # First call: captures "hello", returns empty (no prior history)
        await resolver.resolve([user_event])

        # Second call: should return the captured user message
        result = await resolver.resolve([])
        assert len(result.content) == 1
        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert len(msg.content) == 1
        assert msg.content[0].text == "hello"

    @pytest.mark.asyncio
    async def test_user_input_not_in_same_call_output(self):
        """The current turn's user_input is NOT in the output of the
        same resolve() call — it's captured for next turn."""
        resolver = SessionResolver(make_config())
        user_event = make_user_input_event("current turn input")
        result = await resolver.resolve([user_event])

        # Output should not contain the current turn's message
        for item in result.content:
            if isinstance(item, Message) and item.role == "user":
                assert item.content[0].text != "current turn input"


# ---------------------------------------------------------------------------
# 6. History capture — assistant_response events
# ---------------------------------------------------------------------------


class TestSessionResolverAssistantResponseCapture:
    @pytest.mark.asyncio
    async def test_assistant_response_captured_into_history(self):
        """After resolving with an assistant_response event, the assistant
        message is captured and appears in a subsequent resolve call."""
        resolver = SessionResolver(make_config(max_executions=100))
        assistant_event = make_assistant_response_event("I can help")

        # First call: captures assistant response
        await resolver.resolve([assistant_event])

        # Second call: should return the captured assistant message
        result = await resolver.resolve([])
        assert len(result.content) == 1
        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "assistant"
        assert len(msg.content) == 1
        assert msg.content[0].text == "I can help"

    @pytest.mark.asyncio
    async def test_assistant_response_extracts_content_from_completion_response(self):
        """CompletionResponse.content is extracted into Message(role='assistant').
        The CompletionResponse wrapper itself is not stored."""
        resolver = SessionResolver(make_config(max_executions=100))
        assistant_event = make_assistant_response_event("response text")

        await resolver.resolve([assistant_event])
        result = await resolver.resolve([])

        msg = result.content[0]
        assert isinstance(msg, Message)
        assert msg.role == "assistant"
        # Content should be the list of ContentBlocks, not the CompletionResponse
        assert msg.content[0].text == "response text"


# ---------------------------------------------------------------------------
# 7. Multi-turn accumulation
# ---------------------------------------------------------------------------


class TestSessionResolverMultiTurn:
    @pytest.mark.asyncio
    async def test_second_turn_returns_first_turn_history(self):
        """After one complete turn (user + assistant), the next resolve
        returns both messages from that prior turn."""
        resolver = SessionResolver(make_config(max_executions=100))

        # Turn 1: user input
        await resolver.resolve([make_user_input_event("hi")])
        # Turn 1: assistant response
        await resolver.resolve([make_assistant_response_event("hello!")])

        # Turn 2: should return prior turn's history
        result = await resolver.resolve([make_user_input_event("what's up?")])

        assert len(result.content) == 2
        assert result.content[0].role == "user"
        assert result.content[0].content[0].text == "hi"
        assert result.content[1].role == "assistant"
        assert result.content[1].content[0].text == "hello!"

    @pytest.mark.asyncio
    async def test_three_turns_returns_accumulated_history(self):
        """After two complete turns, the third resolve returns all four
        prior messages in chronological order."""
        resolver = SessionResolver(make_config(max_executions=100))

        # Turn 1
        await resolver.resolve([make_user_input_event("turn 1 user")])
        await resolver.resolve([make_assistant_response_event("turn 1 assistant")])

        # Turn 2
        await resolver.resolve([make_user_input_event("turn 2 user")])
        await resolver.resolve([make_assistant_response_event("turn 2 assistant")])

        # Turn 3: should return all 4 prior messages
        result = await resolver.resolve([make_user_input_event("turn 3 user")])

        assert len(result.content) == 4
        assert result.content[0].role == "user"
        assert result.content[0].content[0].text == "turn 1 user"
        assert result.content[1].role == "assistant"
        assert result.content[1].content[0].text == "turn 1 assistant"
        assert result.content[2].role == "user"
        assert result.content[2].content[0].text == "turn 2 user"
        assert result.content[3].role == "assistant"
        assert result.content[3].content[0].text == "turn 2 assistant"

    @pytest.mark.asyncio
    async def test_history_preserves_chronological_order(self):
        """Messages appear in the order they were captured:
        user, assistant, user, assistant, ..."""
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("u1")])
        await resolver.resolve([make_assistant_response_event("a1")])
        await resolver.resolve([make_user_input_event("u2")])
        await resolver.resolve([make_assistant_response_event("a2")])

        result = await resolver.resolve([])
        roles = [msg.role for msg in result.content]
        assert roles == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestSessionResolverEdgeCases:
    @pytest.mark.asyncio
    async def test_resolve_with_no_matching_events_returns_current_history(self):
        """Non-matching events don't affect history; current history is returned."""
        resolver = SessionResolver(make_config(max_executions=100))

        # Seed one turn of history
        await resolver.resolve([make_user_input_event("seeded")])
        await resolver.resolve([make_assistant_response_event("reply")])

        # Call with non-matching event
        result = await resolver.resolve([make_other_event("layer_ready")])

        assert len(result.content) == 2
        assert result.content[0].role == "user"
        assert result.content[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_resolve_with_empty_events_returns_current_history(self):
        """Empty events list doesn't capture anything; returns existing history."""
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("msg")])
        result = await resolver.resolve([])

        assert len(result.content) == 1
        assert result.content[0].role == "user"

    @pytest.mark.asyncio
    async def test_user_input_event_with_no_data_is_ignored(self):
        """A user_input event with data=None should not add to history."""
        resolver = SessionResolver(make_config(max_executions=100))

        empty_event = Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="core",
            data=None,
        )
        await resolver.resolve([empty_event])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_user_input_event_with_empty_data_is_ignored(self):
        """A user_input event with data=[] should not add to history."""
        resolver = SessionResolver(make_config(max_executions=100))

        empty_event = Event(
            name="user_input",
            phase=EventPhase.STARTING,
            source_layer="core",
            data=[],
        )
        await resolver.resolve([empty_event])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_assistant_response_event_with_no_data_is_ignored(self):
        """An assistant_response event with data=None should not add to history."""
        resolver = SessionResolver(make_config(max_executions=100))

        empty_event = Event(
            name="assistant_response",
            phase=EventPhase.COMPLETED,
            source_layer="core",
            data=None,
        )
        await resolver.resolve([empty_event])
        result = await resolver.resolve([])

        assert result.content == []

    @pytest.mark.asyncio
    async def test_output_is_copy_not_reference(self):
        """Returned content should not be a mutable reference to internal state.
        Mutating the output must not corrupt future resolve() calls."""
        resolver = SessionResolver(make_config(max_executions=100))

        await resolver.resolve([make_user_input_event("persistent")])
        result1 = await resolver.resolve([])

        # Mutate the returned list
        result1.content.clear()

        # Internal history should be unaffected
        result2 = await resolver.resolve([])
        assert len(result2.content) == 1
        assert result2.content[0].content[0].text == "persistent"


# ---------------------------------------------------------------------------
# 9. execution_count
# ---------------------------------------------------------------------------


class TestSessionResolverExecutionCount:
    def test_execution_count_starts_at_zero(self):
        """Fresh resolver should have execution_count == 0."""
        resolver = SessionResolver(make_config())
        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_execution_count_increments_after_resolve(self):
        """execution_count increments once per resolve() call."""
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_user_input_event()])
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_each_call(self):
        """Multiple calls accumulate in execution_count."""
        resolver = SessionResolver(make_config(max_executions=100))
        await resolver.resolve([make_user_input_event()])
        await resolver.resolve([make_assistant_response_event()])
        await resolver.resolve([make_user_input_event()])
        assert resolver.execution_count == 3

    @pytest.mark.asyncio
    async def test_execution_count_increments_even_with_no_matching_events(self):
        """execution_count increments even when no matching events found."""
        resolver = SessionResolver(make_config())
        await resolver.resolve([make_other_event()])
        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 10. build() classmethod
# ---------------------------------------------------------------------------


class TestSessionResolverBuild:
    def test_build_returns_session_resolver_instance(self):
        """build() must return a SessionResolver instance."""
        config = make_config()
        result = SessionResolver.build(config, Dependencies())
        assert isinstance(result, SessionResolver)

    def test_build_with_populated_deps_also_works(self):
        """build() must accept and ignore a non-empty Dependencies container."""
        config = make_config()
        deps = Dependencies(llm={"default": lambda *a, **kw: None})
        result = SessionResolver.build(config, deps)
        assert isinstance(result, SessionResolver)

    def test_build_result_satisfies_resolver_protocol(self):
        """Instance returned by build() must satisfy isinstance(x, Resolver)."""
        config = make_config()
        result = SessionResolver.build(config, Dependencies())
        assert isinstance(result, Resolver)

    def test_build_state_matches_direct_construction(self):
        """build() must produce an instance with the same observable state
        as one constructed via SessionResolver(config) directly."""
        config = make_config(max_executions=3)
        via_build = SessionResolver.build(config, Dependencies())
        via_init = SessionResolver(config)
        assert via_build.max_executions == via_init.max_executions
        assert via_build.name == via_init.name
        assert via_build.execution_count == via_init.execution_count
