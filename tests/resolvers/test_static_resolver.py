"""Tests for StaticResolver.

Covers:
  FR5:  StaticResolver subscribes to turn_start / EventPhase.STARTING by default.
  FR6:  StaticResolver raises ValueError at construction if config["text"] is missing.
  FR9:  max_executions is read from ResolverConfig.max_executions (default 1).
  FR10: execution_count increments after each resolve() call.
  FR11: resolver exposes a name attribute.
  FR13: subscriptions fall back to default (turn_start/STARTING) when config list is empty;
        override when config provides subscriptions.

Acceptance criteria exercised:
  AC5:  resolve() returns ResolvedContent with one TextBlock matching config text.
  AC6:  ValueError at construction when config["text"] is absent.
  AC9:  StaticResolver satisfies isinstance(x, Resolver).
  AC10: execution_count increments after each resolve() call.
  AC14: Custom config subscriptions override the turn_start default.
  AC15: config["text"] is read at resolve-time (hot-reload: mutating the dict changes output).
"""

import pytest

from sr2.config.models import EventSubscriptionConfig, ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.resolvers.static import StaticResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(text: str | None = "hello world", **kwargs) -> ResolverConfig:
    """Build a minimal ResolverConfig for StaticResolver.

    Passing text=None omits the key entirely to trigger the missing-text path.
    """
    cfg: dict = {}
    if text is not None:
        cfg["text"] = text
    return ResolverConfig(type="static", config=cfg, **kwargs)


def make_turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


# ---------------------------------------------------------------------------
# 1. Construction — happy path
# ---------------------------------------------------------------------------


class TestStaticResolverConstruction:
    def test_constructs_with_valid_config(self):
        """AC5 pre: resolver builds without error when config has text."""
        resolver = StaticResolver(make_config("You are a helpful assistant."))
        assert resolver is not None

    def test_raises_value_error_when_text_missing(self):
        """AC6: Missing config["text"] must raise ValueError at construction."""
        with pytest.raises(ValueError):
            StaticResolver(make_config(text=None))

    def test_raises_value_error_when_text_is_empty_key_absent(self):
        """AC6: Completely absent key — not just falsy — must raise."""
        cfg = ResolverConfig(type="static", config={})
        with pytest.raises(ValueError):
            StaticResolver(cfg)

    def test_empty_string_text_is_accepted(self):
        """Empty string is a valid text value — not the same as missing."""
        resolver = StaticResolver(make_config(text=""))
        assert resolver is not None


# ---------------------------------------------------------------------------
# 2. Resolver protocol conformance (AC9)
# ---------------------------------------------------------------------------


class TestStaticResolverProtocolConformance:
    def test_isinstance_resolver(self):
        """AC9: StaticResolver must satisfy the Resolver protocol."""
        resolver = StaticResolver(make_config())
        assert isinstance(resolver, Resolver)

    def test_has_subscriptions_attribute(self):
        resolver = StaticResolver(make_config())
        assert hasattr(resolver, "subscriptions")
        assert isinstance(resolver.subscriptions, list)

    def test_has_max_executions_attribute(self):
        resolver = StaticResolver(make_config())
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_has_name_attribute(self):
        """FR11: resolver exposes a name attribute."""
        resolver = StaticResolver(make_config())
        assert hasattr(resolver, "name")
        assert isinstance(resolver.name, str)
        assert resolver.name  # non-empty


# ---------------------------------------------------------------------------
# 3. Default subscriptions (FR5 / FR13)
# ---------------------------------------------------------------------------


class TestStaticResolverDefaultSubscriptions:
    def test_default_subscription_is_turn_start(self):
        """FR5 / FR13: Default subscription must be turn_start."""
        resolver = StaticResolver(make_config())
        names = [s.event_name for s in resolver.subscriptions]
        assert "turn_start" in names

    def test_default_subscription_phase_is_starting(self):
        """FR5: The turn_start subscription must use EventPhase.STARTING."""
        resolver = StaticResolver(make_config())
        turn_start_subs = [
            s for s in resolver.subscriptions if s.event_name == "turn_start"
        ]
        assert turn_start_subs, "No turn_start subscription found"
        assert any(
            s.phase == EventPhase.STARTING for s in turn_start_subs
        ), "turn_start subscription must have phase STARTING"

    def test_empty_config_subscriptions_falls_back_to_default(self):
        """FR13: Empty config.subscriptions => use turn_start/STARTING default."""
        cfg = ResolverConfig(type="static", config={"text": "hi"}, subscriptions=[])
        resolver = StaticResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "turn_start" in names


# ---------------------------------------------------------------------------
# 4. Custom subscriptions override (AC14 / FR13)
# ---------------------------------------------------------------------------


class TestStaticResolverSubscriptionOverride:
    def test_custom_subscriptions_replace_defaults(self):
        """AC14: Non-empty config.subscriptions override the turn_start default."""
        cfg = ResolverConfig(
            type="static",
            config={"text": "custom"},
            subscriptions=[
                EventSubscriptionConfig(event="layer_ready", phase="completed")
            ],
        )
        resolver = StaticResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "layer_ready" in names
        assert "turn_start" not in names

    def test_custom_subscription_phase_is_respected(self):
        """AC14: Phase from config subscription is preserved."""
        cfg = ResolverConfig(
            type="static",
            config={"text": "hi"},
            subscriptions=[
                EventSubscriptionConfig(event="my_event", phase="completed")
            ],
        )
        resolver = StaticResolver(cfg)
        my_subs = [s for s in resolver.subscriptions if s.event_name == "my_event"]
        assert my_subs
        assert my_subs[0].phase == EventPhase.COMPLETED

    def test_custom_subscription_none_phase_is_preserved(self):
        """AC14: Phase=None in config subscription means any phase."""
        cfg = ResolverConfig(
            type="static",
            config={"text": "hi"},
            subscriptions=[
                EventSubscriptionConfig(event="my_event", phase=None)
            ],
        )
        resolver = StaticResolver(cfg)
        my_subs = [s for s in resolver.subscriptions if s.event_name == "my_event"]
        assert my_subs
        assert my_subs[0].phase is None

    def test_multiple_custom_subscriptions(self):
        """AC14: Multiple subscriptions are all registered."""
        cfg = ResolverConfig(
            type="static",
            config={"text": "hi"},
            subscriptions=[
                EventSubscriptionConfig(event="alpha", phase="starting"),
                EventSubscriptionConfig(event="beta", phase="completed"),
            ],
        )
        resolver = StaticResolver(cfg)
        names = [s.event_name for s in resolver.subscriptions]
        assert "alpha" in names
        assert "beta" in names
        assert "turn_start" not in names


# ---------------------------------------------------------------------------
# 5. max_executions (FR9)
# ---------------------------------------------------------------------------


class TestStaticResolverMaxExecutions:
    def test_default_max_executions_is_one(self):
        """FR9: Default max_executions == 1."""
        resolver = StaticResolver(make_config())
        assert resolver.max_executions == 1

    def test_max_executions_reads_from_resolver_config(self):
        """FR9: max_executions is read from ResolverConfig, not hardcoded."""
        cfg = ResolverConfig(type="static", config={"text": "hi"}, max_executions=5)
        resolver = StaticResolver(cfg)
        assert resolver.max_executions == 5

    def test_max_executions_zero_is_accepted(self):
        """Edge: zero executions is a valid (if unusual) configuration."""
        cfg = ResolverConfig(type="static", config={"text": "hi"}, max_executions=0)
        resolver = StaticResolver(cfg)
        assert resolver.max_executions == 0


# ---------------------------------------------------------------------------
# 6. resolve() — return value (AC5)
# ---------------------------------------------------------------------------


class TestStaticResolverResolve:
    @pytest.mark.asyncio
    async def test_resolve_returns_resolved_content(self):
        """AC5: resolve() must return a ResolvedContent instance."""
        resolver = StaticResolver(make_config("system prompt text"))
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result, ResolvedContent)

    @pytest.mark.asyncio
    async def test_resolve_content_has_one_text_block(self):
        """AC5: ResolvedContent.content must contain exactly one TextBlock."""
        resolver = StaticResolver(make_config("You are Edi."))
        result = await resolver.resolve([make_turn_start_event()])
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)

    @pytest.mark.asyncio
    async def test_resolve_text_matches_config(self):
        """AC5: The TextBlock text must match config["text"]."""
        text = "You are a concise assistant."
        resolver = StaticResolver(make_config(text))
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == text

    @pytest.mark.asyncio
    async def test_resolve_with_empty_events_list(self):
        """resolve() must work even if called with an empty events list."""
        resolver = StaticResolver(make_config("hi"))
        result = await resolver.resolve([])
        assert isinstance(result, ResolvedContent)
        assert result.content[0].text == "hi"

    @pytest.mark.asyncio
    async def test_resolve_with_multiple_events(self):
        """resolve() receives whatever events the engine passes — any length."""
        resolver = StaticResolver(make_config("hello"))
        events = [
            make_turn_start_event(),
            Event(name="other_event", phase=EventPhase.COMPLETED, source_layer="mem"),
        ]
        result = await resolver.resolve(events)
        assert result.content[0].text == "hello"

    @pytest.mark.asyncio
    async def test_resolver_name_in_result(self):
        """ResolvedContent.resolver_name must be set (non-empty string)."""
        resolver = StaticResolver(make_config())
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result.resolver_name, str)
        assert result.resolver_name  # non-empty

    @pytest.mark.asyncio
    async def test_source_layer_in_result(self):
        """ResolvedContent.source_layer must be set (non-empty string)."""
        resolver = StaticResolver(make_config())
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result.source_layer, str)
        assert result.source_layer  # non-empty


# ---------------------------------------------------------------------------
# 7. execution_count (FR10 / AC10)
# ---------------------------------------------------------------------------


class TestStaticResolverExecutionCount:
    def test_execution_count_starts_at_zero(self):
        """FR10: Fresh resolver should have execution_count == 0."""
        resolver = StaticResolver(make_config())
        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_execution_count_increments_after_resolve(self):
        """AC10: execution_count increments once per resolve() call."""
        resolver = StaticResolver(make_config())
        await resolver.resolve([make_turn_start_event()])
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_each_call(self):
        """AC10: Multiple calls accumulate in execution_count."""
        resolver = StaticResolver(make_config())
        await resolver.resolve([make_turn_start_event()])
        await resolver.resolve([make_turn_start_event()])
        await resolver.resolve([make_turn_start_event()])
        assert resolver.execution_count == 3

    @pytest.mark.asyncio
    async def test_execution_count_increments_regardless_of_events(self):
        """FR10: execution_count increments even with an empty events list."""
        resolver = StaticResolver(make_config())
        await resolver.resolve([])
        assert resolver.execution_count == 1


# ---------------------------------------------------------------------------
# 8. Hot-reload safety (AC15)
# ---------------------------------------------------------------------------


class TestStaticResolverHotReload:
    @pytest.mark.asyncio
    async def test_config_text_read_at_resolve_time(self):
        """AC15: Mutating config dict after construction changes the resolved text."""
        cfg_dict: dict = {"text": "original"}
        resolver_cfg = ResolverConfig(type="static", config=cfg_dict)
        resolver = StaticResolver(resolver_cfg)

        result_before = await resolver.resolve([make_turn_start_event()])
        assert result_before.content[0].text == "original"

        # Mutate the config dict directly
        cfg_dict["text"] = "updated"

        result_after = await resolver.resolve([make_turn_start_event()])
        assert result_after.content[0].text == "updated"

    @pytest.mark.asyncio
    async def test_text_not_cached_at_construction(self):
        """AC15: Two calls with same config dict return same current value."""
        cfg_dict: dict = {"text": "v1"}
        resolver_cfg = ResolverConfig(type="static", config=cfg_dict)
        resolver = StaticResolver(resolver_cfg)

        result1 = await resolver.resolve([make_turn_start_event()])
        cfg_dict["text"] = "v2"
        result2 = await resolver.resolve([make_turn_start_event()])

        # They must differ — proving resolve-time (not construction-time) reads
        assert result1.content[0].text == "v1"
        assert result2.content[0].text == "v2"
        assert result1.content[0].text != result2.content[0].text


# ---------------------------------------------------------------------------
# 9. build() classmethod
# ---------------------------------------------------------------------------


class TestStaticResolverBuild:
    def test_build_returns_static_resolver_instance(self):
        """build() must return a StaticResolver instance."""
        config = make_config("built text")
        result = StaticResolver.build(config, Dependencies())
        assert isinstance(result, StaticResolver)

    def test_build_with_populated_deps_also_works(self):
        """build() must accept and ignore a non-empty Dependencies container."""
        config = make_config("built text")
        deps = Dependencies(llm={"default": lambda *a, **kw: None})
        result = StaticResolver.build(config, deps)
        assert isinstance(result, StaticResolver)

    def test_build_result_satisfies_resolver_protocol(self):
        """Instance returned by build() must satisfy isinstance(x, Resolver)."""
        config = make_config("hello")
        result = StaticResolver.build(config, Dependencies())
        assert isinstance(result, Resolver)

    def test_build_state_matches_direct_construction(self):
        """build() must produce an instance with the same observable state
        as one constructed via StaticResolver(config) directly."""
        config = make_config("some static text")
        via_build = StaticResolver.build(config, Dependencies())
        via_init = StaticResolver(config)
        # Config text is read at resolve-time from the same config object,
        # so both instances share the same source. Compare max_executions
        # (from config) and that both have the same name.
        assert via_build.max_executions == via_init.max_executions
        assert via_build.name == via_init.name

    @pytest.mark.asyncio
    async def test_build_content_resolves_correctly(self):
        """Instance from build() resolves to the configured text value."""
        config = make_config("expected content")
        resolver = StaticResolver.build(config, Dependencies())
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "expected content"
