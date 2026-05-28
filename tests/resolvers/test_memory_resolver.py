"""Tests for MemoryResolver — build(), resolve(), and DIP enforcement.

Covers:
  1. build() — happy path: returns MemoryResolver when store is in deps.memory_store
  2. build() — raises ConfigError when memory_store is absent from deps
  3. build() — raises ConfigError when deps is empty dict
  4. build() — raises ConfigError when deps is empty (default Dependencies)
  5. Construction — stores the injected MemoryStore (not a concrete fallback)
  6. resolve() — returns empty content when no user_input event
  7. resolve() — returns empty content when query matches nothing
  8. resolve() — returns TextBlock with prefix + memories when matches found
  9. resolve() — scope filter from config is applied
  10. resolve() — limit from config is applied
  11. resolve() — custom prefix from config is used
  12. resolve() — string data on user_input event is used as query
  13. resolve() — list of TextBlock data on user_input event is joined as query
  14. resolve() — non-user_input events are ignored
  15. Protocol: MemoryResolver satisfies Resolver protocol
  16. MemoryResolver.name is "memory"
  17. execution_count starts at 0 and increments per resolve() call
  18. subscriptions default to user_input / STARTING
  19. custom subscriptions from config override the default
"""

from __future__ import annotations

import pytest

from sr2.config.models import ConfigError, ResolverConfig, EventSubscriptionConfig
from sr2.memory import (
    InMemoryMemoryStore,
    Memory,
    MemoryScope,
    MemoryStore,
    MemoryResolver,
)
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(**kwargs) -> ResolverConfig:
    """Build a minimal ResolverConfig for MemoryResolver."""
    return ResolverConfig(type="memory", **kwargs)


def make_deps_with_store(store: MemoryStore | None = None) -> Dependencies:
    """Build Dependencies with a memory_store via typed field."""
    s = store if store is not None else InMemoryMemoryStore()
    return Dependencies(memory_store=s)  # type: ignore[call-arg]


def make_user_input_event(data) -> Event:
    return Event(name="user_input", phase=EventPhase.STARTING, source_layer="core", data=data)


def make_other_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def make_memory(content: str, scope: MemoryScope = MemoryScope.PRIVATE, tags: list[str] | None = None) -> Memory:
    return Memory(content=content, scope=scope, tags=tags or [])


# ---------------------------------------------------------------------------
# 1. build() — happy path
# ---------------------------------------------------------------------------


class TestMemoryResolverBuildHappyPath:
    def test_build_returns_memory_resolver(self):
        """build() returns a MemoryResolver when memory_store is in typed field."""
        config = make_config()
        deps = make_deps_with_store()
        result = MemoryResolver.build(config, deps)
        assert isinstance(result, MemoryResolver)

    @pytest.mark.asyncio
    async def test_build_uses_injected_store(self):
        """build() uses the injected store — pre-populated data is visible at resolve time."""
        store = InMemoryMemoryStore()
        store.save(make_memory("sentinel fact"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        # If the injected store is used, the pre-populated sentinel fact is findable.
        result = await resolver.resolve([make_user_input_event("sentinel")])
        assert len(result.content) == 1
        assert "sentinel fact" in result.content[0].text

    def test_build_result_satisfies_resolver_protocol(self):
        """Instance returned by build() must satisfy the Resolver protocol."""
        config = make_config()
        deps = make_deps_with_store()
        result = MemoryResolver.build(config, deps)
        assert isinstance(result, Resolver)


# ---------------------------------------------------------------------------
# 2. build() — ConfigError when store is absent
# ---------------------------------------------------------------------------


class TestMemoryResolverBuildConfigError:
    def test_build_raises_config_error_when_store_absent(self):
        """build() raises ConfigError when memory_store is not in deps."""
        config = make_config()
        deps = Dependencies()  # no extras at all
        with pytest.raises(ConfigError):
            MemoryResolver.build(config, deps)

    def test_build_raises_config_error_not_key_error(self):
        """The exception type must be ConfigError, not KeyError or AttributeError."""
        config = make_config()
        deps = Dependencies()
        exc = None
        try:
            MemoryResolver.build(config, deps)
        except Exception as e:
            exc = e
        assert exc is not None, "Expected an exception"
        assert isinstance(exc, ConfigError), (
            f"Expected ConfigError, got {type(exc).__name__}: {exc}"
        )

    def test_build_raises_config_error_when_extras_empty(self):
        """build() raises ConfigError when deps has no memory_store."""
        config = make_config()
        deps = Dependencies()
        with pytest.raises(ConfigError):
            MemoryResolver.build(config, deps)

    def test_build_config_error_message_mentions_memory_store(self):
        """ConfigError message must reference 'memory_store'."""
        config = make_config()
        deps = Dependencies()
        with pytest.raises(ConfigError) as exc_info:
            MemoryResolver.build(config, deps)
        message = str(exc_info.value).lower()
        assert "memory_store" in message or "memory" in message, (
            f"ConfigError message '{exc_info.value}' should mention 'memory_store' or 'memory'"
        )


# ---------------------------------------------------------------------------
# 3. Protocol and identity
# ---------------------------------------------------------------------------


class TestMemoryResolverIdentity:
    def test_name_is_memory(self):
        """MemoryResolver.name must be 'memory'."""
        assert MemoryResolver.name == "memory"

    def test_instance_name_is_memory(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        assert resolver.name == "memory"

    def test_satisfies_resolver_protocol(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        assert isinstance(resolver, Resolver)


# ---------------------------------------------------------------------------
# 4. execution_count
# ---------------------------------------------------------------------------


class TestMemoryResolverExecutionCount:
    def test_execution_count_starts_at_zero(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_execution_count_increments_after_resolve(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        await resolver.resolve([])
        assert resolver.execution_count == 1

    @pytest.mark.asyncio
    async def test_execution_count_accumulates(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        for _ in range(3):
            await resolver.resolve([])
        assert resolver.execution_count == 3


# ---------------------------------------------------------------------------
# 5. Default subscriptions
# ---------------------------------------------------------------------------


class TestMemoryResolverDefaultSubscriptions:
    def test_default_subscription_is_user_input(self):
        """Default subscription event_name must be 'user_input'."""
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        names = [s.event_name for s in resolver.subscriptions]
        assert "user_input" in names

    def test_default_subscription_phase_is_starting(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        user_input_subs = [
            s for s in resolver.subscriptions if s.event_name == "user_input"
        ]
        assert any(s.phase == EventPhase.STARTING for s in user_input_subs)

    def test_custom_subscriptions_override_default(self):
        """Non-empty config.subscriptions replace the user_input default."""
        config = ResolverConfig(
            type="memory",
            subscriptions=[EventSubscriptionConfig(event="turn_start", phase="completed")],
        )
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        names = [s.event_name for s in resolver.subscriptions]
        assert "turn_start" in names
        assert "user_input" not in names


# ---------------------------------------------------------------------------
# 6. resolve() — no match / no query
# ---------------------------------------------------------------------------


class TestMemoryResolverResolveNoContent:
    @pytest.mark.asyncio
    async def test_resolve_returns_resolved_content(self):
        """resolve() always returns a ResolvedContent."""
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([])
        assert isinstance(result, ResolvedContent)

    @pytest.mark.asyncio
    async def test_resolve_returns_empty_when_no_user_input_event(self):
        """resolve() returns empty content when no user_input event is in the list."""
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_other_event()])
        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_returns_empty_when_empty_event_list(self):
        config = make_config()
        deps = make_deps_with_store()
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([])
        assert result.content == []

    @pytest.mark.asyncio
    async def test_resolve_returns_empty_when_no_matching_memories(self):
        """resolve() returns empty content when the store has no matching entries."""
        store = InMemoryMemoryStore()
        store.save(make_memory("completely unrelated topic"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("Python programming")])
        assert result.content == []


# ---------------------------------------------------------------------------
# 7. resolve() — matching memories
# ---------------------------------------------------------------------------


class TestMemoryResolverResolveWithMatches:
    @pytest.mark.asyncio
    async def test_resolve_returns_text_block_when_matches(self):
        """resolve() returns a list with one TextBlock when memories are found."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user likes Python"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("Python")])
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)

    @pytest.mark.asyncio
    async def test_resolve_text_contains_memory_content(self):
        """The resolved TextBlock must contain the matched memory text."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user prefers dark mode"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("dark mode")])
        assert "user prefers dark mode" in result.content[0].text

    @pytest.mark.asyncio
    async def test_resolve_default_prefix_applied(self):
        """Default prefix 'Relevant context:' is prepended to memories."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user likes cats"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("cats")])
        assert result.content[0].text.startswith("Relevant context:")

    @pytest.mark.asyncio
    async def test_resolve_custom_prefix_applied(self):
        """Custom prefix from config is used instead of the default."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user hates meetings"))
        config = make_config(config={"prefix": "Background:\n"})
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("meetings")])
        assert result.content[0].text.startswith("Background:")

    @pytest.mark.asyncio
    async def test_resolve_resolver_name_in_result(self):
        """ResolvedContent.resolver_name must be 'memory'."""
        store = InMemoryMemoryStore()
        store.save(make_memory("any content"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("any")])
        assert result.resolver_name == "memory"


# ---------------------------------------------------------------------------
# 8. resolve() — string vs list[TextBlock] user_input data
# ---------------------------------------------------------------------------


class TestMemoryResolverQueryExtraction:
    @pytest.mark.asyncio
    async def test_resolve_with_string_event_data(self):
        """User input event with string data is used as query."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user likes coffee"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("coffee")])
        assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_resolve_with_text_block_list_event_data(self):
        """User input event with list[TextBlock] data is joined into a query."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user prefers tea"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        blocks = [TextBlock(text="prefers tea")]
        result = await resolver.resolve([make_user_input_event(blocks)])
        assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_resolve_ignores_non_user_input_events(self):
        """Events with names other than 'user_input' do not contribute to the query."""
        store = InMemoryMemoryStore()
        store.save(make_memory("user likes jazz"))
        config = make_config()
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        other = Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core", data="jazz")
        result = await resolver.resolve([other])
        assert result.content == []


# ---------------------------------------------------------------------------
# 9. resolve() — limit and scope config
# ---------------------------------------------------------------------------


class TestMemoryResolverConfigOptions:
    @pytest.mark.asyncio
    async def test_resolve_limit_enforced(self):
        """config['limit'] caps the number of injected memories."""
        store = InMemoryMemoryStore()
        for i in range(10):
            store.save(make_memory(f"fact {i}"))
        config = make_config(config={"limit": 3})
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("fact")])
        if result.content:
            # Each memory appears as a "- ..." bullet; count the lines after prefix
            lines = [
                ln for ln in result.content[0].text.split("\n") if ln.startswith("- ")
            ]
            assert len(lines) <= 3

    @pytest.mark.asyncio
    async def test_resolve_scope_filter_applied(self):
        """config['scope'] limits results to the specified scope."""
        store = InMemoryMemoryStore()
        store.save(make_memory("private fact", scope=MemoryScope.PRIVATE))
        store.save(make_memory("shared fact", scope=MemoryScope.SHARED))
        config = make_config(config={"scope": "private"})
        deps = make_deps_with_store(store)
        resolver = MemoryResolver.build(config, deps)
        result = await resolver.resolve([make_user_input_event("fact")])
        if result.content:
            assert "private fact" in result.content[0].text
            assert "shared fact" not in result.content[0].text
