import pytest

from sr2.resolvers.registry import (
    ContentResolverRegistry,
    ResolvedContent,
    ResolverContext,
)


class MockResolver:
    """A mock resolver that satisfies the ContentResolver protocol."""

    async def resolve(self, key, config, context):
        return ResolvedContent(key=key, content="mock content", tokens=10)


class AnotherMockResolver:
    """A second mock resolver for overwrite testing."""

    async def resolve(self, key, config, context):
        return ResolvedContent(key=key, content="another mock", tokens=20)


def test_register_and_retrieve():
    """Register a resolver and retrieve it by source name."""
    registry = ContentResolverRegistry()
    resolver = MockResolver()
    registry.register("config", resolver)

    retrieved = registry.get("config")
    assert retrieved is resolver


def test_get_unregistered_raises_key_error():
    """get() on an unregistered source raises KeyError."""
    registry = ContentResolverRegistry()

    with pytest.raises(KeyError, match="No resolver registered for source: unknown"):
        registry.get("unknown")


def test_has_returns_correct_boolean():
    """has() returns True for registered sources, False otherwise."""
    registry = ContentResolverRegistry()
    resolver = MockResolver()
    registry.register("config", resolver)

    assert registry.has("config") is True
    assert registry.has("missing") is False


def test_registered_sources_lists_all():
    """registered_sources returns all registered source names."""
    registry = ContentResolverRegistry()
    registry.register("config", MockResolver())
    registry.register("input", MockResolver())
    registry.register("session", MockResolver())

    sources = registry.registered_sources
    assert sorted(sources) == ["config", "input", "session"]


def test_register_same_name_overwrites():
    """Registering the same source name twice overwrites the first resolver."""
    registry = ContentResolverRegistry()
    first = MockResolver()
    second = AnotherMockResolver()

    registry.register("config", first)
    registry.register("config", second)

    retrieved = registry.get("config")
    assert retrieved is second
    assert retrieved is not first


@pytest.mark.asyncio
async def test_mock_resolver_resolve():
    """Register a mock resolver and call resolve() successfully."""
    registry = ContentResolverRegistry()
    resolver = MockResolver()
    registry.register("config", resolver)

    ctx = ResolverContext(
        agent_config={"system_prompt": "You are helpful."},
        trigger_input="Hello",
        session_id="sess-123",
        interface_type="user_message",
    )

    retrieved = registry.get("config")
    result = await retrieved.resolve(
        key="system_prompt",
        config={"source": "config"},
        context=ctx,
    )

    assert isinstance(result, ResolvedContent)
    assert result.key == "system_prompt"
    assert result.content == "mock content"
    assert result.tokens == 10
    assert result.metadata is None
