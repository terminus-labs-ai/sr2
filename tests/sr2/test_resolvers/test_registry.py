import pytest

from sr2.resolvers.registry import (
    ContentResolverRegistry,
    ResolvedContent,
    ResolverContext,
)


class MockResolver:
    """A mock resolver that satisfies the ContentResolver protocol.

    Echoes config back into content so callers can verify config is passed through.
    """

    async def resolve(self, key, config, context):
        # Reflect config into content so tests can verify config propagation
        if config:
            content = f"mock content config={config}"
        else:
            content = "mock content"
        tokens = len(content) // 4
        return ResolvedContent(key=key, content=content, tokens=tokens)


class AnotherMockResolver:
    """A second mock resolver for overwrite testing."""

    async def resolve(self, key, config, context):
        content = "another mock"
        return ResolvedContent(key=key, content=content, tokens=len(content) // 4)


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
async def test_mock_resolver_resolve_with_config():
    """Register a mock resolver and call resolve() — config is reflected in output."""
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
    config = {"source": "config"}
    result = await retrieved.resolve(
        key="system_prompt",
        config=config,
        context=ctx,
    )

    assert isinstance(result, ResolvedContent)
    assert result.key == "system_prompt"
    # Config is echoed into the content so we can verify it was passed through
    assert "config=" in result.content
    assert "'source': 'config'" in result.content
    assert result.tokens == len(result.content) // 4


@pytest.mark.asyncio
async def test_mock_resolver_resolve_empty_config():
    """Mock resolver with empty config returns base content."""
    resolver = MockResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    result = await resolver.resolve(key="test", config={}, context=ctx)

    assert result.content == "mock content"
    assert "config=" not in result.content


@pytest.mark.asyncio
async def test_mock_resolver_key_matches_input():
    """Resolver protocol: returned key must match the input key parameter."""
    resolver = MockResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    for key in ["system_prompt", "user_input", "memories"]:
        result = await resolver.resolve(key=key, config={}, context=ctx)
        assert result.key == key, f"Expected key '{key}', got '{result.key}'"


@pytest.mark.asyncio
async def test_mock_resolver_tokens_use_estimate():
    """Mock resolver tokens match estimate_tokens(content)."""
    from sr2.resolvers.registry import estimate_tokens

    resolver = MockResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")
    result = await resolver.resolve(key="test", config={}, context=ctx)

    expected = estimate_tokens(result.content)
    assert result.tokens == expected
