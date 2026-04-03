import pytest

from sr2.resolvers.runtime_resolver import RuntimeResolver
from sr2.resolvers.registry import ResolvedContent, ResolverContext


@pytest.mark.asyncio
async def test_current_timestamp_format():
    """'current_timestamp' returns labeled datetime string."""
    resolver = RuntimeResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    result = await resolver.resolve("current_timestamp", {}, ctx)

    assert isinstance(result, ResolvedContent)
    assert result.key == "current_timestamp"
    assert result.content.startswith("Current date and time (UTC):")


@pytest.mark.asyncio
async def test_unknown_key_raises():
    """Unknown key raises KeyError."""
    resolver = RuntimeResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    with pytest.raises(KeyError, match="Unknown runtime key: bogus"):
        await resolver.resolve("bogus", {}, ctx)


@pytest.mark.asyncio
async def test_returned_content_non_empty():
    """Returned content is non-empty."""
    resolver = RuntimeResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    result = await resolver.resolve("current_timestamp", {}, ctx)

    assert len(result.content) > 0
    assert result.tokens > 0
