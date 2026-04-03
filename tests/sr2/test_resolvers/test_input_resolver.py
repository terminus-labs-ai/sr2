import pytest

from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ResolvedContent, ResolverContext, estimate_tokens


@pytest.mark.asyncio
async def test_dict_input_reads_key():
    """Dict input: reads key from dict."""
    resolver = InputResolver()
    ctx = ResolverContext(
        agent_config={},
        trigger_input={"user_message": "Hello world", "channel": "slack"},
    )

    result = await resolver.resolve("user_message", {}, ctx)

    assert isinstance(result, ResolvedContent)
    assert result.key == "user_message"
    assert result.content == "Hello world"
    assert result.tokens == estimate_tokens("Hello world")


@pytest.mark.asyncio
async def test_dict_input_missing_key_raises():
    """Dict input: missing key raises KeyError."""
    resolver = InputResolver()
    ctx = ResolverContext(
        agent_config={},
        trigger_input={"user_message": "Hello"},
    )

    with pytest.raises(KeyError, match="Key 'missing' not found in trigger_input"):
        await resolver.resolve("missing", {}, ctx)


@pytest.mark.asyncio
async def test_string_input_returns_directly():
    """String input: returns the string directly regardless of key."""
    resolver = InputResolver()
    ctx = ResolverContext(
        agent_config={},
        trigger_input="Hello world from user",
    )

    result = await resolver.resolve("any_key", {}, ctx)

    assert result.content == "Hello world from user"
    assert result.tokens == estimate_tokens("Hello world from user")


@pytest.mark.asyncio
async def test_non_string_input_converts():
    """Non-string, non-dict input: converts to string."""
    resolver = InputResolver()
    ctx = ResolverContext(
        agent_config={},
        trigger_input=42,
    )

    result = await resolver.resolve("value", {}, ctx)

    assert result.content == "42"
    assert result.tokens == estimate_tokens("42")
