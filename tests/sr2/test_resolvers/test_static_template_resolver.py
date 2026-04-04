import pytest

from sr2.resolvers.static_template_resolver import StaticTemplateResolver
from sr2.resolvers.registry import ResolvedContent, ResolverContext


@pytest.mark.asyncio
async def test_returns_template_string():
    """Happy path: returns template string."""
    resolver = StaticTemplateResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")
    config = {"template": "You are a helpful assistant."}

    result = await resolver.resolve("intro", config, ctx)

    assert isinstance(result, ResolvedContent)
    assert result.key == "intro"
    assert result.content == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_missing_template_raises():
    """Missing template key raises KeyError."""
    resolver = StaticTemplateResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    with pytest.raises(KeyError, match="StaticTemplateResolver requires 'template' field in config"):
        await resolver.resolve("intro", {}, ctx)


@pytest.mark.asyncio
async def test_template_token_count():
    """Template with multiple words has correct token count."""
    resolver = StaticTemplateResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")
    config = {"template": "one two three four five"}

    result = await resolver.resolve("words", config, ctx)

    assert result.tokens == 5


@pytest.mark.asyncio
async def test_different_templates_produce_different_content():
    """Different config templates produce different resolved content."""
    resolver = StaticTemplateResolver()
    ctx = ResolverContext(agent_config={}, trigger_input="hello")

    result_a = await resolver.resolve("intro", {"template": "You are a pirate."}, ctx)
    result_b = await resolver.resolve("intro", {"template": "You are a scientist."}, ctx)

    assert result_a.content == "You are a pirate."
    assert result_b.content == "You are a scientist."
    assert result_a.content != result_b.content
