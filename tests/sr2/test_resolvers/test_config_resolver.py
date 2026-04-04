import pytest

from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.registry import ResolvedContent, ResolverContext, estimate_tokens


@pytest.mark.asyncio
async def test_resolve_existing_key():
    """Happy path: key exists in agent_config.

    Note: ConfigResolver ignores the `config` dict parameter — it reads
    from context.agent_config[key] directly. The empty dict is correct here.
    """
    resolver = ConfigResolver()
    ctx = ResolverContext(
        agent_config={"system_prompt": "You are helpful."},
        trigger_input="hello",
    )

    result = await resolver.resolve("system_prompt", {}, ctx)

    assert isinstance(result, ResolvedContent)
    assert result.key == "system_prompt"
    assert result.content == "You are helpful."
    assert result.tokens == estimate_tokens("You are helpful.")


@pytest.mark.asyncio
async def test_resolve_missing_key_raises():
    """Missing key raises KeyError."""
    resolver = ConfigResolver()
    ctx = ResolverContext(
        agent_config={"system_prompt": "You are helpful."},
        trigger_input="hello",
    )

    with pytest.raises(KeyError, match="Key 'missing' not found in agent_config"):
        await resolver.resolve("missing", {}, ctx)


@pytest.mark.asyncio
async def test_non_string_values_converted():
    """Non-string values are converted to string."""
    resolver = ConfigResolver()
    ctx = ResolverContext(
        agent_config={"max_tokens": 4096, "enabled": True},
        trigger_input="hello",
    )

    result = await resolver.resolve("max_tokens", {}, ctx)
    assert result.content == "4096"
    assert result.tokens == estimate_tokens("4096")

    result_bool = await resolver.resolve("enabled", {}, ctx)
    assert result_bool.content == "True"
    assert result_bool.tokens == estimate_tokens("True")
