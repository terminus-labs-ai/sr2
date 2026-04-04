"""Tests for MCPPromptResolver."""

import pytest

from sr2.resolvers.mcp_prompt_resolver import MCPPromptResolver
from sr2.resolvers.registry import ResolvedContent, ResolverContext


@pytest.fixture
def context():
    return ResolverContext(
        agent_config={},
        trigger_input="test",
        session_id="sess-1",
        interface_type="user_message",
    )


@pytest.mark.asyncio
async def test_resolve_calls_get_prompt_fn(context):
    async def mock_get_prompt(name, arguments=None, server_name=None):
        return f"[user] Review {arguments.get('language', 'code')}"

    resolver = MCPPromptResolver(mock_get_prompt)
    result = await resolver.resolve(
        key="code_review",
        config={"server": "prompts", "arguments": {"language": "python"}},
        context=context,
    )

    assert isinstance(result, ResolvedContent)
    assert result.key == "code_review"
    assert "Review python" in result.content
    assert result.metadata["source"] == "mcp_prompt"
    assert result.metadata["server"] == "prompts"


@pytest.mark.asyncio
async def test_resolve_without_arguments(context):
    async def mock_get_prompt(name, arguments=None, server_name=None):
        return f"[system] Default prompt for {name}"

    resolver = MCPPromptResolver(mock_get_prompt)
    result = await resolver.resolve(
        key="greeting",
        config={},
        context=context,
    )

    assert "Default prompt for greeting" in result.content
    assert result.metadata["server"] is None


@pytest.mark.asyncio
async def test_resolve_token_count(context):
    async def mock_get_prompt(name, arguments=None, server_name=None):
        return "one two three"

    resolver = MCPPromptResolver(mock_get_prompt)
    result = await resolver.resolve(key="test", config={}, context=context)

    assert result.tokens == 3


@pytest.mark.asyncio
async def test_different_arguments_produce_different_output(context):
    """Config arguments are passed through and affect resolved content."""
    async def mock_get_prompt(name, arguments=None, server_name=None):
        lang = (arguments or {}).get("language", "unknown")
        return f"Review {lang} code"

    resolver = MCPPromptResolver(mock_get_prompt)

    result_py = await resolver.resolve(
        key="review", config={"arguments": {"language": "python"}}, context=context,
    )
    result_rs = await resolver.resolve(
        key="review", config={"arguments": {"language": "rust"}}, context=context,
    )

    assert "python" in result_py.content
    assert "rust" in result_rs.content
    assert result_py.content != result_rs.content


@pytest.mark.asyncio
async def test_server_config_passed_to_fn(context):
    """Config server field is forwarded to the get_prompt_fn."""
    captured = {}

    async def mock_get_prompt(name, arguments=None, server_name=None):
        captured["server_name"] = server_name
        return "content"

    resolver = MCPPromptResolver(mock_get_prompt)
    await resolver.resolve(key="p", config={"server": "my-server"}, context=context)

    assert captured["server_name"] == "my-server"


@pytest.mark.asyncio
async def test_no_server_config_passes_none(context):
    """Empty config passes None for server_name."""
    captured = {}

    async def mock_get_prompt(name, arguments=None, server_name=None):
        captured["server_name"] = server_name
        return "content"

    resolver = MCPPromptResolver(mock_get_prompt)
    await resolver.resolve(key="p", config={}, context=context)

    assert captured["server_name"] is None
