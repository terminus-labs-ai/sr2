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
