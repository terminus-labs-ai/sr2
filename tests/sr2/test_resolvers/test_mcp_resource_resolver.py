"""Tests for MCPResourceResolver."""

import pytest

from sr2.resolvers.mcp_resource_resolver import MCPResourceResolver
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
async def test_resolve_calls_read_fn(context):
    async def mock_read(uri, server_name=None):
        return f"Content of {uri} from {server_name}"

    resolver = MCPResourceResolver(mock_read)
    result = await resolver.resolve(
        key="file:///tmp/test.txt",
        config={"server": "fs"},
        context=context,
    )

    assert isinstance(result, ResolvedContent)
    assert result.key == "file:///tmp/test.txt"
    assert "Content of file:///tmp/test.txt from fs" in result.content
    assert result.metadata["source"] == "mcp_resource"
    assert result.metadata["server"] == "fs"


@pytest.mark.asyncio
async def test_resolve_without_server(context):
    async def mock_read(uri, server_name=None):
        return "auto-detected content"

    resolver = MCPResourceResolver(mock_read)
    result = await resolver.resolve(
        key="some://uri",
        config={},
        context=context,
    )

    assert result.content == "auto-detected content"
    assert result.metadata["server"] is None


@pytest.mark.asyncio
async def test_resolve_token_count(context):
    async def mock_read(uri, server_name=None):
        return "one two three four five"

    resolver = MCPResourceResolver(mock_read)
    result = await resolver.resolve(key="uri", config={}, context=context)

    assert result.tokens == 5


@pytest.mark.asyncio
async def test_server_config_routes_to_correct_server(context):
    """Different server config values route to different servers."""
    captured = {}

    async def mock_read(uri, server_name=None):
        captured["server_name"] = server_name
        return f"from {server_name}"

    resolver = MCPResourceResolver(mock_read)

    result_a = await resolver.resolve(key="uri", config={"server": "alpha"}, context=context)
    assert captured["server_name"] == "alpha"
    assert result_a.metadata["server"] == "alpha"

    result_b = await resolver.resolve(key="uri", config={"server": "beta"}, context=context)
    assert captured["server_name"] == "beta"
    assert result_b.metadata["server"] == "beta"

    assert result_a.content != result_b.content
