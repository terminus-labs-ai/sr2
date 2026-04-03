import pytest

from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.registry import ResolvedContent, ResolverContext


@pytest.mark.asyncio
async def test_formats_messages():
    """Happy path: formats messages as 'role: content'."""
    resolver = SessionResolver()
    ctx = ResolverContext(
        agent_config={
            "session_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        },
        trigger_input="hello",
    )

    result = await resolver.resolve("history", {}, ctx)

    assert isinstance(result, ResolvedContent)
    assert result.key == "history"
    assert result.content == "user: Hello\nassistant: Hi there"


@pytest.mark.asyncio
async def test_window_limits_messages():
    """Window parameter limits to last N messages."""
    resolver = SessionResolver()
    ctx = ResolverContext(
        agent_config={
            "session_history": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third"},
            ],
        },
        trigger_input="hello",
    )

    result = await resolver.resolve("history", {"window": 2}, ctx)

    assert result.content == "assistant: Second\nuser: Third"


@pytest.mark.asyncio
async def test_empty_history_returns_empty():
    """Empty history returns empty string."""
    resolver = SessionResolver()
    ctx = ResolverContext(
        agent_config={"session_history": []},
        trigger_input="hello",
    )

    result = await resolver.resolve("history", {}, ctx)

    assert result.content == ""
    assert result.tokens == 0


@pytest.mark.asyncio
async def test_missing_session_history_returns_empty():
    """Missing session_history key returns empty string."""
    resolver = SessionResolver()
    ctx = ResolverContext(
        agent_config={},
        trigger_input="hello",
    )

    result = await resolver.resolve("history", {}, ctx)

    assert result.content == ""
    assert result.tokens == 0
