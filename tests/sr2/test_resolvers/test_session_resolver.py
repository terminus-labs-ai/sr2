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
async def test_window_one_returns_last_message():
    """Window=1 returns only the last message."""
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

    result = await resolver.resolve("history", {"window": 1}, ctx)

    assert result.content == "user: Third"


@pytest.mark.asyncio
async def test_no_window_config_returns_all_messages():
    """Empty config (no window key) returns all messages without truncation."""
    resolver = SessionResolver()
    history = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user", "content": "Third"},
        {"role": "assistant", "content": "Fourth"},
        {"role": "user", "content": "Fifth"},
    ]
    ctx = ResolverContext(
        agent_config={"session_history": history},
        trigger_input="hello",
    )

    result = await resolver.resolve("history", {}, ctx)

    assert result.content.count("\n") == 4  # 5 messages, 4 newlines
    assert result.content.startswith("user: First")
    assert result.content.endswith("user: Fifth")


@pytest.mark.asyncio
async def test_window_larger_than_history_returns_all():
    """Window larger than history length returns everything."""
    resolver = SessionResolver()
    ctx = ResolverContext(
        agent_config={
            "session_history": [
                {"role": "user", "content": "Only"},
            ],
        },
        trigger_input="hello",
    )

    result_windowed = await resolver.resolve("history", {"window": 100}, ctx)
    result_no_window = await resolver.resolve("history", {}, ctx)

    assert result_windowed.content == result_no_window.content == "user: Only"


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
