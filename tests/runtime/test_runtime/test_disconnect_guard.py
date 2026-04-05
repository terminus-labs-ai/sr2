"""Tests for HTTPPlugin._run_with_disconnect_guard."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from sr2_runtime.plugins.http import HTTPPlugin


def _make_plugin() -> HTTPPlugin:
    return HTTPPlugin(
        interface_name="test_http",
        config={"session": {"name": "test", "lifecycle": "persistent"}},
        agent_callback=AsyncMock(return_value="ok"),
    )


@pytest.mark.asyncio
async def test_returns_result_when_client_stays_connected():
    """Coroutine completes normally when client remains connected."""
    plugin = _make_plugin()
    request = AsyncMock()
    request.is_disconnected = AsyncMock(return_value=False)

    async def my_coro():
        return "success"

    result = await plugin._run_with_disconnect_guard(request, my_coro())
    assert result == "success"


@pytest.mark.asyncio
async def test_cancels_on_disconnect():
    """Coroutine is cancelled when the HTTP client disconnects."""
    plugin = _make_plugin()

    disconnect_after = 0
    call_count = 0

    async def is_disconnected():
        nonlocal call_count
        call_count += 1
        return call_count > disconnect_after

    request = AsyncMock()
    request.is_disconnected = is_disconnected

    async def slow_coro():
        await asyncio.sleep(10)
        return "should not reach"

    # Client disconnects on first poll
    disconnect_after = 0
    result = await plugin._run_with_disconnect_guard(request, slow_coro())
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_type():
    """Return type is None when client disconnects."""
    plugin = _make_plugin()
    request = AsyncMock()
    request.is_disconnected = AsyncMock(return_value=True)

    async def slow():
        await asyncio.sleep(10)

    result = await plugin._run_with_disconnect_guard(request, slow())
    assert result is None
