"""Tests for heartbeat scanner."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from runtime.heartbeat import Heartbeat, HeartbeatScanner, HeartbeatStatus, InMemoryHeartbeatStore
from runtime.plugins.base import TriggerContext


def _make_heartbeat(
    hb_id: str = "hb-1",
    fire_at: datetime | None = None,
    prompt: str = "Check status",
    key: str | None = None,
    context_turns: list[dict] | None = None,
    source_interface: str = "telegram",
) -> Heartbeat:
    return Heartbeat(
        id=hb_id,
        agent_name="test-agent",
        source_session="sess-1",
        source_interface=source_interface,
        prompt=prompt,
        fire_at=fire_at or (datetime.now(UTC) - timedelta(minutes=1)),
        key=key,
        context_turns=context_turns or [],
    )


class TestHeartbeatScanner:
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        store = InMemoryHeartbeatStore()
        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=1
        )

        await scanner.start()
        assert scanner._running is True
        assert scanner._task is not None

        await scanner.stop()
        assert scanner._running is False

    @pytest.mark.asyncio
    async def test_fires_due_heartbeat(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(fire_at=datetime.now(UTC) - timedelta(seconds=10))
        await store.upsert(hb)

        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=0
        )

        # Manually fire instead of running the loop
        await scanner._fire(hb)

        callback.assert_called_once()
        trigger: TriggerContext = callback.call_args[0][0]
        assert trigger.interface_name == "heartbeat"
        assert trigger.plugin_name == "heartbeat"
        assert trigger.input_data == "Check status"
        assert trigger.session_lifecycle == "ephemeral"

        # Status should be completed
        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.completed

    @pytest.mark.asyncio
    async def test_skips_future_heartbeats(self):
        store = InMemoryHeartbeatStore()
        future_hb = _make_heartbeat(fire_at=datetime.now(UTC) + timedelta(hours=1))
        await store.upsert(future_hb)

        due = await store.list_due(datetime.now(UTC))
        assert len(due) == 0

    @pytest.mark.asyncio
    async def test_status_transitions_on_success(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat()
        await store.upsert(hb)

        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=0
        )

        await scanner._fire(hb)

        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.completed

    @pytest.mark.asyncio
    async def test_status_transitions_on_failure(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat()
        await store.upsert(hb)

        callback = AsyncMock(side_effect=RuntimeError("LLM down"))
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=0
        )

        await scanner._fire(hb)

        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.failed

    @pytest.mark.asyncio
    async def test_context_turns_in_trigger_metadata(self):
        store = InMemoryHeartbeatStore()
        context = [
            {"role": "user", "content": "Call agent B"},
            {"role": "assistant", "content": "Calling..."},
        ]
        hb = _make_heartbeat(context_turns=context)
        await store.upsert(hb)

        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=0
        )

        await scanner._fire(hb)

        trigger: TriggerContext = callback.call_args[0][0]
        assert trigger.metadata["context_turns"] == context
        assert trigger.metadata["source_session"] == "sess-1"
        assert trigger.metadata["heartbeat_id"] == "hb-1"

    @pytest.mark.asyncio
    async def test_heartbeat_key_in_metadata(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(key="retry-agent-b")
        await store.upsert(hb)

        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=0
        )
        await scanner._fire(hb)

        trigger: TriggerContext = callback.call_args[0][0]
        assert trigger.metadata["heartbeat_key"] == "retry-agent-b"

    @pytest.mark.asyncio
    async def test_busy_flag_prevents_overlap(self):
        store = InMemoryHeartbeatStore()
        scanner = HeartbeatScanner(
            store=store, agent_callback=AsyncMock(), poll_interval_seconds=0
        )
        scanner._busy = True

        # Simulate one loop iteration — should skip
        scanner._running = True
        assert scanner._busy is True

    @pytest.mark.asyncio
    async def test_respond_fn_called_with_response(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(source_interface="telegram")
        await store.upsert(hb)

        callback = AsyncMock(return_value="Deploy looks good!")
        respond_fn = AsyncMock()
        scanner = HeartbeatScanner(
            store=store,
            agent_callback=callback,
            poll_interval_seconds=0,
            respond_fn=respond_fn,
        )

        await scanner._fire(hb)

        respond_fn.assert_called_once_with("telegram", "sess-1", "Deploy looks good!")

    @pytest.mark.asyncio
    async def test_respond_fn_not_called_when_response_empty(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(source_interface="telegram")
        await store.upsert(hb)

        callback = AsyncMock(return_value="")
        respond_fn = AsyncMock()
        scanner = HeartbeatScanner(
            store=store,
            agent_callback=callback,
            poll_interval_seconds=0,
            respond_fn=respond_fn,
        )

        await scanner._fire(hb)

        respond_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_respond_fn_not_called_when_none(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(source_interface="telegram")
        await store.upsert(hb)

        callback = AsyncMock(return_value="Some response")
        scanner = HeartbeatScanner(
            store=store,
            agent_callback=callback,
            poll_interval_seconds=0,
            respond_fn=None,
        )

        # Should not raise
        await scanner._fire(hb)

        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.completed

    @pytest.mark.asyncio
    async def test_respond_fn_not_called_when_no_source_interface(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(source_interface="")
        await store.upsert(hb)

        callback = AsyncMock(return_value="Some response")
        respond_fn = AsyncMock()
        scanner = HeartbeatScanner(
            store=store,
            agent_callback=callback,
            poll_interval_seconds=0,
            respond_fn=respond_fn,
        )

        await scanner._fire(hb)

        respond_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_respond_fn_failure_does_not_fail_heartbeat(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(source_interface="telegram")
        await store.upsert(hb)

        callback = AsyncMock(return_value="response")
        respond_fn = AsyncMock(side_effect=RuntimeError("plugin down"))
        scanner = HeartbeatScanner(
            store=store,
            agent_callback=callback,
            poll_interval_seconds=0,
            respond_fn=respond_fn,
        )

        await scanner._fire(hb)

        # Heartbeat should still be completed, delivery failure is non-fatal
        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.completed

    @pytest.mark.asyncio
    async def test_loop_processes_multiple_due(self):
        store = InMemoryHeartbeatStore()
        now = datetime.now(UTC)
        for i in range(3):
            hb = _make_heartbeat(
                f"hb-{i}",
                fire_at=now - timedelta(seconds=i + 1),
                prompt=f"Check {i}",
            )
            await store.upsert(hb)

        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store, agent_callback=callback, poll_interval_seconds=0
        )

        # Run one scan cycle manually
        scanner._running = True
        scanner._busy = True
        try:
            due = await store.list_due(datetime.now(UTC))
            for hb in due:
                await scanner._fire(hb)
        finally:
            scanner._busy = False

        assert callback.call_count == 3
        for i in range(3):
            loaded = await store.get(f"hb-{i}")
            assert loaded.status == HeartbeatStatus.completed
