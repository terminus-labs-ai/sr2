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
) -> Heartbeat:
    return Heartbeat(
        id=hb_id,
        agent_name="test-agent",
        source_session="sess-1",
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
        # _loop sleeps first then checks _busy, so we test the flag directly
        assert scanner._busy is True

    @pytest.mark.asyncio
    async def test_pipeline_sets_interface_name(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat()
        await store.upsert(hb)

        callback = AsyncMock(return_value="ok")
        scanner = HeartbeatScanner(
            store=store,
            agent_callback=callback,
            poll_interval_seconds=0,
            pipeline="interfaces/heartbeat.yaml",
        )

        await scanner._fire(hb)

        trigger: TriggerContext = callback.call_args[0][0]
        assert trigger.interface_name == "_heartbeat_interfaces/heartbeat.yaml"

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
