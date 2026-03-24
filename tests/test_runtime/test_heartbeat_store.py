"""Tests for heartbeat store backends."""

from datetime import UTC, datetime, timedelta

import pytest

from runtime.heartbeat import Heartbeat, HeartbeatStatus, InMemoryHeartbeatStore


def _make_heartbeat(
    hb_id: str = "hb-1",
    agent_name: str = "test-agent",
    fire_at: datetime | None = None,
    key: str | None = None,
    prompt: str = "Check status",
    source_session: str = "sess-1",
    source_interface: str = "telegram",
) -> Heartbeat:
    return Heartbeat(
        id=hb_id,
        agent_name=agent_name,
        source_session=source_session,
        source_interface=source_interface,
        prompt=prompt,
        fire_at=fire_at or (datetime.now(UTC) + timedelta(minutes=5)),
        key=key,
    )


class TestInMemoryHeartbeatStore:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat()
        await store.upsert(hb)

        loaded = await store.get("hb-1")
        assert loaded is not None
        assert loaded.id == "hb-1"
        assert loaded.prompt == "Check status"
        assert loaded.status == HeartbeatStatus.pending

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        store = InMemoryHeartbeatStore()
        assert await store.get("nope") is None

    @pytest.mark.asyncio
    async def test_get_by_key(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(key="retry-agent-b")
        await store.upsert(hb)

        loaded = await store.get_by_key("retry-agent-b")
        assert loaded is not None
        assert loaded.id == "hb-1"

    @pytest.mark.asyncio
    async def test_get_by_key_nonexistent(self):
        store = InMemoryHeartbeatStore()
        assert await store.get_by_key("nope") is None

    @pytest.mark.asyncio
    async def test_list_due_filters_by_time(self):
        store = InMemoryHeartbeatStore()
        past = _make_heartbeat("hb-past", fire_at=datetime.now(UTC) - timedelta(minutes=1))
        future = _make_heartbeat("hb-future", fire_at=datetime.now(UTC) + timedelta(hours=1))
        await store.upsert(past)
        await store.upsert(future)

        due = await store.list_due(datetime.now(UTC))
        assert len(due) == 1
        assert due[0].id == "hb-past"

    @pytest.mark.asyncio
    async def test_list_due_excludes_non_pending(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(fire_at=datetime.now(UTC) - timedelta(minutes=1))
        await store.upsert(hb)
        await store.update_status("hb-1", HeartbeatStatus.completed)

        due = await store.list_due(datetime.now(UTC))
        assert len(due) == 0

    @pytest.mark.asyncio
    async def test_list_due_ordered_by_fire_at(self):
        store = InMemoryHeartbeatStore()
        now = datetime.now(UTC)
        hb2 = _make_heartbeat("hb-2", fire_at=now - timedelta(minutes=1))
        hb1 = _make_heartbeat("hb-1", fire_at=now - timedelta(minutes=5))
        await store.upsert(hb2)
        await store.upsert(hb1)

        due = await store.list_due(now)
        assert [h.id for h in due] == ["hb-1", "hb-2"]

    @pytest.mark.asyncio
    async def test_list_due_limited_to_50(self):
        store = InMemoryHeartbeatStore()
        now = datetime.now(UTC)
        for i in range(60):
            hb = _make_heartbeat(f"hb-{i}", fire_at=now - timedelta(seconds=i))
            await store.upsert(hb)

        due = await store.list_due(now)
        assert len(due) == 50

    @pytest.mark.asyncio
    async def test_update_status(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat()
        await store.upsert(hb)

        await store.update_status("hb-1", HeartbeatStatus.firing)
        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.firing

    @pytest.mark.asyncio
    async def test_cancel_by_key(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(key="my-key")
        await store.upsert(hb)

        result = await store.cancel_by_key("my-key")
        assert result is True

        loaded = await store.get("hb-1")
        assert loaded.status == HeartbeatStatus.cancelled

    @pytest.mark.asyncio
    async def test_cancel_by_key_nonexistent(self):
        store = InMemoryHeartbeatStore()
        result = await store.cancel_by_key("nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_by_key_already_completed(self):
        store = InMemoryHeartbeatStore()
        hb = _make_heartbeat(key="my-key")
        await store.upsert(hb)
        await store.update_status("hb-1", HeartbeatStatus.completed)

        result = await store.cancel_by_key("my-key")
        assert result is False

    @pytest.mark.asyncio
    async def test_idempotent_upsert_same_key(self):
        store = InMemoryHeartbeatStore()
        hb1 = _make_heartbeat("hb-1", key="retry-b", prompt="First")
        await store.upsert(hb1)

        new_fire_at = datetime.now(UTC) + timedelta(hours=1)
        hb2 = _make_heartbeat("hb-2", key="retry-b", prompt="Updated", fire_at=new_fire_at)
        await store.upsert(hb2)

        # Should still be one heartbeat with original ID but updated fields
        loaded = await store.get_by_key("retry-b")
        assert loaded.id == "hb-1"
        assert loaded.prompt == "Updated"
        assert loaded.fire_at == new_fire_at
        assert loaded.status == HeartbeatStatus.pending

        # hb-2 ID should not exist
        assert await store.get("hb-2") is None

    @pytest.mark.asyncio
    async def test_idempotent_upsert_copies_source_interface(self):
        store = InMemoryHeartbeatStore()
        hb1 = _make_heartbeat("hb-1", key="retry-b", source_interface="telegram")
        await store.upsert(hb1)

        hb2 = _make_heartbeat("hb-2", key="retry-b", source_interface="http")
        await store.upsert(hb2)

        loaded = await store.get_by_key("retry-b")
        assert loaded.source_interface == "http"

    @pytest.mark.asyncio
    async def test_list_pending(self):
        store = InMemoryHeartbeatStore()
        hb1 = _make_heartbeat("hb-1", agent_name="agent-a")
        hb2 = _make_heartbeat("hb-2", agent_name="agent-a")
        hb3 = _make_heartbeat("hb-3", agent_name="agent-b")
        await store.upsert(hb1)
        await store.upsert(hb2)
        await store.upsert(hb3)

        await store.update_status("hb-2", HeartbeatStatus.completed)

        pending = await store.list_pending("agent-a")
        assert len(pending) == 1
        assert pending[0].id == "hb-1"

    @pytest.mark.asyncio
    async def test_keyless_heartbeats_never_conflict(self):
        store = InMemoryHeartbeatStore()
        hb1 = _make_heartbeat("hb-1", prompt="First")
        hb2 = _make_heartbeat("hb-2", prompt="Second")
        await store.upsert(hb1)
        await store.upsert(hb2)

        assert (await store.get("hb-1")).prompt == "First"
        assert (await store.get("hb-2")).prompt == "Second"
