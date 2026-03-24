"""Tests for heartbeat tools (schedule and cancel)."""

from datetime import UTC, datetime, timedelta

import pytest

from runtime.heartbeat import (
    CancelHeartbeatTool,
    HeartbeatStatus,
    InMemoryHeartbeatStore,
    ScheduleHeartbeatTool,
)


def _make_schedule_tool(
    store=None,
    agent_name="test-agent",
    max_context_turns=10,
    session_id="sess-1",
    turns=None,
    interface_name="telegram",
):
    store = store or InMemoryHeartbeatStore()
    return (
        ScheduleHeartbeatTool(
            store=store,
            agent_name=agent_name,
            max_context_turns=max_context_turns,
            session_resolver=lambda: session_id,
            session_turns_resolver=lambda: turns or [],
            interface_resolver=lambda: interface_name,
        ),
        store,
    )


class TestScheduleHeartbeatTool:
    @pytest.mark.asyncio
    async def test_creates_heartbeat(self):
        tool, store = _make_schedule_tool()
        result = await tool.execute(delay_seconds=300, prompt="Check agent B")

        assert "scheduled" in result
        pending = await store.list_pending("test-agent")
        assert len(pending) == 1
        assert pending[0].prompt == "Check agent B"
        assert pending[0].source_session == "sess-1"
        assert pending[0].status == HeartbeatStatus.pending

    @pytest.mark.asyncio
    async def test_captures_source_interface(self):
        tool, store = _make_schedule_tool(interface_name="telegram")
        await tool.execute(delay_seconds=300, prompt="Check deploy")

        pending = await store.list_pending("test-agent")
        assert pending[0].source_interface == "telegram"

    @pytest.mark.asyncio
    async def test_captures_source_interface_none_becomes_empty(self):
        tool, store = _make_schedule_tool(interface_name=None)
        await tool.execute(delay_seconds=300, prompt="Check deploy")

        pending = await store.list_pending("test-agent")
        assert pending[0].source_interface == ""

    @pytest.mark.asyncio
    async def test_captures_last_n_turns(self):
        turns = [
            {"role": "user", "content": f"msg-{i}"}
            for i in range(15)
        ]
        tool, store = _make_schedule_tool(turns=turns, max_context_turns=5)
        await tool.execute(delay_seconds=60, prompt="Follow up")

        pending = await store.list_pending("test-agent")
        hb = pending[0]
        assert len(hb.context_turns) == 5
        assert hb.context_turns[0]["content"] == "msg-10"
        assert hb.context_turns[-1]["content"] == "msg-14"

    @pytest.mark.asyncio
    async def test_zero_max_context_turns(self):
        turns = [{"role": "user", "content": "hello"}]
        tool, store = _make_schedule_tool(turns=turns, max_context_turns=0)
        await tool.execute(delay_seconds=60, prompt="No context")

        pending = await store.list_pending("test-agent")
        assert len(pending[0].context_turns) == 0

    @pytest.mark.asyncio
    async def test_idempotent_key(self):
        tool, store = _make_schedule_tool()
        await tool.execute(delay_seconds=300, prompt="First", key="retry-b")
        await tool.execute(delay_seconds=600, prompt="Updated", key="retry-b")

        pending = await store.list_pending("test-agent")
        assert len(pending) == 1
        assert pending[0].prompt == "Updated"

    @pytest.mark.asyncio
    async def test_negative_delay_rejected(self):
        tool, store = _make_schedule_tool()
        result = await tool.execute(delay_seconds=-1, prompt="Bad")
        assert "Error" in result

        pending = await store.list_pending("test-agent")
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_zero_delay_rejected(self):
        tool, store = _make_schedule_tool()
        result = await tool.execute(delay_seconds=0, prompt="Bad")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_rejected(self):
        tool, store = _make_schedule_tool()
        result = await tool.execute(delay_seconds=60, prompt="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_whitespace_prompt_rejected(self):
        tool, store = _make_schedule_tool()
        result = await tool.execute(delay_seconds=60, prompt="   ")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_fire_at_is_in_future(self):
        tool, store = _make_schedule_tool()
        before = datetime.now(UTC)
        await tool.execute(delay_seconds=120, prompt="Later")

        pending = await store.list_pending("test-agent")
        hb = pending[0]
        assert hb.fire_at > before
        assert hb.fire_at < before + timedelta(seconds=125)

    @pytest.mark.asyncio
    async def test_tool_definition_schema(self):
        tool, _ = _make_schedule_tool()
        defn = tool.tool_definition
        assert defn["name"] == "schedule_heartbeat"
        assert "delay_seconds" in defn["parameters"]["properties"]
        assert "prompt" in defn["parameters"]["properties"]
        assert "key" in defn["parameters"]["properties"]
        assert defn["parameters"]["required"] == ["delay_seconds", "prompt"]


class TestCancelHeartbeatTool:
    @pytest.mark.asyncio
    async def test_cancels_by_key(self):
        store = InMemoryHeartbeatStore()
        schedule, _ = _make_schedule_tool(store=store)
        await schedule.execute(delay_seconds=300, prompt="Check", key="my-key")

        cancel = CancelHeartbeatTool(store=store)
        result = await cancel.execute(key="my-key")
        assert "cancelled" in result

        hb = await store.get_by_key("my-key")
        assert hb.status == HeartbeatStatus.cancelled

    @pytest.mark.asyncio
    async def test_cancel_unknown_key(self):
        store = InMemoryHeartbeatStore()
        cancel = CancelHeartbeatTool(store=store)
        result = await cancel.execute(key="nonexistent")
        assert "No pending heartbeat" in result

    @pytest.mark.asyncio
    async def test_cancel_empty_key(self):
        store = InMemoryHeartbeatStore()
        cancel = CancelHeartbeatTool(store=store)
        result = await cancel.execute(key="")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_tool_definition_schema(self):
        cancel = CancelHeartbeatTool(store=InMemoryHeartbeatStore())
        defn = cancel.tool_definition
        assert defn["name"] == "cancel_heartbeat"
        assert defn["parameters"]["required"] == ["key"]
