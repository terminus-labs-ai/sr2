"""Tests for session management."""

from datetime import UTC, datetime, timedelta

import pytest

from runtime.session import Session, SessionConfig, SessionManager


class TestSession:
    """Tests for the Session class."""

    def test_add_user_message(self):
        session = Session("s1")
        session.add_user_message("Hello")
        assert len(session.turns) == 1
        assert session.turns[0]["role"] == "user"
        assert session.turns[0]["content"] == "Hello"

    def test_add_tool_call_adds_two_turns(self):
        session = Session("s1")
        session.add_tool_call(
            tool_name="search",
            arguments={"query": "test"},
            result="found it",
            call_id="tc_1",
        )
        assert len(session.turns) == 2
        # First turn: assistant with tool_calls
        assert session.turns[0]["role"] == "assistant"
        assert session.turns[0]["content_type"] == "tool_call"
        assert session.turns[0]["tool_calls"][0]["function"]["name"] == "search"
        # Second turn: tool_result
        assert session.turns[1]["role"] == "tool_result"
        assert session.turns[1]["content"] == "found it"
        assert session.turns[1]["tool_call_id"] == "tc_1"

    def test_get_last_user_message(self):
        session = Session("s1")
        session.add_user_message("First")
        session.add_assistant_message("Reply")
        session.add_user_message("Second")
        assert session.get_last_user_message() == "Second"

    def test_get_last_user_message_none_when_empty(self):
        session = Session("s1")
        assert session.get_last_user_message() is None

    def test_to_history_returns_all_turns(self):
        session = Session("s1")
        session.add_user_message("Hi")
        session.add_assistant_message("Hey")
        history = session.to_history()
        assert len(history) == 2
        assert history is session.turns

    def test_turn_count(self):
        session = Session("s1")
        session.add_user_message("A")
        session.add_assistant_message("B")
        assert session.turn_count == 2

    def test_user_message_count(self):
        session = Session("s1")
        session.add_user_message("A")
        session.add_assistant_message("B")
        session.add_user_message("C")
        assert session.user_message_count == 2

    def test_inject_message(self):
        session = Session("s1")
        session.inject_message("assistant", "Injected!", {"source": "test"})
        assert len(session.turns) == 1
        assert session.turns[0]["injected"] is True
        assert session.turns[0]["metadata"]["source"] == "test"

    def test_rolling_lifecycle_enforces_max_turns(self):
        config = SessionConfig(name="rolling", lifecycle="rolling", max_turns=3)
        session = Session("s1", config)
        for i in range(5):
            session.add_user_message(f"msg {i}")
        assert len(session.turns) == 3
        assert session.turns[0]["content"] == "msg 2"

    def test_persistent_lifecycle_no_truncation(self):
        config = SessionConfig(name="persistent", lifecycle="persistent", max_turns=3)
        session = Session("s1", config)
        for i in range(5):
            session.add_user_message(f"msg {i}")
        assert len(session.turns) == 5


class TestSessionManager:
    """Tests for the SessionManager."""

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new(self):
        mgr = SessionManager()
        session = await mgr.get_or_create("s1")
        assert session.id == "s1"
        assert "s1" in mgr.active_sessions

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(self):
        mgr = SessionManager()
        s1 = await mgr.get_or_create("s1")
        s1.add_user_message("Hello")
        s2 = await mgr.get_or_create("s1")
        assert s2 is s1
        assert s2.turn_count == 1

    @pytest.mark.asyncio
    async def test_close_removes_session(self):
        mgr = SessionManager()
        await mgr.get_or_create("s1")
        closed = await mgr.close("s1")
        assert closed is not None
        assert mgr.get("s1") is None

    @pytest.mark.asyncio
    async def test_destroy_removes_session(self):
        mgr = SessionManager()
        await mgr.get_or_create("s1")
        destroyed = await mgr.destroy("s1")
        assert destroyed is not None
        assert mgr.get("s1") is None

    def test_cleanup_idle_closes_old_sessions(self):
        mgr = SessionManager(idle_timeout_minutes=1)
        # Manually create sessions to avoid async
        config = SessionConfig(name="old_session", max_turns=200, idle_timeout_minutes=1)
        old_session = Session("old_session", config)
        old_session.last_activity = datetime.now(UTC) - timedelta(minutes=5)
        mgr._sessions["old_session"] = old_session

        new_session = Session("new_session", SessionConfig(name="new_session", max_turns=200, idle_timeout_minutes=1))
        mgr._sessions["new_session"] = new_session

        closed = mgr.cleanup_idle()
        assert closed == 1
        assert mgr.get("old_session") is None
        assert mgr.get("new_session") is not None

    def test_cleanup_idle_skips_ephemeral(self):
        mgr = SessionManager(idle_timeout_minutes=1)
        eph = mgr.create_ephemeral("heartbeat")
        eph.last_activity = datetime.now(UTC) - timedelta(minutes=5)
        closed = mgr.cleanup_idle()
        assert closed == 0

    def test_create_ephemeral(self):
        mgr = SessionManager()
        session = mgr.create_ephemeral("heartbeat")
        assert session.config.lifecycle == "ephemeral"
        assert session.id.startswith("heartbeat_")
        assert session.id in mgr.active_sessions

    @pytest.mark.asyncio
    async def test_session_configs_applied(self):
        configs = {
            "main": SessionConfig(name="main", max_turns=50, idle_timeout_minutes=10),
        }
        mgr = SessionManager(session_configs=configs)
        session = await mgr.get_or_create("main")
        assert session.config.max_turns == 50
        assert session.config.idle_timeout_minutes == 10

    @pytest.mark.asyncio
    async def test_default_config_applied_to_unknown_sessions(self):
        default = SessionConfig(name="_default", max_turns=42, idle_timeout_minutes=15)
        mgr = SessionManager(default_config=default)
        session = await mgr.get_or_create("unknown")
        assert session.config.max_turns == 42
        assert session.config.idle_timeout_minutes == 15
