"""Tests for session store backends."""

import pytest

from sr2_runtime.session import Session, SessionConfig, InMemorySessionStore


class TestInMemorySessionStore:
    """Tests for InMemorySessionStore."""

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        store = InMemorySessionStore()
        config = SessionConfig(name="test", lifecycle="persistent", max_turns=100)
        session = Session("s1", config)
        session.add_user_message("Hello")
        session.add_assistant_message("Hi there")

        await store.save(session)
        loaded = await store.load("s1")

        assert loaded is not None
        assert loaded.id == "s1"
        assert loaded.config.name == "test"
        assert loaded.config.lifecycle == "persistent"
        assert len(loaded.turns) == 2
        assert loaded.turns[0]["content"] == "Hello"
        assert loaded.turns[1]["content"] == "Hi there"

    @pytest.mark.asyncio
    async def test_load_unknown_returns_none(self):
        store = InMemorySessionStore()
        assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemorySessionStore()
        session = Session("s1")
        await store.save(session)

        assert await store.delete("s1") is True
        assert await store.load("s1") is None
        assert await store.delete("s1") is False

    @pytest.mark.asyncio
    async def test_list_active(self):
        store = InMemorySessionStore()
        await store.save(Session("s1"))
        await store.save(Session("s2"))

        active = await store.list_active()
        assert set(active) == {"s1", "s2"}

    @pytest.mark.asyncio
    async def test_update_turns(self):
        store = InMemorySessionStore()
        session = Session("s1")
        session.add_user_message("old")
        await store.save(session)

        new_turns = [{"role": "user", "content": "new"}]
        await store.update_turns("s1", new_turns)

        loaded = await store.load("s1")
        assert len(loaded.turns) == 1
        assert loaded.turns[0]["content"] == "new"

    @pytest.mark.asyncio
    async def test_roundtrip_with_tool_calls(self):
        store = InMemorySessionStore()
        session = Session("s1")
        session.add_user_message("Search for X")
        session.add_tool_call("search", {"query": "X"}, "Found X", "tc_1")
        session.add_assistant_message("I found X!")
        session.inject_message("assistant", "Notification", {"source": "timer"})

        await store.save(session)
        loaded = await store.load("s1")

        assert loaded.turn_count == session.turn_count
        assert loaded.turns[0]["role"] == "user"
        assert loaded.turns[1]["content_type"] == "tool_call"
        assert loaded.turns[2]["role"] == "tool_result"
        assert loaded.turns[3]["role"] == "assistant"
        assert loaded.turns[4]["injected"] is True


class TestSessionManagerWithStore:
    """Tests for SessionManager with a backing store."""

    @pytest.mark.asyncio
    async def test_get_or_create_loads_from_store(self):
        from sr2_runtime.session import SessionManager

        store = InMemorySessionStore()

        # Pre-populate store with a session
        session = Session("cached", SessionConfig(name="cached"))
        session.add_user_message("I was here before")
        await store.save(session)

        mgr = SessionManager(store=store)
        loaded = await mgr.get_or_create("cached")
        assert loaded.turn_count == 1
        assert loaded.turns[0]["content"] == "I was here before"

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_if_not_in_store(self):
        from sr2_runtime.session import SessionManager

        store = InMemorySessionStore()
        mgr = SessionManager(store=store)

        session = await mgr.get_or_create("brand_new")
        assert session.id == "brand_new"
        assert session.turn_count == 0

        # Should also be persisted to store
        stored = await store.load("brand_new")
        assert stored is not None

    @pytest.mark.asyncio
    async def test_save_session_persists_to_store(self):
        from sr2_runtime.session import SessionManager

        store = InMemorySessionStore()
        mgr = SessionManager(store=store)

        session = await mgr.get_or_create("s1")
        session.add_user_message("Hello")
        await mgr.save_session("s1")

        loaded = await store.load("s1")
        assert loaded.turn_count == 1

    @pytest.mark.asyncio
    async def test_destroy_removes_from_cache_and_store(self):
        from sr2_runtime.session import SessionManager

        store = InMemorySessionStore()
        mgr = SessionManager(store=store)

        await mgr.get_or_create("s1")
        await mgr.destroy("s1")

        assert mgr.get("s1") is None
        assert await store.load("s1") is None

    @pytest.mark.asyncio
    async def test_load_active_sessions(self):
        from sr2_runtime.session import SessionManager

        store = InMemorySessionStore()

        # Pre-populate store
        await store.save(Session("s1", SessionConfig(name="s1")))
        await store.save(Session("s2", SessionConfig(name="s2")))

        mgr = SessionManager(store=store)
        loaded = await mgr.load_active_sessions()
        assert loaded == 2
        assert mgr.get("s1") is not None
        assert mgr.get("s2") is not None

    @pytest.mark.asyncio
    async def test_ephemeral_sessions_not_saved(self):
        from sr2_runtime.session import SessionManager

        store = InMemorySessionStore()
        mgr = SessionManager(store=store)

        eph = mgr.create_ephemeral("heartbeat")
        await mgr.save_session(eph.id)

        # Ephemeral sessions should NOT be persisted
        assert await store.load(eph.id) is None
