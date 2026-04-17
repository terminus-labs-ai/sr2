"""Tests for bridge session persistence.

Covers:
- Store-level CRUD: save/load roundtrip, upsert, cascade delete
- Engine integration: persistence after optimize(), restore on startup
- Graceful degradation: DB errors don't break requests
- Restored session behavior: hash detection, compaction continuity
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from sr2.compaction.engine import ConversationTurn
from sr2.config.models import CompactionConfig, CostGateConfig, PipelineConfig
from sr2.pipeline.conversation import ConversationZones

from sr2_bridge.adapters.anthropic import AnthropicAdapter
from sr2_bridge.config import BridgeConfig, BridgeMemoryConfig, BridgeSessionConfig
from sr2_bridge.engine import BridgeEngine
from sr2_bridge.persistence import BridgeSessionStore
from sr2_bridge.session_tracker import BridgeSession, SessionTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path():
    """Temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_session():
    return BridgeSession(
        session_id="test-session",
        created_at=1000.0,
        last_seen=2000.0,
        request_count=5,
        last_message_count=10,
        last_message_hash="abc123",
        turn_counter=8,
    )


@pytest.fixture
def sample_zones():
    return ConversationZones(
        summarized=["Summary of turns 1-3", "Summary of turns 4-6"],
        compacted=[
            ConversationTurn(
                turn_number=7,
                role="assistant",
                content="[compacted tool output]",
                content_type="tool_output",
                metadata={"tool_name": "bash"},
                compacted=True,
            ),
        ],
        raw=[
            ConversationTurn(
                turn_number=8,
                role="user",
                content="What happened?",
            ),
            ConversationTurn(
                turn_number=9,
                role="assistant",
                content="Here is the result.",
            ),
        ],
    )


def _make_persistent_engine(db_path: str, raw_window: int = 5) -> BridgeEngine:
    """Create a BridgeEngine with persistence enabled."""
    config = PipelineConfig(compaction=CompactionConfig(raw_window=raw_window, cost_gate=CostGateConfig(enabled=False)))
    bridge_config = BridgeConfig(
        session=BridgeSessionConfig(persistence=True),
        memory=BridgeMemoryConfig(db_path=db_path),
    )
    return BridgeEngine(config, bridge_config=bridge_config)


# ---------------------------------------------------------------------------
# Store-level tests
# ---------------------------------------------------------------------------


class TestBridgeSessionStore:
    @pytest.mark.asyncio
    async def test_connect_creates_tables(self, db_path):
        store = BridgeSessionStore(db_path)
        await store.connect()
        assert store._conn is not None
        await store.disconnect()

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, db_path, sample_session, sample_zones):
        store = BridgeSessionStore(db_path)
        await store.connect()

        await store.save_session(sample_session, sample_zones)
        results = await store.load_all_sessions()

        assert len(results) == 1
        loaded_session, loaded_zones = results[0]

        # Session metadata
        assert loaded_session.session_id == "test-session"
        assert loaded_session.created_at == 1000.0
        assert loaded_session.last_seen == 2000.0
        assert loaded_session.request_count == 5
        assert loaded_session.last_message_count == 10
        assert loaded_session.last_message_hash == "abc123"
        assert loaded_session.turn_counter == 8

        # Summaries
        assert loaded_zones.summarized == [
            "Summary of turns 1-3",
            "Summary of turns 4-6",
        ]

        # Compacted turns
        assert len(loaded_zones.compacted) == 1
        ct = loaded_zones.compacted[0]
        assert ct.turn_number == 7
        assert ct.role == "assistant"
        assert ct.content == "[compacted tool output]"
        assert ct.content_type == "tool_output"
        assert ct.metadata == {"tool_name": "bash"}
        assert ct.compacted is True

        # Raw turns
        assert len(loaded_zones.raw) == 2
        assert loaded_zones.raw[0].turn_number == 8
        assert loaded_zones.raw[0].role == "user"
        assert loaded_zones.raw[1].turn_number == 9

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_delete_session_cascades(self, db_path, sample_session, sample_zones):
        store = BridgeSessionStore(db_path)
        await store.connect()

        await store.save_session(sample_session, sample_zones)
        await store.delete_session("test-session")

        results = await store.load_all_sessions()
        assert len(results) == 0

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_load_empty_db(self, db_path):
        store = BridgeSessionStore(db_path)
        await store.connect()

        results = await store.load_all_sessions()
        assert results == []

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_save_updates_existing_session(self, db_path, sample_session, sample_zones):
        store = BridgeSessionStore(db_path)
        await store.connect()

        await store.save_session(sample_session, sample_zones)

        # Update session and save again
        sample_session.request_count = 10
        sample_session.last_message_count = 20
        sample_session.turn_counter = 15
        updated_zones = ConversationZones(
            summarized=["New summary"],
            compacted=[],
            raw=[
                ConversationTurn(turn_number=15, role="user", content="Latest message"),
            ],
        )
        await store.save_session(sample_session, updated_zones)

        results = await store.load_all_sessions()
        assert len(results) == 1
        loaded_session, loaded_zones = results[0]
        assert loaded_session.request_count == 10
        assert loaded_session.turn_counter == 15
        assert loaded_zones.summarized == ["New summary"]
        assert len(loaded_zones.compacted) == 0
        assert len(loaded_zones.raw) == 1

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, db_path):
        store = BridgeSessionStore(db_path)
        await store.connect()

        for i in range(3):
            session = BridgeSession(
                session_id=f"session-{i}",
                created_at=float(i),
                last_seen=float(i),
            )
            zones = ConversationZones(
                raw=[
                    ConversationTurn(
                        turn_number=0, role="user", content=f"Message {i}"
                    ),
                ],
            )
            await store.save_session(session, zones)

        results = await store.load_all_sessions()
        assert len(results) == 3

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_turn_without_metadata(self, db_path):
        store = BridgeSessionStore(db_path)
        await store.connect()

        session = BridgeSession(session_id="no-meta", created_at=0.0, last_seen=0.0)
        zones = ConversationZones(
            raw=[
                ConversationTurn(
                    turn_number=0, role="user", content="Hello", metadata=None
                ),
            ],
        )
        await store.save_session(session, zones)

        results = await store.load_all_sessions()
        assert len(results) == 1
        assert results[0][1].raw[0].metadata is None

        await store.disconnect()

    @pytest.mark.asyncio
    async def test_no_op_without_connection(self):
        store = BridgeSessionStore(":memory:")
        # Should not raise
        session = BridgeSession(session_id="x", created_at=0.0, last_seen=0.0)
        zones = ConversationZones()
        await store.save_session(session, zones)
        results = await store.load_all_sessions()
        assert results == []
        await store.delete_session("x")
        await store.disconnect()


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestPersistenceIntegration:
    """Test that persistence actually works end-to-end through the engine."""

    @pytest.mark.asyncio
    async def test_engine_persists_after_optimize(self, db_path):
        """optimize() should write session state to the DB."""
        engine = _make_persistent_engine(db_path)
        await engine.session_store.connect()

        session = BridgeSession(session_id="persist-test")
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)

        # Verify data is in the DB
        results = await engine.session_store.load_all_sessions()
        assert len(results) == 1
        loaded_session, loaded_zones = results[0]
        assert loaded_session.session_id == "persist-test"
        assert loaded_session.last_message_count == 2
        assert loaded_session.turn_counter == 2
        assert len(loaded_zones.raw) == 2

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_engine_updates_db_on_subsequent_optimize(self, db_path):
        """Each optimize() call should update the DB with latest state."""
        engine = _make_persistent_engine(db_path)
        await engine.session_store.connect()

        session = BridgeSession(session_id="update-test")
        adapter = AnthropicAdapter()

        # First request
        msgs = [{"role": "user", "content": "Hello"}]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)

        # Second request with more messages
        msgs.append({"role": "assistant", "content": "Hi"})
        msgs.append({"role": "user", "content": "How are you?"})
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)

        results = await engine.session_store.load_all_sessions()
        assert len(results) == 1
        loaded_session, loaded_zones = results[0]
        assert loaded_session.last_message_count == 3
        assert loaded_session.turn_counter == 3
        total_turns = len(loaded_zones.compacted) + len(loaded_zones.raw)
        assert total_turns == 3

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_restore_sessions_on_startup(self, db_path):
        """Simulate restart: save state, create new engine, restore, verify."""
        # Phase 1: build up state and persist
        engine1 = _make_persistent_engine(db_path, raw_window=2)
        await engine1.session_store.connect()

        session = BridgeSession(session_id="restart-test")
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Turn 3"},
        ]
        await engine1.optimize(system=None, messages=msgs, session=session, adapter=adapter)

        # Verify compaction happened (raw_window=2)
        metrics1 = engine1.get_session_metrics(session)
        assert metrics1["compacted_count"] == 3
        assert metrics1["raw_count"] == 2

        await engine1.shutdown()

        # Phase 2: new engine, restore from DB
        engine2 = _make_persistent_engine(db_path, raw_window=2)
        await engine2.session_store.connect()

        loaded = await engine2.session_store.load_all_sessions()
        assert len(loaded) == 1

        restored_session, restored_zones = loaded[0]

        # Restore into tracker and conversation manager
        tracker = SessionTracker(BridgeSessionConfig())
        tracker.restore_session(restored_session)
        engine2.conversation_manager.restore_zones(restored_session.session_id, restored_zones)

        # Verify restored state matches
        assert restored_session.session_id == "restart-test"
        assert restored_session.last_message_count == 5
        assert restored_session.turn_counter == 5
        assert len(restored_zones.compacted) == 3
        assert len(restored_zones.raw) == 2

        await engine2.shutdown()

    @pytest.mark.asyncio
    async def test_restored_session_continues_optimizing(self, db_path):
        """After restore, new optimize() calls should work correctly."""
        # Phase 1: build state
        engine1 = _make_persistent_engine(db_path, raw_window=3)
        await engine1.session_store.connect()

        session1 = BridgeSession(session_id="continue-test")
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        await engine1.optimize(system=None, messages=msgs, session=session1, adapter=adapter)
        await engine1.shutdown()

        # Phase 2: restore and continue
        engine2 = _make_persistent_engine(db_path, raw_window=3)
        await engine2.session_store.connect()

        loaded = await engine2.session_store.load_all_sessions()
        restored_session, restored_zones = loaded[0]
        engine2.conversation_manager.restore_zones(restored_session.session_id, restored_zones)

        # Add new messages (full history as Claude Code would send)
        msgs.append({"role": "user", "content": "New message after restart"})
        await engine2.optimize(
            system=None, messages=msgs, session=restored_session, adapter=adapter
        )

        assert restored_session.last_message_count == 3
        assert restored_session.turn_counter == 3

        # Verify the new state is persisted
        results = await engine2.session_store.load_all_sessions()
        assert len(results) == 1
        final_session, final_zones = results[0]
        total_turns = len(final_zones.compacted) + len(final_zones.raw)
        assert total_turns == 3

        await engine2.shutdown()

    @pytest.mark.asyncio
    async def test_restored_session_hash_detects_duplicates(self, db_path):
        """Restored last_message_hash should prevent re-processing identical messages."""
        # Phase 1: optimize and persist
        engine1 = _make_persistent_engine(db_path)
        await engine1.session_store.connect()

        session1 = BridgeSession(session_id="hash-test")
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        await engine1.optimize(system=None, messages=msgs, session=session1, adapter=adapter)
        saved_hash = session1.last_message_hash
        assert saved_hash != ""
        await engine1.shutdown()

        # Phase 2: restore and send same messages
        engine2 = _make_persistent_engine(db_path)
        await engine2.session_store.connect()

        loaded = await engine2.session_store.load_all_sessions()
        restored_session, restored_zones = loaded[0]
        engine2.conversation_manager.restore_zones(restored_session.session_id, restored_zones)

        # Verify hash was restored
        assert restored_session.last_message_hash == saved_hash

        # Same messages → should passthrough (no re-optimization)
        injection, optimized = await engine2.optimize(
            system=None, messages=msgs, session=restored_session, adapter=adapter
        )
        # Passthrough returns None injection and original messages
        assert injection is None
        assert optimized is msgs

        await engine2.shutdown()

    @pytest.mark.asyncio
    async def test_destroy_session_deletes_from_db(self, db_path):
        """engine.destroy_session() should remove the session from the DB."""
        engine = _make_persistent_engine(db_path)
        await engine.session_store.connect()

        session = BridgeSession(session_id="destroy-test")
        adapter = AnthropicAdapter()

        msgs = [{"role": "user", "content": "Hello"}]
        await engine.optimize(system=None, messages=msgs, session=session, adapter=adapter)

        # Verify it's in the DB
        results = await engine.session_store.load_all_sessions()
        assert len(results) == 1

        # Destroy — this fires a background task for DB deletion
        engine.destroy_session("destroy-test")

        # Give the background task a moment to complete
        import asyncio

        await asyncio.sleep(0.1)

        results = await engine.session_store.load_all_sessions()
        assert len(results) == 0

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_persistence_failure_does_not_break_optimize(self, db_path):
        """If DB write fails, optimize() should still return valid results."""
        engine = _make_persistent_engine(db_path)
        await engine.session_store.connect()

        session = BridgeSession(session_id="fail-test")
        adapter = AnthropicAdapter()

        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        # Mock save_session to raise
        with patch.object(
            engine.session_store, "save_session", side_effect=Exception("DB write failed")
        ):
            injection, optimized = await engine.optimize(
                system=None, messages=msgs, session=session, adapter=adapter
            )

        # optimize() should still succeed
        assert len(optimized) == 2
        assert session.last_message_count == 2

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_engine_without_persistence_has_no_store(self):
        """Engine with persistence=False should not create a store."""
        config = PipelineConfig()
        engine = BridgeEngine(config)  # default config, persistence=False
        assert engine.session_store is None

    @pytest.mark.asyncio
    async def test_restore_session_into_tracker(self):
        """SessionTracker.restore_session() should make the session accessible."""
        tracker = SessionTracker(BridgeSessionConfig())
        session = BridgeSession(
            session_id="restored",
            created_at=1000.0,
            last_seen=2000.0,
            request_count=42,
            turn_counter=10,
        )
        tracker.restore_session(session)

        assert tracker.active_sessions == 1
        retrieved = tracker.get("restored")
        assert retrieved is session
        assert retrieved.request_count == 42
        assert retrieved.turn_counter == 10

    @pytest.mark.asyncio
    async def test_restore_zones_into_conversation_manager(self, db_path):
        """ConversationManager.restore_zones() should make zones accessible."""
        engine = _make_persistent_engine(db_path)

        zones = ConversationZones(
            summarized=["Old summary"],
            compacted=[
                ConversationTurn(
                    turn_number=0, role="user", content="Compacted", compacted=True
                ),
            ],
            raw=[
                ConversationTurn(turn_number=1, role="user", content="Recent"),
            ],
        )
        engine.conversation_manager.restore_zones("restored-session", zones)

        retrieved = engine.conversation_manager.zones("restored-session")
        assert retrieved.summarized == ["Old summary"]
        assert len(retrieved.compacted) == 1
        assert len(retrieved.raw) == 1
        assert retrieved.compacted[0].content == "Compacted"
