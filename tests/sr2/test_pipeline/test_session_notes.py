"""Tests for session notes — compaction-immune agent working memory."""

import json

import pytest

from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, SummarizationConfig
from sr2.pipeline.conversation import ConversationManager, ConversationZones
from sr2.summarization.engine import SummarizationEngine


def _make_compaction_engine(raw_window: int = 3) -> CompactionEngine:
    config = CompactionConfig(
        enabled=True,
        raw_window=raw_window,
        min_content_size=10,
        rules=[CompactionRuleConfig(type="tool_output", strategy="schema_and_sample")],
    )
    return CompactionEngine(config)


def _make_turn(num: int, content: str = "turn content") -> ConversationTurn:
    return ConversationTurn(turn_number=num, role="assistant", content=content)


class TestSessionNotesAPI:
    """Session notes CRUD operations."""

    def test_add_session_note(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)

        mgr.add_session_note("current task: fix auth bug")

        assert mgr.get_session_notes() == ["current task: fix auth bug"]

    def test_add_multiple_notes(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)

        mgr.add_session_note("note 1")
        mgr.add_session_note("note 2")
        mgr.add_session_note("note 3")

        assert mgr.get_session_notes() == ["note 1", "note 2", "note 3"]

    def test_replace_session_notes(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.add_session_note("old note")

        mgr.replace_session_notes(["new note 1", "new note 2"])

        assert mgr.get_session_notes() == ["new note 1", "new note 2"]

    def test_clear_session_notes(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.add_session_note("note 1")
        mgr.add_session_note("note 2")

        mgr.clear_session_notes()

        assert mgr.get_session_notes() == []

    def test_get_session_notes_returns_copy(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.add_session_note("note")

        result = mgr.get_session_notes()
        result.append("mutated")

        assert mgr.get_session_notes() == ["note"]

    def test_session_notes_per_session(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.add_session_note("note for A", session_id="A")
        mgr.add_session_note("note for B", session_id="B")

        assert mgr.get_session_notes(session_id="A") == ["note for A"]
        assert mgr.get_session_notes(session_id="B") == ["note for B"]

    def test_session_notes_in_zones(self):
        zones = ConversationZones(session_notes=["note 1", "note 2"])
        assert zones.session_notes == ["note 1", "note 2"]

    def test_session_notes_contribute_to_total_tokens(self):
        zones = ConversationZones(session_notes=["A" * 40])  # 10 tokens
        assert zones.total_tokens == 10


class TestSessionNotesSurviveCompaction:
    """Session notes must be unchanged after compaction and summarization."""

    def test_notes_survive_compaction(self):
        engine = _make_compaction_engine(raw_window=2)
        mgr = ConversationManager(compaction_engine=engine, raw_window=2)

        mgr.add_session_note("important context")
        for i in range(10):
            mgr.add_turn(_make_turn(i))

        mgr.run_compaction()

        assert mgr.get_session_notes() == ["important context"]

    @pytest.mark.asyncio
    async def test_notes_survive_summarization(self):
        async def mock_llm(system: str, prompt: str) -> str:
            return json.dumps({
                "summary_of_turns": "0-5",
                "key_decisions": ["Use Python"],
                "unresolved": [],
                "facts": [],
                "user_preferences": [],
                "errors_encountered": [],
            })

        engine = _make_compaction_engine(raw_window=2)
        summ_config = SummarizationConfig(
            trigger="token_threshold", threshold=0.5, preserve_recent_turns=0
        )
        summ_engine = SummarizationEngine(config=summ_config, llm_callable=mock_llm)
        mgr = ConversationManager(
            compaction_engine=engine,
            summarization_engine=summ_engine,
            raw_window=2,
            compacted_max_tokens=100,
        )

        mgr.add_session_note("working on auth refactor")
        mgr.add_session_note("decided to use JWT")

        # Fill compacted zone to trigger summarization
        for i in range(5):
            mgr.zones().compacted.append(_make_turn(i, content="x" * 800))

        await mgr.run_summarization()

        assert mgr.get_session_notes() == ["working on auth refactor", "decided to use JWT"]


class TestSessionNotesSerialization:
    """Session notes must survive persistence round-trips."""

    def test_zones_with_notes_restore(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.add_session_note("persisted note", session_id="s1")

        # Simulate save/restore
        zones = mgr.zones(session_id="s1")
        new_mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        new_mgr.restore_zones("s1", zones)

        assert new_mgr.get_session_notes(session_id="s1") == ["persisted note"]

    def test_zones_dataclass_with_notes(self):
        zones = ConversationZones(
            summarized=["summary"],
            compacted=[_make_turn(0)],
            raw=[_make_turn(1)],
            session_notes=["note A", "note B"],
        )
        assert zones.session_notes == ["note A", "note B"]
        # Notes contribute to total_tokens
        assert zones.total_tokens > 0

    def test_destroy_session_clears_notes(self):
        engine = _make_compaction_engine()
        mgr = ConversationManager(compaction_engine=engine, raw_window=3)
        mgr.add_session_note("ephemeral", session_id="s1")
        assert mgr.get_session_notes(session_id="s1") == ["ephemeral"]

        mgr.destroy_session("s1")

        # After destroy, a new session should have empty notes
        assert mgr.get_session_notes(session_id="s1") == []
