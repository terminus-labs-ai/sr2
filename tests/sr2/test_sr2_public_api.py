"""Tests for SR2 facade public API methods (Fix 09: private attribute access from bridge).

Verifies that the 6 new public methods exist on the SR2 facade and behave correctly:
  - get_zones(session_id)         — returns ConversationZones for a session
  - get_zone_transitions(session_id) — returns zone transition counts dict
  - restore_zones(session_id, zones) — restores zones from persistence
  - is_circuit_breaker_open(feature) — checks if a circuit breaker is open
  - get_circuit_breaker_status()  — returns full status dict for all stages
  - get_degradation_level()       — returns simplified 3-level degradation string

All 6 methods replace bridge-side private attribute access:
  self._sr2._conversation.zones()
  self._sr2._conversation.get_zone_transitions()
  self._sr2._conversation.restore_zones()
  self._sr2._engine._circuit_breaker.is_open()
  self._sr2._engine._circuit_breaker.status()
  (degradation logic from BridgeEngine.degradation_level property)
"""

import tempfile

import pytest

from sr2.config.models import MemoryConfig, PipelineConfig, RetrievalConfig, SummarizationConfig
from sr2.pipeline.conversation import ConversationZones
from sr2.sr2 import SR2, SR2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_sr2(
    config_dir: str,
    agent_yaml: dict | None = None,
    preloaded_config: PipelineConfig | None = None,
) -> SR2:
    """Build a minimal SR2 instance for testing, using preloaded config."""
    if preloaded_config is None:
        preloaded_config = PipelineConfig(
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )

    return SR2(
        SR2Config(
            config_dir=config_dir,
            agent_yaml=agent_yaml or {},
            preloaded_config=preloaded_config,
        )
    )


# ---------------------------------------------------------------------------
# Tests: method existence
# ---------------------------------------------------------------------------


class TestPublicApiMethodsExist:
    """All 6 new public methods must exist and be callable on the SR2 class."""

    def test_get_zones_exists(self):
        assert hasattr(SR2, "get_zones"), "SR2 must have a get_zones() method"

    def test_get_zones_is_callable(self):
        assert callable(getattr(SR2, "get_zones", None))

    def test_get_zone_transitions_exists(self):
        assert hasattr(SR2, "get_zone_transitions"), (
            "SR2 must have a get_zone_transitions() method"
        )

    def test_get_zone_transitions_is_callable(self):
        assert callable(getattr(SR2, "get_zone_transitions", None))

    def test_restore_zones_exists(self):
        assert hasattr(SR2, "restore_zones"), "SR2 must have a restore_zones() method"

    def test_restore_zones_is_callable(self):
        assert callable(getattr(SR2, "restore_zones", None))

    def test_is_circuit_breaker_open_exists(self):
        assert hasattr(SR2, "is_circuit_breaker_open"), (
            "SR2 must have an is_circuit_breaker_open() method"
        )

    def test_is_circuit_breaker_open_is_callable(self):
        assert callable(getattr(SR2, "is_circuit_breaker_open", None))

    def test_get_circuit_breaker_status_exists(self):
        assert hasattr(SR2, "get_circuit_breaker_status"), (
            "SR2 must have a get_circuit_breaker_status() method"
        )

    def test_get_circuit_breaker_status_is_callable(self):
        assert callable(getattr(SR2, "get_circuit_breaker_status", None))

    def test_get_degradation_level_exists(self):
        assert hasattr(SR2, "get_degradation_level"), (
            "SR2 must have a get_degradation_level() method"
        )

    def test_get_degradation_level_is_callable(self):
        assert callable(getattr(SR2, "get_degradation_level", None))


# ---------------------------------------------------------------------------
# Tests: get_zones()
# ---------------------------------------------------------------------------


class TestGetZones:
    """get_zones(session_id) must return the ConversationZones for a session."""

    def test_returns_conversation_zones_instance(self):
        """Return type must be ConversationZones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_zones("test_session")
            assert isinstance(result, ConversationZones), (
                f"Expected ConversationZones, got {type(result)}"
            )

    def test_default_session_id(self):
        """get_zones() with no session_id arg must not raise and return ConversationZones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_zones()
            assert isinstance(result, ConversationZones)

    def test_fresh_session_has_empty_zones(self):
        """A session that has never been used must return empty zones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            zones = sr2.get_zones("brand_new_session")
            assert zones.raw == []
            assert zones.compacted == []
            assert zones.summarized == []

    def test_returns_same_object_as_conversation_manager(self):
        """get_zones() must return the same ConversationZones as _conversation.zones()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "compare_session"
            public = sr2.get_zones(session_id)
            private = sr2._conversation.zones(session_id)
            assert public is private, (
                "get_zones() must return the same object as _conversation.zones()"
            )

    def test_different_sessions_return_different_zones(self):
        """Different session_ids must return different ConversationZones objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            zones_a = sr2.get_zones("session_a")
            zones_b = sr2.get_zones("session_b")
            assert zones_a is not zones_b

    def test_session_id_argument_accepted(self):
        """Positional and keyword session_id must both work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            # Positional
            zones_pos = sr2.get_zones("pos_session")
            # Keyword
            zones_kw = sr2.get_zones(session_id="pos_session")
            assert zones_pos is zones_kw


# ---------------------------------------------------------------------------
# Tests: get_zone_transitions()
# ---------------------------------------------------------------------------


class TestGetZoneTransitions:
    """get_zone_transitions(session_id) must return a dict of transition counts."""

    def test_returns_dict(self):
        """Return type must be dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_zone_transitions("session_x")
            assert isinstance(result, dict), (
                f"Expected dict, got {type(result)}"
            )

    def test_fresh_session_returns_empty_dict(self):
        """A session with no transitions must return an empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_zone_transitions("fresh_session")
            assert result == {}

    def test_default_session_id(self):
        """Calling with no session_id must not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_zone_transitions()
            assert isinstance(result, dict)

    def test_matches_conversation_manager_transitions(self):
        """Must return the same data as _conversation.get_zone_transitions()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "transition_compare"
            public = sr2.get_zone_transitions(session_id)
            private = sr2._conversation.get_zone_transitions(session_id)
            assert public == private

    def test_returns_copy_not_reference(self):
        """Mutating the returned dict must not affect the internal state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "copy_test"
            transitions = sr2.get_zone_transitions(session_id)
            # Mutate the returned dict
            transitions["raw_to_compacted"] = 999
            # Internal state must be unchanged
            transitions_again = sr2.get_zone_transitions(session_id)
            assert transitions_again.get("raw_to_compacted", 0) != 999

    def test_dict_values_are_ints(self):
        """All values in the transition dict must be ints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_zone_transitions("check_types")
            for key, value in result.items():
                assert isinstance(value, int), (
                    f"Transition count for '{key}' must be int, got {type(value)}"
                )


# ---------------------------------------------------------------------------
# Tests: restore_zones()
# ---------------------------------------------------------------------------


class TestRestoreZones:
    """restore_zones(session_id, zones) must restore previously persisted zones."""

    def test_restore_zones_returns_none(self):
        """restore_zones() must return None (no return value)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            zones = ConversationZones(summarized=["A summary"])
            result = sr2.restore_zones("restore_session", zones)
            assert result is None

    def test_restored_zones_are_retrievable_via_get_zones(self):
        """After restore_zones(), get_zones() must return the restored zones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "restore_test"
            original_zones = ConversationZones(
                summarized=["Summary of previous session"],
            )
            sr2.restore_zones(session_id, original_zones)
            retrieved = sr2.get_zones(session_id)
            assert retrieved is original_zones, (
                "get_zones() must return the exact zones object that was restored"
            )

    def test_restore_replaces_existing_zones(self):
        """Restoring to an existing session_id must replace the current zones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "overwrite_test"
            # First touch — creates empty zones
            original = sr2.get_zones(session_id)
            assert original.summarized == []

            # Restore new zones with content
            new_zones = ConversationZones(summarized=["Persisted summary"])
            sr2.restore_zones(session_id, new_zones)

            # get_zones() must now return the new zones
            current = sr2.get_zones(session_id)
            assert current is new_zones
            assert current.summarized == ["Persisted summary"]

    def test_restore_preserves_summarized_content(self):
        """Restored zones must preserve summarized, compacted, and raw content."""
        from sr2.compaction.engine import ConversationTurn

        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "content_test"
            turn = ConversationTurn(turn_number=1, role="user", content="Hello")
            persisted = ConversationZones(
                summarized=["Old summary"],
                compacted=[turn],
                raw=[],
            )
            sr2.restore_zones(session_id, persisted)
            retrieved = sr2.get_zones(session_id)
            assert retrieved.summarized == ["Old summary"]
            assert len(retrieved.compacted) == 1
            assert retrieved.compacted[0].content == "Hello"

    def test_delegates_to_conversation_manager(self):
        """restore_zones() must delegate to _conversation.restore_zones()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            session_id = "delegate_test"
            zones = ConversationZones(summarized=["Test"])
            sr2.restore_zones(session_id, zones)
            # Verify via the private attribute directly
            assert sr2._conversation.zones(session_id) is zones


# ---------------------------------------------------------------------------
# Tests: is_circuit_breaker_open()
# ---------------------------------------------------------------------------


class TestIsCircuitBreakerOpen:
    """is_circuit_breaker_open(feature) must check the circuit breaker state."""

    def test_returns_bool(self):
        """Return type must be bool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.is_circuit_breaker_open("summarization")
            assert isinstance(result, bool), (
                f"Expected bool, got {type(result)}"
            )

    def test_fresh_breaker_is_closed(self):
        """A feature with no failures must report as closed (False)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            assert sr2.is_circuit_breaker_open("summarization") is False
            assert sr2.is_circuit_breaker_open("memory_extraction") is False

    def test_unknown_feature_is_closed(self):
        """An unknown feature name must be closed (False), not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.is_circuit_breaker_open("nonexistent_feature_xyz")
            assert result is False

    def test_open_after_threshold_failures(self):
        """A feature must be open after recording threshold-many failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            # Default threshold is 3 (from DegradationConfig default)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            assert sr2.is_circuit_breaker_open("summarization") is True

    def test_closed_after_success(self):
        """A feature that was opened must be closed after recording success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            assert sr2.is_circuit_breaker_open("summarization") is True

            breaker.record_success("summarization")
            assert sr2.is_circuit_breaker_open("summarization") is False

    def test_matches_circuit_breaker_is_open(self):
        """Must return the same value as _engine._circuit_breaker.is_open()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            feature = "memory_extraction"
            public = sr2.is_circuit_breaker_open(feature)
            private = sr2._engine._circuit_breaker.is_open(feature)
            assert public == private

    def test_different_features_are_independent(self):
        """Opening one feature's breaker must not affect other features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            assert sr2.is_circuit_breaker_open("summarization") is True
            assert sr2.is_circuit_breaker_open("memory_extraction") is False


# ---------------------------------------------------------------------------
# Tests: get_circuit_breaker_status()
# ---------------------------------------------------------------------------


class TestGetCircuitBreakerStatus:
    """get_circuit_breaker_status() must return the full status dict."""

    def test_returns_dict(self):
        """Return type must be dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_circuit_breaker_status()
            assert isinstance(result, dict), (
                f"Expected dict, got {type(result)}"
            )

    def test_empty_when_no_failures(self):
        """With no failures recorded, the status dict must be empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_circuit_breaker_status()
            assert result == {}

    def test_contains_stage_after_failure(self):
        """After a failure, the affected stage must appear in the status dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            sr2._engine._circuit_breaker.record_failure("summarization")
            status = sr2.get_circuit_breaker_status()
            assert "summarization" in status

    def test_stage_entry_is_dict_with_expected_keys(self):
        """Each stage entry in the status must have 'failures' and 'is_open' keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            sr2._engine._circuit_breaker.record_failure("summarization")
            status = sr2.get_circuit_breaker_status()
            stage = status["summarization"]
            assert "failures" in stage, "Stage status must have 'failures' key"
            assert "is_open" in stage, "Stage status must have 'is_open' key"

    def test_open_stage_reports_is_open_true(self):
        """An open breaker must report is_open=True in status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            status = sr2.get_circuit_breaker_status()
            assert status["summarization"]["is_open"] is True
            assert status["summarization"]["failures"] == 3

    def test_matches_circuit_breaker_status(self):
        """Must return the same data as _engine._circuit_breaker.status()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            sr2._engine._circuit_breaker.record_failure("memory_extraction")
            public = sr2.get_circuit_breaker_status()
            private = sr2._engine._circuit_breaker.status()
            assert public == private

    def test_multiple_stages_in_status(self):
        """Status must include all stages that have been tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            breaker.record_failure("summarization")
            breaker.record_failure("memory_extraction")
            status = sr2.get_circuit_breaker_status()
            assert "summarization" in status
            assert "memory_extraction" in status


# ---------------------------------------------------------------------------
# Tests: get_degradation_level()
# ---------------------------------------------------------------------------


class TestGetDegradationLevel:
    """get_degradation_level() returns a simplified 3-level heuristic string.

    This is a simplified heuristic (not the full 5-level DegradationLadder):
      'full'             — all breakers closed
      'compaction_only'  — summarization OR memory_extraction is open
      'passthrough'      — both summarization AND memory_extraction are open

    Note: The method lives in SR2 because degradation semantics belong to the
    core library, not the bridge.
    """

    def test_returns_string(self):
        """Return type must be str."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_degradation_level()
            assert isinstance(result, str), (
                f"Expected str, got {type(result)}"
            )

    def test_returns_full_when_no_failures(self):
        """With no circuit breakers open, level must be 'full'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            assert sr2.get_degradation_level() == "full"

    def test_returns_compaction_only_when_summarization_open(self):
        """Level must be 'compaction_only' when only summarization breaker is open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            assert sr2.get_degradation_level() == "compaction_only"

    def test_returns_compaction_only_when_memory_extraction_open(self):
        """Level must be 'compaction_only' when only memory_extraction breaker is open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("memory_extraction")
            assert sr2.get_degradation_level() == "compaction_only"

    def test_returns_passthrough_when_both_open(self):
        """Level must be 'passthrough' when both summarization and memory_extraction are open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            for _ in range(3):
                breaker.record_failure("memory_extraction")
            assert sr2.get_degradation_level() == "passthrough"

    def test_unrelated_breaker_does_not_change_level(self):
        """An open breaker for an unrelated feature must not change the level from 'full'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("scope_detection")
            # Only summarization + memory_extraction affect the level
            assert sr2.get_degradation_level() == "full"

    def test_recovery_from_compaction_only_to_full(self):
        """After recording success, level must return to 'full'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            assert sr2.get_degradation_level() == "compaction_only"

            breaker.record_success("summarization")
            assert sr2.get_degradation_level() == "full"

    def test_recovery_from_passthrough_to_compaction_only(self):
        """Closing one of two open breakers must downgrade from 'passthrough' to 'compaction_only'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            for _ in range(3):
                breaker.record_failure("memory_extraction")
            assert sr2.get_degradation_level() == "passthrough"

            breaker.record_success("summarization")
            assert sr2.get_degradation_level() == "compaction_only"

    def test_valid_levels_only(self):
        """get_degradation_level() must only return one of the 3 valid level strings."""
        valid_levels = {"full", "compaction_only", "passthrough"}
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            assert sr2.get_degradation_level() in valid_levels

            breaker = sr2._engine._circuit_breaker
            for _ in range(3):
                breaker.record_failure("summarization")
            assert sr2.get_degradation_level() in valid_levels

            for _ in range(3):
                breaker.record_failure("memory_extraction")
            assert sr2.get_degradation_level() in valid_levels
