import pytest

from sr2.degradation import DEGRADATION_ORDER
from sr2.degradation.ladder import DegradationLadder


class TestDegradationLadder:
    """Tests for DegradationLadder."""

    def test_initial_level_is_full(self):
        ladder = DegradationLadder()
        assert ladder.level == "full"

    def test_degrade_moves_through_levels_in_order(self):
        ladder = DegradationLadder()
        for expected in DEGRADATION_ORDER[1:]:
            result = ladder.degrade()
            assert result == expected
            assert ladder.level == expected

    def test_degrade_at_bottom_stays_at_bottom(self):
        ladder = DegradationLadder()
        # Move to the bottom level
        for _ in range(len(DEGRADATION_ORDER) - 1):
            ladder.degrade()
        assert ladder.level == "system_prompt_only"
        # Degrading again should stay at the bottom
        result = ladder.degrade()
        assert result == "system_prompt_only"
        assert ladder.level == "system_prompt_only"

    def test_reset_returns_to_full(self):
        ladder = DegradationLadder()
        ladder.degrade()
        ladder.degrade()
        assert ladder.level != "full"
        ladder.reset()
        assert ladder.level == "full"

    def test_should_skip_summarization(self):
        ladder = DegradationLadder()
        assert ladder.should_skip("summarization") is False
        ladder.degrade()  # now at "skip_summarization"
        assert ladder.should_skip("summarization") is True

    def test_should_skip_retrieval(self):
        ladder = DegradationLadder()
        # Move to "skip_intent"
        ladder.degrade()  # skip_summarization
        ladder.degrade()  # skip_intent
        assert ladder.level == "skip_intent"
        assert ladder.should_skip("retrieval") is False
        ladder.degrade()  # raw_context
        assert ladder.level == "raw_context"
        assert ladder.should_skip("retrieval") is True


# Complete skip matrix from source: level -> set of skipped stages
_SKIP_MATRIX: dict[str, set[str]] = {
    "full": set(),
    "skip_summarization": {"summarization"},
    "skip_intent": {"summarization", "intent_detection"},
    "raw_context": {"summarization", "intent_detection", "retrieval", "compaction"},
    "system_prompt_only": {
        "summarization",
        "intent_detection",
        "retrieval",
        "compaction",
        "session",
        "memory",
    },
}

# All stages that appear anywhere in the skip map
_ALL_STAGES = sorted(
    {"summarization", "intent_detection", "retrieval", "compaction", "session", "memory"}
)

# Build parametrize list: (level, stage, expected_skip)
_LEVEL_STAGE_PARAMS = [
    pytest.param(level, stage, stage in skipped, id=f"{level}-{stage}")
    for level, skipped in _SKIP_MATRIX.items()
    for stage in _ALL_STAGES
]


class TestDegradationLadderMatrix:
    """Exhaustive test: every level x stage combination returns the correct should_skip value."""

    @pytest.mark.parametrize("level,stage,expected_skip", _LEVEL_STAGE_PARAMS)
    def test_should_skip_matrix(self, level: str, stage: str, expected_skip: bool):
        ladder = DegradationLadder()
        # Advance ladder to the target level
        while ladder.level != level:
            ladder.degrade()
        assert ladder.should_skip(stage) is expected_skip
