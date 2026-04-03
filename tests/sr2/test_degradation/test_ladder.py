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
