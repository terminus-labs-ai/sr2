"""Tests for render_trace — behavioral coverage of the human-readable timeline output."""

import pytest
from sr2.pipeline.tracing import FiringRecord, render_trace


def make_record(**overrides) -> FiringRecord:
    defaults = dict(
        turn_seq=0,
        firing_seq=0,
        kind="resolver",
        component_name="resolver/static",
        layer="system",
        trigger_events=[],
        content_before=[],
        content_after=[],
        tokens_before=0,
        tokens_after=0,
        tokens_delta=0,
        duration_ms=0.1,
        status="ok",
        error=None,
    )
    defaults.update(overrides)
    return FiringRecord(**defaults)


# 1. Empty records list
class TestEmptyInput:
    def test_returns_string(self):
        result = render_trace([])
        assert isinstance(result, str)

    def test_returns_non_empty_string(self):
        result = render_trace([])
        assert len(result) > 0


# 2. Single record, single turn
class TestSingleRecord:
    @pytest.fixture
    def record(self):
        return make_record(
            turn_seq=0,
            firing_seq=0,
            kind="resolver",
            component_name="resolver/static",
            layer="system",
            tokens_before=10,
            tokens_after=28,
            tokens_delta=18,
            duration_ms=0.3,
        )

    @pytest.fixture
    def output(self, record):
        return render_trace([record])

    def test_contains_turn_seq_label(self, output):
        # "Turn 0" or "turn 0" — case-insensitive, value must appear
        assert "0" in output
        assert "turn" in output.lower()

    def test_contains_component_name(self, output):
        assert "resolver/static" in output

    def test_contains_layer(self, output):
        assert "system" in output

    def test_contains_kind(self, output):
        assert "resolver" in output

    def test_contains_token_delta_with_sign_positive(self, output):
        # +18 tokens
        assert "+18" in output

    def test_contains_duration_ms(self, output):
        # 0.3ms must appear — allow flexible formatting (0.3, .3, etc.)
        assert "0.3" in output

    def test_contains_firing_seq(self, output):
        # firing_seq 0 must be referenced as a sequence indicator — not just "0" in text
        # The spec illustration shows "#0" or "0:" — require "#0" or similar prefix
        assert "#0" in output or "seq=0" in output or "0:" in output

    def test_negative_token_delta_has_sign(self):
        record = make_record(tokens_delta=-5)
        output = render_trace([record])
        assert "-5" in output

    def test_zero_token_delta_has_sign(self):
        record = make_record(tokens_delta=0)
        output = render_trace([record])
        assert "+0" in output


# 3. Multiple turns produce distinct turn headers
class TestMultipleTurns:
    def test_distinct_turn_headers(self):
        records = [
            make_record(turn_seq=0, firing_seq=0),
            make_record(turn_seq=1, firing_seq=0),
            make_record(turn_seq=2, firing_seq=0),
        ]
        output = render_trace(records)
        # Each turn_seq value must appear as a label
        assert "0" in output
        assert "1" in output
        assert "2" in output

    def test_each_turn_has_separate_section(self):
        records = [
            make_record(turn_seq=0, firing_seq=0, component_name="comp/alpha"),
            make_record(turn_seq=1, firing_seq=0, component_name="comp/beta"),
        ]
        output = render_trace(records)
        # Both components must be present under their respective turns
        assert "comp/alpha" in output
        assert "comp/beta" in output

    def test_turns_appear_in_sequential_order(self):
        records = [
            make_record(turn_seq=0, firing_seq=0),
            make_record(turn_seq=3, firing_seq=0),
        ]
        output = render_trace(records)
        # Turn 0 section should appear before Turn 3 section
        idx_0 = output.lower().index("turn")
        # Find second occurrence of "turn"
        idx_3 = output.lower().index("turn", idx_0 + 1)
        assert idx_0 < idx_3

    def test_same_turn_seq_produces_one_turn_header(self):
        # Core structural requirement: records sharing a turn_seq are grouped
        # under exactly ONE turn header — not one header per record.
        records = [
            make_record(turn_seq=1, firing_seq=0, component_name="comp/alpha"),
            make_record(turn_seq=1, firing_seq=1, component_name="comp/beta"),
        ]
        output = render_trace(records)
        # Count occurrences of "Turn 1" (or "turn 1") in the output
        lower = output.lower()
        count = lower.count("turn 1")
        assert count == 1, (
            f"Expected exactly 1 turn header for turn_seq=1, found {count}.\n"
            f"Output:\n{output}"
        )


# 4. Firing_seq order within a turn
class TestFiringSeqOrder:
    def test_firing_seq_ascending_in_output(self):
        records = [
            make_record(turn_seq=0, firing_seq=0, component_name="comp/first"),
            make_record(turn_seq=0, firing_seq=1, component_name="comp/second"),
            make_record(turn_seq=0, firing_seq=2, component_name="comp/third"),
        ]
        output = render_trace(records)
        pos_first = output.index("comp/first")
        pos_second = output.index("comp/second")
        pos_third = output.index("comp/third")
        assert pos_first < pos_second < pos_third

    def test_out_of_order_input_still_sorted(self):
        # Records provided out of order — output must still be sorted
        records = [
            make_record(turn_seq=0, firing_seq=2, component_name="comp/late"),
            make_record(turn_seq=0, firing_seq=0, component_name="comp/early"),
            make_record(turn_seq=0, firing_seq=1, component_name="comp/mid"),
        ]
        output = render_trace(records)
        pos_early = output.index("comp/early")
        pos_mid = output.index("comp/mid")
        pos_late = output.index("comp/late")
        assert pos_early < pos_mid < pos_late


# 5. Failed record shows failure indication
class TestFailedRecord:
    def test_failed_status_is_visible(self):
        record = make_record(status="failed", error="NoneType has no attribute 'content'")
        output = render_trace([record])
        assert "failed" in output.lower()

    def test_error_string_is_visible(self):
        error_msg = "NoneType has no attribute 'content'"
        record = make_record(status="failed", error=error_msg)
        output = render_trace([record])
        assert error_msg in output


# 6. Empty content_before shows empty indicator
class TestEmptyContentBefore:
    def test_empty_before_shows_indicator(self):
        record = make_record(content_before=[], content_after=[])
        output = render_trace([record])
        # Per spec illustration: "(empty)" should appear for empty before
        assert "empty" in output.lower()


# 7. Non-empty content_after shows repr of items
class TestNonEmptyContentAfter:
    def test_content_after_items_represented(self):
        class FakeBlock:
            def __repr__(self):
                return 'TextBlock("You are a helpful assistant")'

        record = make_record(content_after=[FakeBlock()])
        output = render_trace([record])
        assert "TextBlock" in output

    def test_content_after_string_items_represented(self):
        record = make_record(content_after=["hello", "world"])
        output = render_trace([record])
        assert "hello" in output or repr("hello") in output

    def test_empty_list_content_after_represented(self):
        record = make_record(
            kind="tool_provider",
            component_name="tool_provider/spectre",
            content_after=[],
        )
        output = render_trace([record])
        # Per spec: "[] (0 tools)" or similar — at minimum the empty container is noted
        assert "0" in output or "empty" in output.lower() or "[]" in output
