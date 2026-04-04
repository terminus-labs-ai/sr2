"""Tests for sample eval suites."""

import pytest

from sr2.eval.sample_suites import (
    create_coherence_suite,
    create_compaction_suite,
    create_summarization_suite,
    create_all_suites,
)


def test_coherence_suite_creation() -> None:
    """Test creating coherence eval suite."""
    suite = create_coherence_suite()
    assert len(suite) == 2
    assert all(case.id.startswith("coherence_") for case in suite)
    assert all("coherence" in case.tags for case in suite)


def test_coherence_suite_content() -> None:
    """Test coherence suite cases have correct structure."""
    suite = create_coherence_suite()
    for case in suite:
        assert case.system_prompt
        assert len(case.conversation_turns) > 0
        assert case.expected_tokens > 0
        assert case.expected_key_facts or case.expected_decisions


def test_compaction_suite_creation() -> None:
    """Test creating compaction eval suite."""
    suite = create_compaction_suite()
    assert len(suite) == 2
    assert all(case.id.startswith("compaction_") for case in suite)
    assert all("compaction" in case.tags for case in suite)


def test_summarization_suite_creation() -> None:
    """Test creating summarization eval suite."""
    suite = create_summarization_suite()
    assert len(suite) == 2
    assert all(case.id.startswith("summarization_") for case in suite)
    assert all("summarization" in case.tags for case in suite)


def test_all_suites_creation() -> None:
    """Test creating all suites at once."""
    all_suites = create_all_suites()
    assert "coherence" in all_suites
    assert "compaction" in all_suites
    assert "summarization" in all_suites
    assert len(all_suites["coherence"]) == 2
    assert len(all_suites["compaction"]) == 2
    assert len(all_suites["summarization"]) == 2


def test_all_cases_have_unique_ids() -> None:
    """Test that all cases in all suites have unique IDs."""
    all_suites = create_all_suites()
    all_cases = []
    for suite in all_suites.values():
        all_cases.extend(suite)

    case_ids = [case.id for case in all_cases]
    assert len(case_ids) == len(set(case_ids))  # All unique


def test_suite_cases_have_conversation_turns() -> None:
    """Test all cases have realistic conversation turns."""
    all_suites = create_all_suites()
    for suite in all_suites.values():
        for case in suite:
            assert len(case.conversation_turns) >= 2
            for user, assistant in case.conversation_turns:
                assert user  # Non-empty user message
                assert assistant  # Non-empty assistant response
