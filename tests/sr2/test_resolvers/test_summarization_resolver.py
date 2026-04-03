"""Tests for summarization resolver."""

import pytest

from sr2.resolvers.registry import ResolverContext
from sr2.resolvers.summarization_resolver import SummarizationResolver
from sr2.summarization.engine import StructuredSummary


@pytest.fixture
def context():
    return ResolverContext(agent_config={}, trigger_input="test")


class TestSummarizationResolver:
    """Tests for SummarizationResolver."""

    @pytest.mark.asyncio
    async def test_no_summaries_empty(self, context):
        """No summaries returns empty content."""
        resolver = SummarizationResolver()
        result = await resolver.resolve("summaries", {}, context)

        assert result.content == ""
        assert result.tokens == 0

    @pytest.mark.asyncio
    async def test_one_summary_returned(self, context):
        """One summary returned as-is."""
        resolver = SummarizationResolver()
        resolver.add_summary("Summary of turns 1-10: User discussed Python.")
        result = await resolver.resolve("summaries", {}, context)

        assert "Summary of turns 1-10" in result.content
        assert result.metadata["summary_count"] == 1

    @pytest.mark.asyncio
    async def test_multiple_summaries_concatenated(self, context):
        """Multiple summaries concatenated with --- separator."""
        resolver = SummarizationResolver()
        resolver.add_summary("Summary 1")
        resolver.add_summary("Summary 2")
        result = await resolver.resolve("summaries", {}, context)

        assert "Summary 1" in result.content
        assert "Summary 2" in result.content
        assert "---" in result.content
        assert result.metadata["summary_count"] == 2

    @pytest.mark.asyncio
    async def test_max_tokens_drops_oldest(self, context):
        """max_tokens exceeded drops oldest summaries."""
        resolver = SummarizationResolver()
        resolver.add_summary("A" * 400)  # ~100 tokens
        resolver.add_summary("B" * 400)  # ~100 tokens
        resolver.add_summary("C" * 40)   # ~10 tokens

        result = await resolver.resolve("summaries", {"max_tokens": 50}, context)

        # Oldest summaries should be dropped
        assert "A" * 400 not in result.content
        assert result.metadata["summary_count"] < 3

    def test_format_structured_summary_all_fields(self):
        """format_structured_summary formats all fields correctly."""
        summary = StructuredSummary(
            summary_of_turns="1-10",
            key_decisions=["Use Python", "Deploy to AWS"],
            unresolved=["Database choice"],
            facts=["User is Alice"],
            user_preferences=["Dark mode"],
            errors_encountered=["Import error"],
            raw_text="raw",
        )
        text = SummarizationResolver.format_structured_summary(summary)

        assert "[Summary of turns 1-10]" in text
        assert "Decisions: Use Python; Deploy to AWS" in text
        assert "Unresolved: Database choice" in text
        assert "Facts: User is Alice" in text
        assert "Preferences: Dark mode" in text
        assert "Errors: Import error" in text

    def test_format_structured_summary_empty_fields(self):
        """format_structured_summary with empty fields only includes non-empty."""
        summary = StructuredSummary(
            summary_of_turns="5-10",
            key_decisions=["One decision"],
            unresolved=[],
            facts=[],
            user_preferences=[],
            errors_encountered=[],
            raw_text="raw",
        )
        text = SummarizationResolver.format_structured_summary(summary)

        assert "Decisions:" in text
        assert "Unresolved:" not in text
        assert "Facts:" not in text
