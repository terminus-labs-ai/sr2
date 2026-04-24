"""Tests for summarization engine."""

import json

import pytest

from sr2.config.models import SummarizationConfig
from sr2.summarization.engine import (
    StructuredSummary,
    SummarizationEngine,
)


class TestSummarizationEngine:
    """Tests for SummarizationEngine."""

    @pytest.mark.asyncio
    async def test_structured_format_returns_structured_summary(self):
        """Structured format returns StructuredSummary with parsed fields."""
        response = json.dumps({
            "summary_of_turns": "1-10",
            "key_decisions": ["Use Python"],
            "unresolved": ["Database choice"],
            "facts": ["User is Alice"],
            "user_preferences": ["Dark mode"],
            "errors_encountered": [],
        })

        async def mock_llm(system: str, prompt: str) -> str:
            return response

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns text", "1-10", original_tokens=500)

        assert isinstance(result.summary, StructuredSummary)
        assert result.summary.key_decisions == ["Use Python"]
        assert result.summary.facts == ["User is Alice"]

    @pytest.mark.asyncio
    async def test_prose_format_returns_string(self):
        """Prose format returns plain string."""
        async def mock_llm(system: str, prompt: str) -> str:
            return "The user discussed Python programming."

        config = SummarizationConfig(output_format="prose")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns text", "1-10", original_tokens=500)

        assert isinstance(result.summary, str)
        assert "Python" in result.summary

    @pytest.mark.asyncio
    async def test_malformed_json_fallback(self):
        """Malformed JSON returns StructuredSummary with empty lists and raw_text."""
        async def mock_llm(system: str, prompt: str) -> str:
            return "not valid json {{"

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns", "1-5", original_tokens=200)

        assert isinstance(result.summary, StructuredSummary)
        assert result.summary.key_decisions == []
        assert result.summary.raw_text == "not valid json {{"

    @pytest.mark.asyncio
    async def test_json_in_markdown_fences(self):
        """JSON in markdown fences parsed correctly."""
        response = '```json\n{"key_decisions": ["Deploy"], "unresolved": []}\n```'

        async def mock_llm(system: str, prompt: str) -> str:
            return response

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns", "1-5", original_tokens=200)

        assert isinstance(result.summary, StructuredSummary)
        assert result.summary.key_decisions == ["Deploy"]

    def test_should_trigger_above_threshold(self):
        """should_trigger() returns True when tokens exceed threshold."""
        config = SummarizationConfig(enabled=True, trigger="token_threshold", threshold=0.75)
        engine = SummarizationEngine(config=config)

        assert engine.should_trigger(compacted_tokens=8000, max_tokens=10000) is True

    def test_should_trigger_below_threshold(self):
        """should_trigger() returns False when below threshold."""
        config = SummarizationConfig(trigger="token_threshold", threshold=0.75)
        engine = SummarizationEngine(config=config)

        assert engine.should_trigger(compacted_tokens=5000, max_tokens=10000) is False

    @pytest.mark.asyncio
    async def test_correct_token_counts(self):
        """SummarizationResult reports correct token counts."""
        async def mock_llm(system: str, prompt: str) -> str:
            return "A brief summary."

        config = SummarizationConfig(output_format="prose")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("long text", "1-20", original_tokens=1000)

        assert result.original_tokens == 1000
        assert result.summary_tokens > 0
        assert result.turn_range == "1-20"

    @pytest.mark.asyncio
    async def test_think_block_json_parsed_correctly(self):
        """<think> prefix is stripped and JSON fields parsed correctly."""
        payload = {"key_decisions": ["Use Redis"], "facts": ["Cache hit rate 90%"], "unresolved": []}
        response = f"<think>internal reasoning</think>\n{json.dumps(payload)}"

        async def mock_llm(system: str, prompt: str) -> str:
            return response

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns text", "1-10", original_tokens=500)

        assert isinstance(result.summary, StructuredSummary)
        assert result.summary.key_decisions == ["Use Redis"]
        assert result.summary.facts == ["Cache hit rate 90%"]
        assert result.summary.raw_text == response

    @pytest.mark.asyncio
    async def test_thinking_block_and_fence_parsed_correctly(self):
        """<thinking> block + ```json fence both stripped and fields parsed."""
        payload = {"key_decisions": ["Deploy to prod"], "unresolved": ["Cost estimate"]}
        response = (
            "<thinking>\nreasoning step 1\nreasoning step 2\n</thinking>\n"
            f"```json\n{json.dumps(payload)}\n```"
        )

        async def mock_llm(system: str, prompt: str) -> str:
            return response

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns text", "1-10", original_tokens=500)

        assert isinstance(result.summary, StructuredSummary)
        assert result.summary.key_decisions == ["Deploy to prod"]
        assert result.summary.unresolved == ["Cost estimate"]
        assert result.summary.raw_text == response

    @pytest.mark.asyncio
    async def test_missing_fields_default_to_empty(self):
        """Missing fields in JSON default to empty lists."""
        response = json.dumps({"key_decisions": ["Only field"]})

        async def mock_llm(system: str, prompt: str) -> str:
            return response

        config = SummarizationConfig(output_format="structured")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)
        result = await engine.summarize("turns", "1-5", original_tokens=200)

        assert result.summary.key_decisions == ["Only field"]
        assert result.summary.unresolved == []
        assert result.summary.facts == []
        assert result.summary.user_preferences == []
        assert result.summary.errors_encountered == []


class TestSummarizationTriggerTypes:
    """Tests for trigger type behavior: token_threshold, topic_shift, manual."""

    def test_manual_trigger_never_auto_triggers(self):
        """With trigger='manual', should_trigger() always returns False."""
        config = SummarizationConfig(trigger="manual", threshold=0.1)
        engine = SummarizationEngine(config=config)

        # Even with tokens far exceeding threshold, manual trigger never fires
        assert engine.should_trigger(compacted_tokens=9999, max_tokens=100) is False
        assert engine.should_trigger(compacted_tokens=100, max_tokens=100) is False
        assert engine.should_trigger(compacted_tokens=0, max_tokens=100) is False

    @pytest.mark.asyncio
    async def test_manual_trigger_summarize_still_works_when_called_explicitly(self):
        """With trigger='manual', explicit summarize() call still works."""
        async def mock_llm(system: str, prompt: str) -> str:
            return "Manual summary of the conversation."

        config = SummarizationConfig(trigger="manual", output_format="prose")
        engine = SummarizationEngine(config=config, llm_callable=mock_llm)

        # should_trigger is False
        assert engine.should_trigger(compacted_tokens=9999, max_tokens=100) is False

        # But explicit summarize() works fine
        result = await engine.summarize("conversation text", "1-10", original_tokens=500)
        assert isinstance(result.summary, str)
        assert "Manual summary" in result.summary

    def test_topic_shift_trigger_does_not_respond_to_token_count(self):
        """With trigger='topic_shift', token counts alone don't trigger summarization."""
        config = SummarizationConfig(trigger="topic_shift", threshold=0.1)
        engine = SummarizationEngine(config=config)

        # Even with tokens way over threshold, topic_shift doesn't respond to token counts
        assert engine.should_trigger(compacted_tokens=9999, max_tokens=100) is False
        assert engine.should_trigger(compacted_tokens=100, max_tokens=100) is False

    def test_disabled_engine_never_triggers(self):
        """With enabled=False, no trigger type fires."""
        for trigger in ("token_threshold", "topic_shift", "manual"):
            config = SummarizationConfig(enabled=False, trigger=trigger, threshold=0.1)
            engine = SummarizationEngine(config=config)
            assert engine.should_trigger(compacted_tokens=9999, max_tokens=100) is False, (
                f"Disabled engine with trigger={trigger} should never trigger"
            )
