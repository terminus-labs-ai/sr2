"""Tests for summarization prompt builder."""

from sr2.config.models import SummarizationConfig
from sr2.summarization.prompts import SummarizationPromptBuilder


class TestSummarizationPromptBuilder:
    """Tests for SummarizationPromptBuilder."""

    def test_includes_preserve_items(self):
        """build_prompt() includes preserve items."""
        config = SummarizationConfig()
        builder = SummarizationPromptBuilder(config)
        prompt = builder.build_prompt("some turns", "1-10")

        assert "decisions_and_reasoning" in prompt
        assert "unresolved_issues" in prompt

    def test_includes_discard_items(self):
        """build_prompt() includes discard items."""
        config = SummarizationConfig()
        builder = SummarizationPromptBuilder(config)
        prompt = builder.build_prompt("some turns", "1-10")

        assert "successful_routine_actions" in prompt
        assert "redundant_confirmations" in prompt

    def test_structured_format_includes_json_template(self):
        """build_prompt() with structured format includes JSON template."""
        config = SummarizationConfig(output_format="structured")
        builder = SummarizationPromptBuilder(config)
        prompt = builder.build_prompt("turns", "1-10")

        assert "JSON object" in prompt
        assert "key_decisions" in prompt
        assert "unresolved" in prompt

    def test_prose_format_instruction(self):
        """build_prompt() with prose format includes prose instruction."""
        config = SummarizationConfig(output_format="prose")
        builder = SummarizationPromptBuilder(config)
        prompt = builder.build_prompt("turns", "1-10")

        assert "prose summary" in prompt
        assert "JSON" not in prompt

    def test_includes_turns_text(self):
        """build_prompt() includes the turns text."""
        config = SummarizationConfig()
        builder = SummarizationPromptBuilder(config)
        prompt = builder.build_prompt("User: Hello\nAssistant: Hi there", "1-2")

        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt

    def test_system_prompt_non_empty(self):
        """build_system_prompt() returns a non-empty string."""
        config = SummarizationConfig()
        builder = SummarizationPromptBuilder(config)
        system = builder.build_system_prompt()

        assert len(system) > 0
        assert "summarizer" in system.lower()
