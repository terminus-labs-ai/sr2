"""Prompt builder for the summarization LLM call."""

from sr2.config.models import SummarizationConfig


class SummarizationPromptBuilder:
    """Builds prompts for the summarization LLM call."""

    def __init__(self, config: SummarizationConfig):
        self._config = config

    def build_prompt(self, turns_text: str, turn_range: str) -> str:
        """Build the summarization prompt.

        Args:
            turns_text: The conversation turns to summarize (already compacted).
            turn_range: e.g. "21-40" for labeling the summary.

        Returns: The full prompt string to send to the LLM.
        """
        preserve_text = "\n".join(f"  - {p}" for p in self._config.preserve)
        discard_text = "\n".join(f"  - {d}" for d in self._config.discard)

        if self._config.output_format == "structured":
            format_instruction = """Output ONLY a JSON object with these fields:
{
  "summary_of_turns": "<turn range>",
  "key_decisions": ["..."],
  "unresolved": ["..."],
  "facts": ["..."],
  "user_preferences": ["..."],
  "errors_encountered": ["..."]
}
No markdown, no explanation, just the JSON object."""
        else:
            format_instruction = "Output a concise prose summary."

        return f"""Summarize these conversation turns ({turn_range}).

PRESERVE (include in summary):
{preserve_text}

DISCARD (omit from summary):
{discard_text}

{format_instruction}

Conversation turns:
{turns_text}"""

    def build_system_prompt(self) -> str:
        """System prompt for the summarization model."""
        return (
            "You are a conversation summarizer. Extract key information accurately. "
            "Never invent information not present in the conversation. "
            "Be concise but preserve all critical details."
        )
