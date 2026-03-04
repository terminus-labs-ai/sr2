"""LLM-powered conversation summarization engine."""

import logging

import json
from dataclasses import dataclass

from sr2.config.models import SummarizationConfig
from sr2.normalization import ResponseNormalizer
from sr2.summarization.prompts import SummarizationPromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class StructuredSummary:
    """Parsed structured summary."""

    summary_of_turns: str
    key_decisions: list[str]
    unresolved: list[str]
    facts: list[str]
    user_preferences: list[str]
    errors_encountered: list[str]
    raw_text: str  # The raw LLM output for fallback


@dataclass
class SummarizationResult:
    """Result of a summarization pass."""

    summary: StructuredSummary | str  # StructuredSummary if structured, str if prose
    original_tokens: int
    summary_tokens: int
    turn_range: str


class SummarizationEngine:
    """LLM-powered conversation summarization."""

    def __init__(
        self,
        config: SummarizationConfig,
        llm_callable=None,
    ):
        """Args:
        config: Summarization configuration.
        llm_callable: async function(system: str, prompt: str) -> str
        """
        self._config = config
        self._llm = llm_callable
        self._prompt_builder = SummarizationPromptBuilder(config)
        self._normalizer = ResponseNormalizer()

    @property
    def preserve_recent_turns(self) -> int:
        """Number of recent compacted turns to exclude from summarization."""
        return self._config.preserve_recent_turns

    async def summarize(
        self,
        turns_text: str,
        turn_range: str,
        original_tokens: int,
    ) -> SummarizationResult:
        """Summarize conversation turns.

        1. Build prompt
        2. Call LLM
        3. Parse response (structured or prose)
        4. Return SummarizationResult
        """
        system = self._prompt_builder.build_system_prompt()
        prompt = self._prompt_builder.build_prompt(turns_text, turn_range)

        raw_response = await self._llm(system, prompt)

        if raw_response is None:
            logger.warning("Summarization LLM returned None — skipping")
            return SummarizationResult(
                summary=turns_text[:200],
                original_tokens=original_tokens,
                summary_tokens=original_tokens,
                turn_range=turn_range,
            )

        if self._config.output_format == "structured":
            summary = self._parse_structured(raw_response, turn_range)
        else:
            summary = raw_response.strip()

        summary_tokens = len(str(summary)) // 4

        logger.info(
            f"Summarization completed - went from {original_tokens} to {summary_tokens}, throughout {turn_range} turns"
        )
        return SummarizationResult(
            summary=summary,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            turn_range=turn_range,
        )

    def _parse_structured(self, raw: str, turn_range: str) -> StructuredSummary:
        """Parse LLM response as structured JSON.

        Handles:
        - Clean JSON
        - JSON in markdown code fences
        - Malformed JSON -> fallback to raw text in StructuredSummary
        """
        cleaned = self._normalizer.normalize(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return StructuredSummary(
                summary_of_turns=turn_range,
                key_decisions=[],
                unresolved=[],
                facts=[],
                user_preferences=[],
                errors_encountered=[],
                raw_text=raw,
            )

        return StructuredSummary(
            summary_of_turns=data.get("summary_of_turns", turn_range),
            key_decisions=data.get("key_decisions", []),
            unresolved=data.get("unresolved", []),
            facts=data.get("facts", []),
            user_preferences=data.get("user_preferences", []),
            errors_encountered=data.get("errors_encountered", []),
            raw_text=raw,
        )

    def should_trigger(self, compacted_tokens: int, max_tokens: int) -> bool:
        """Check if summarization should trigger based on config.

        Returns True if enabled and compacted_tokens exceeds threshold * max_tokens.
        """
        if not self._config.enabled:
            return False
        if self._config.trigger == "token_threshold":
            return compacted_tokens > (self._config.threshold * max_tokens)
        return False
