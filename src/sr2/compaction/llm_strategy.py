"""LLM-powered compaction strategy that produces structured analysis + narrative summary."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from sr2.compaction.engine import ConversationTurn

logger = logging.getLogger(__name__)

# Type alias for async LLM callable
LLMCallable = Callable[[str, str], Awaitable[str]]


@dataclass
class CompactionAnalysis:
    """Structured analysis from LLM compaction — preserves decision context."""

    decisions: list[str] = field(default_factory=list)
    current_state: str = ""
    open_questions: list[str] = field(default_factory=list)
    key_context: list[str] = field(default_factory=list)


@dataclass
class LLMCompactionResult:
    """Result of LLM-powered compaction of a set of turns."""

    summary: str
    analysis: CompactionAnalysis
    original_tokens: int
    compacted_tokens: int


_SYSTEM_PROMPT = """\
You are a conversation compactor. Given a sequence of conversation turns, produce TWO outputs:

1. **analysis**: A structured JSON object with these fields:
   - decisions (list[str]): Key decisions made and their reasoning
   - current_state (str): Current state of the task/conversation
   - open_questions (list[str]): Unresolved items or questions
   - key_context (list[str]): Important facts, constraints, or context that must be preserved

2. **summary**: A concise narrative summary of what happened in these turns.

Respond with a JSON object:
{
  "analysis": { "decisions": [...], "current_state": "...", "open_questions": [...], "key_context": [...] },
  "summary": "..."
}

Preserve decision reasoning. Drop routine actions and confirmations. Be concise."""


class LLMCompactionStrategy:
    """Compacts conversation turns using an LLM to produce analysis + summary."""

    def __init__(
        self,
        llm_callable: LLMCallable,
        max_output_tokens: int = 1000,
    ):
        self._llm = llm_callable
        self._max_output_tokens = max_output_tokens

    async def compact(self, turns: list[ConversationTurn]) -> LLMCompactionResult:
        """Compact a list of turns into structured analysis + narrative summary."""
        turns_text = "\n".join(f"[Turn {t.turn_number}] {t.role}: {t.content}" for t in turns)
        original_tokens = sum(len(t.content) // 4 for t in turns)

        prompt = (
            f"Compact the following {len(turns)} conversation turns "
            f"(~{original_tokens} tokens) into analysis + summary.\n\n{turns_text}"
        )

        try:
            response = await self._llm(_SYSTEM_PROMPT, prompt)
            analysis, summary = self._parse_response(response)
        except Exception:
            logger.warning("LLM compaction failed, falling back to simple summary", exc_info=True)
            summary = f"[{len(turns)} turns, ~{original_tokens} tokens — LLM compaction failed]"
            analysis = CompactionAnalysis()

        compacted_tokens = len(summary) // 4
        return LLMCompactionResult(
            summary=summary,
            analysis=analysis,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
        )

    def _parse_response(self, response: str) -> tuple[CompactionAnalysis, str]:
        """Parse LLM JSON response into analysis + summary."""
        data = json.loads(response)
        analysis_data = data.get("analysis", {})
        analysis = CompactionAnalysis(
            decisions=analysis_data.get("decisions", []),
            current_state=analysis_data.get("current_state", ""),
            open_questions=analysis_data.get("open_questions", []),
            key_context=analysis_data.get("key_context", []),
        )
        summary = data.get("summary", "")
        return analysis, summary
