from typing import Literal

DegradationLevel = Literal[
    "full",
    "skip_summarization",
    "skip_intent",
    "raw_context",
    "system_prompt_only",
]

DEGRADATION_ORDER: list[DegradationLevel] = [
    "full", "skip_summarization", "skip_intent", "raw_context", "system_prompt_only",
]


class DegradationLadder:
    """Tracks current degradation level and determines which stages to skip."""

    def __init__(self):
        self._level: DegradationLevel = "full"

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def degrade(self) -> DegradationLevel:
        """Move one step down the ladder. Returns new level."""
        idx = DEGRADATION_ORDER.index(self._level)
        if idx < len(DEGRADATION_ORDER) - 1:
            self._level = DEGRADATION_ORDER[idx + 1]
        return self._level

    def reset(self) -> None:
        """Reset to full pipeline."""
        self._level = "full"

    def should_skip(self, stage: str) -> bool:
        """Return True if the given stage should be skipped at current level."""
        skip_map: dict[DegradationLevel, set[str]] = {
            "full": set(),
            "skip_summarization": {"summarization"},
            "skip_intent": {"summarization", "intent_detection"},
            "raw_context": {"summarization", "intent_detection", "retrieval", "compaction"},
            "system_prompt_only": {"summarization", "intent_detection", "retrieval", "compaction", "session", "memory"},
        }
        return stage in skip_map.get(self._level, set())
