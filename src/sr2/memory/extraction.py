"""Rule-based memory extraction from conversation turns.

Extracts factual statements, preferences, and decisions from assistant
or user text into Memory objects. Uses heuristic patterns — designed to
be swapped for an LLM-driven extractor later.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from .protocol import MemoryExtractor
from .schema import ExtractionResult, Memory, MemoryScope

# Patterns that signal durable facts worth remembering.
# Each tuple: (compiled regex, scope, default tags)
_FACT_PATTERNS: list[tuple[re.Pattern[str], MemoryScope, list[str]]] = [
    # User preferences/corrections
    (re.compile(r"(?:remember|don't forget|i (?:prefer|like|don't like|want))\b(.{5,100})", re.IGNORECASE),
     MemoryScope.SHARED, ["preference"]),
    # Decisions made
    (re.compile(r"(?:decided|chose|went with|settled on)\b\s+(.{5,100})", re.IGNORECASE),
     MemoryScope.PROJECT, ["decision"]),
    # Corrections/feedback
    (re.compile(r"(?:don't do that again|not (?:like|the way)|wrong\s+(?:approach|way))\b(.{5,100})", re.IGNORECASE),
     MemoryScope.SHARED, ["correction"]),
    # Explicit facts stated
    (re.compile(r"(?:the (?:project|code|system|repo)\s+)?(?:uses?|has|is|runs)\s+(.{5,100})", re.IGNORECASE),
     MemoryScope.PROJECT, ["fact"]),
    # Configuration/tooling facts
    (re.compile(r"(?:installed|configured|set up|enabled|disabled)\s+(.{5,100})", re.IGNORECASE),
     MemoryScope.PROJECT, ["tooling"]),
]

# Minimum content length to extract — avoid noise
MIN_EXTRACT_LEN = 10

# Maximum memories to extract from a single turn
MAX_EXTRACT_PER_TURN = 5


class RuleBasedExtractor(MemoryExtractor):
    """Extract memories from text using regex patterns.

    For each pattern match, deduplicates by content similarity (substring
    overlap) before yielding.
    """

    def extract(
        self,
        turn_text: str,
        turn_id: str | None = None,
        scope_override: MemoryScope | None = None,
    ) -> ExtractionResult:
        """Extract durable facts from a conversation turn."""
        if not turn_text or len(turn_text) < MIN_EXTRACT_LEN:
            return ExtractionResult(source_turn_id=turn_id)

        extracted: list[Memory] = []
        now = datetime.now(timezone.utc)

        for pattern, scope, default_tags in _FACT_PATTERNS:
            if len(extracted) >= MAX_EXTRACT_PER_TURN:
                break

            for m in pattern.finditer(turn_text):
                content = m.group(0).strip()
                # Clean up: remove trailing punctuation noise
                content = re.sub(r"[.,;:!?]{2,}$", "", content).strip()

                if len(content) < MIN_EXTRACT_LEN:
                    continue

                # Deduplicate against already-extracted memories
                if self._similar_to_existing(content, extracted):
                    continue

                memory = Memory(
                    content=content,
                    scope=scope_override or scope,
                    tags=default_tags,
                    created_at=now,
                    last_accessed=now,
                )
                extracted.append(memory)

        return ExtractionResult(
            memories=extracted,
            source_turn_id=turn_id,
            metadata={"extractor": "rule_based", "pattern_count": len(_FACT_PATTERNS)},
        )

    @staticmethod
    def _similar_to_existing(content: str, existing: list[Memory]) -> bool:
        """Check if content overlaps significantly with already-extracted memories."""
        content_lower = content.lower()
        for mem in existing:
            mem_lower = mem.content.lower()
            # Substring containment or high overlap
            if content_lower in mem_lower or mem_lower in content_lower:
                return True
            # Token overlap > 70%
            content_words = set(content_lower.split())
            mem_words = set(mem_lower.split())
            if content_words and mem_words:
                overlap = len(content_words & mem_words)
                union = len(content_words | mem_words)
                if overlap / union > 0.7:
                    return True
        return False
