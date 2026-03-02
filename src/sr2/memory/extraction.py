"""Post-LLM memory extraction from conversation turns."""

import json
import re

from sr2.memory.schema import STABILITY_DEFAULTS, ExtractionResult, Memory
from sr2.memory.store import MemoryStore


class MemoryExtractor:
    """Extracts structured memories from conversation turns using an LLM."""

    def __init__(
        self,
        llm_callable,
        store: MemoryStore,
        key_schema: list[dict] | None = None,
        max_memories_per_turn: int = 5,
    ):
        """Args:
        llm_callable: async function(prompt: str) -> str.
                      Wraps LiteLLM or any LLM call.
        store: MemoryStore to save extracted memories.
        key_schema: list of {"prefix": str, "examples": list[str]}
        max_memories_per_turn: max memories to extract per turn.
        """
        self._llm = llm_callable
        self._store = store
        self._key_schema = key_schema or []
        self._max = max_memories_per_turn

    async def extract(
        self,
        conversation_turn: str,
        conversation_id: str | None = None,
        turn_number: int | None = None,
    ) -> ExtractionResult:
        """Extract memories from a conversation turn.

        1. Build extraction prompt (include key schema for guidance)
        2. Call LLM
        3. Parse response as JSON list of memories
        4. Validate and cap at max_memories_per_turn
        5. Save each memory to store
        6. Return ExtractionResult
        """
        prompt = self._build_prompt(conversation_turn)
        raw_response = await self._llm(prompt)
        memories = self._parse_response(raw_response, conversation_id, turn_number)

        for mem in memories:
            await self._store.save(mem)

        return ExtractionResult(
            memories=memories,
            source_conversation=conversation_id,
            source_turn=turn_number,
        )

    def _build_prompt(self, conversation_turn: str) -> str:
        """Build the extraction prompt.

        The prompt instructs the LLM to:
        - Extract key facts, preferences, and decisions
        - Output ONLY a JSON array of objects with fields: key, value, memory_type, confidence_source
        - Use the key_schema prefixes for guidance
        - Extract at most self._max memories
        - If nothing worth extracting, return an empty array []
        """
        schema_text = ""
        if self._key_schema:
            schema_text = "Use these key prefixes:\n"
            for s in self._key_schema:
                schema_text += f"  - {s['prefix']}: e.g. {', '.join(s.get('examples', []))}\n"

        return f"""Extract structured facts from this conversation turn.
Output ONLY a JSON array. No markdown, no explanation.
Each object: {{"key": "...", "value": "...", "memory_type": "identity|semi_stable|dynamic", "confidence_source": "explicit_statement|direct_answer|contextual_mention|inferred|offhand"}}
{schema_text}
Max {self._max} memories. If nothing to extract, return [].

Conversation turn:
{conversation_turn}"""

    def _parse_response(
        self,
        raw: str,
        conversation_id: str | None,
        turn_number: int | None,
    ) -> list[Memory]:
        """Parse LLM JSON response into Memory objects.

        Handles:
        - Valid JSON array
        - JSON wrapped in markdown code fences
        - Malformed JSON (return empty list, don't crash)
        - More items than max (truncate)
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            return []

        if not isinstance(items, list):
            return []

        memories = []
        for item in items[: self._max]:
            if not isinstance(item, dict):
                continue
            if "key" not in item or "value" not in item:
                continue

            mem_type = item.get("memory_type", "semi_stable")
            if mem_type not in STABILITY_DEFAULTS:
                mem_type = "semi_stable"

            memories.append(
                Memory(
                    key=item["key"],
                    value=item["value"],
                    memory_type=mem_type,
                    stability_score=STABILITY_DEFAULTS.get(mem_type, 0.7),
                    confidence_source=item.get("confidence_source", "contextual_mention"),
                    source_conversation=conversation_id,
                    source_turn=turn_number,
                    raw_text=None,
                )
            )

        return memories
