"""Post-LLM memory extraction from conversation turns."""

import json
import logging
import re

from sr2.memory.schema import CONFIDENCE_SCORES, STABILITY_DEFAULTS, ExtractionResult, Memory
from sr2.memory.store import MemoryStore
from sr2.normalization.normalizer import ResponseNormalizer

logger = logging.getLogger(__name__)

# Patterns that indicate tool/system artifact content — not worth memorising
_TOOL_ARTIFACT_KEYS = re.compile(
    r"^(tool_call|function name|search query|path_\d+|tool calls?|"
    r"required argument|optional argument|language format|time range format|"
    r"entities_to_|search_terms|search_objective|user_request_goal)$",
    re.IGNORECASE,
)

# Values that look like raw JSON blobs (tool call dumps)
_JSON_VALUE = re.compile(r'^\s*[\[{]')

# Keys whose values go stale quickly and provide no lasting personal value
_TRANSIENT_KEYS = re.compile(
    r"^(forks|stars|contributors|license|sla_\w+|c\d+|"
    r"number_of_users_\w+|price_free|premium_plan_pricing)$",
    re.IGNORECASE,
)


class MemoryExtractor:
    """Extracts structured memories from conversation turns using an LLM."""

    def __init__(
        self,
        llm_callable,
        store: MemoryStore,
        key_schema: list[dict] | None = None,
        max_memories_per_turn: int = 5,
        embed_callable=None,
    ):
        """Args:
        llm_callable: async function(prompt: str) -> str.
                      Wraps LiteLLM or any LLM call.
        store: MemoryStore to save extracted memories.
        key_schema: list of {"prefix": str, "examples": list[str]}
        max_memories_per_turn: max memories to extract per turn.
        embed_callable: optional async function(text: str) -> list[float].
                        When provided, embeddings are generated and stored
                        with each memory to enable semantic search.
        """
        self._llm = llm_callable
        self._store = store
        self._key_schema = key_schema or []
        self._max = max_memories_per_turn
        self._embed = embed_callable
        self._normalizer = ResponseNormalizer()

    async def extract(
        self,
        conversation_turn: str,
        conversation_id: str | None = None,
        turn_number: int | None = None,
    ) -> ExtractionResult:
        """Extract memories from a conversation turn.

        1. Build extraction prompt (include key schema for guidance)
        2. Call LLM
        3. Parse and filter response
        4. Deduplicate against existing store entries
        5. Save novel memories (with embedding if available)
        6. Return ExtractionResult
        """
        prompt = self._build_prompt(conversation_turn)
        raw_response = await self._llm(prompt)
        candidates = self._parse_response(raw_response, conversation_id, turn_number)

        saved: list[Memory] = []
        for mem in candidates:
            # Skip if an identical key+value already exists
            if await self._is_duplicate(mem):
                continue
            embedding = None
            if self._embed:
                try:
                    embedding = await self._embed(f"{mem.key}: {mem.value}")
                except Exception:
                    pass  # save without embedding if generation fails
            await self._store.save(mem, embedding=embedding)
            saved.append(mem)

        return ExtractionResult(
            memories=saved,
            source_conversation=conversation_id,
            source_turn=turn_number,
        )

    async def _is_duplicate(self, mem: Memory) -> bool:
        """Return True if the store already has this exact key+value."""
        existing = await self._store.get_by_key(mem.key, include_archived=False)
        return any(e.value.strip().lower() == mem.value.strip().lower() for e in existing)

    def _build_prompt(self, conversation_turn: str) -> str:
        """Build the extraction prompt."""
        schema_text = ""
        if self._key_schema:
            schema_text = "Keys MUST use one of these dot-notation prefixes:\n"
            for s in self._key_schema:
                entry = f"  - {s['prefix']}"
                if s.get("description"):
                    entry += f": {s['description']}"
                if s.get("examples"):
                    entry += f" (e.g. {', '.join(s['examples'])})"
                schema_text += entry + "\n"
            schema_text += (
                "Format: <prefix>.<specific_attribute> in lowercase with dots, no spaces.\n"
            )

        return f"""Extract durable, personal facts from this conversation turn.
Output ONLY a JSON array. No markdown, no explanation.
Each object: {{"key": "...", "value": "...", "memory_type": "identity|semi_stable|dynamic", "confidence_source": "explicit_statement|direct_answer|contextual_mention|inferred|offhand"}}
{schema_text}
EXTRACT: Personal facts, preferences, decisions, goals, relationships, and stable context about the user.
DO NOT EXTRACT:
- Tool call details, function names, search queries, or raw JSON/code
- Transient stats (GitHub stars/forks, API counts, prices that change)
- Empty, null, or placeholder values
- Temporary task statuses that will be irrelevant next session (e.g. "task is in progress")
- Restatements of something already obvious from context

Max {self._max} memories. If nothing durable to extract, return [].

Conversation turn:
{conversation_turn}"""

    def _parse_response(
        self,
        raw: str,
        conversation_id: str | None,
        turn_number: int | None,
    ) -> list[Memory]:
        """Parse LLM JSON response into Memory objects, filtering noise."""
        cleaned = self._normalizer.normalize(raw)

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            # LLM may have wrapped the array in prose; try to extract it
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Memory extraction failed: LLM returned invalid JSON: %s", cleaned[:200])
                    return []
            else:
                logger.warning("Memory extraction failed: LLM returned invalid JSON: %s", cleaned[:200])
                return []

        if not isinstance(items, list):
            logger.warning("Memory extraction failed: expected JSON array, got %s", type(items).__name__)
            return []

        memories = []
        for item in items[: self._max]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()

            # Drop empty or missing key/value
            if not key or not value:
                continue

            # Drop tool/system artifact keys
            if _TOOL_ARTIFACT_KEYS.match(key):
                continue

            # Drop keys that are always transient/low-value
            if _TRANSIENT_KEYS.match(key):
                continue

            # Drop values that are raw JSON blobs
            if _JSON_VALUE.match(value):
                continue

            mem_type = item.get("memory_type", "semi_stable")
            if mem_type not in STABILITY_DEFAULTS:
                mem_type = "semi_stable"

            conf_source = item.get("confidence_source", "contextual_mention")
            if conf_source not in CONFIDENCE_SCORES:
                conf_source = "contextual_mention"

            memories.append(
                Memory(
                    key=key,
                    value=value,
                    memory_type=mem_type,
                    stability_score=STABILITY_DEFAULTS.get(mem_type, 0.7),
                    confidence=CONFIDENCE_SCORES[conf_source],
                    confidence_source=conf_source,
                    source_conversation=conversation_id,
                    source_turn=turn_number,
                    raw_text=None,
                )
            )

        return memories
