"""Post-LLM memory extraction from conversation turns."""

import json
import logging
import os
import re

from sr2.memory.schema import CONFIDENCE_SCORES, STABILITY_DEFAULTS, ExtractionResult, Memory
from sr2.config.models import MemoryScopeConfig
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
_JSON_VALUE = re.compile(r"^\s*[\[{]")

# Keys whose values go stale quickly and provide no lasting personal value
_TRANSIENT_KEYS = re.compile(
    r"^(forks|stars|contributors|license|sla_\w+|c\d+|"
    r"number_of_users_\w+|price_free|premium_plan_pricing)$",
    re.IGNORECASE,
)

# Keys that start with "files_to_modify" — too task-specific
_TASK_SPECIFIC_KEYS = re.compile(r"^files_to_modify", re.IGNORECASE)

# Keys containing error/failure indicators paired with operational values
_ERROR_ARTIFACT_KEYS = re.compile(r"(failure|error|fix_pattern)", re.IGNORECASE)
_ERROR_ARTIFACT_VALUES = re.compile(
    r"(task metadata|dispatcher|validation error|tool error|"
    r"task.?fail|processing error|galaxy.?map.*metadata)",
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
        scope_config: MemoryScopeConfig | None = None,
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
        scope_config: optional scope configuration for memory isolation.
        """
        self._llm = llm_callable
        self._store = store
        self._key_schema = key_schema or []
        self._max = max_memories_per_turn
        self._embed = embed_callable
        self._scope_config = scope_config
        self._key_hint_limit: int = 30  # overridden by MemoryConfig.key_hint_limit
        self._normalizer = ResponseNormalizer()

    async def extract(
        self,
        conversation_turn: str,
        conversation_id: str | None = None,
        turn_number: int | None = None,
        current_context: dict | None = None,
    ) -> ExtractionResult:
        """Extract memories from a conversation turn.

        1. Build extraction prompt (include key schema for guidance)
        2. Call LLM
        3. Parse and filter response
        4. Stamp scope on each memory
        5. Deduplicate against existing store entries
        6. Save novel memories (with embedding if available)
        7. Return ExtractionResult
        """
        logger.debug(
            "Memory extraction: turn length=%d chars, conversation_id=%s, turn=%s",
            len(conversation_turn),
            conversation_id,
            turn_number,
        )
        # Pre-fetch existing keys for hint injection (reduces synonym invention)
        existing_keys: list[str] = []
        if self._key_hint_limit > 0 and self._scope_config and self._scope_config.agent_name:
            try:
                scope_ref = f"agent:{self._scope_config.agent_name}"
                existing_keys = await self._store.list_top_keys(
                    scope_ref, limit=self._key_hint_limit,
                )
            except Exception:
                logger.debug("Failed to fetch existing keys for hints", exc_info=True)

        prompt = self._build_prompt(conversation_turn, existing_keys=existing_keys)
        raw_response = await self._llm(prompt)
        logger.debug(
            "Memory extraction: raw LLM response (%d chars): %.300s",
            len(raw_response) if raw_response else 0,
            raw_response[:300] if raw_response else "<None>",
        )
        candidates = self._parse_response(raw_response)
        logger.debug("Memory extraction: %d candidates after parsing", len(candidates))

        # Stamp scope on each candidate, filtering out any blocked by allowed_write
        candidates = [mem for mem in candidates if self._stamp_scope(mem, current_context)]

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
                    logger.error("Memory embedding failed for key=%s", mem.key, exc_info=True)
            await self._store.save(mem, embedding=embedding)
            saved.append(mem)

        source = (current_context or {}).get("source") if current_context else None
        return ExtractionResult(
            memories=saved,
            source=source,
        )

    def _stamp_scope(self, memory: Memory, current_context: dict | None) -> bool:
        """Stamp scope fields on a memory based on scope_config.

        Returns True if the memory is allowed, False if blocked by allowed_write.
        """
        if not self._scope_config:
            return True

        ctx = current_context or {}
        target_scope = self._scope_config.default_write

        if target_scope is None:
            # No write scope configured (e.g. memory disabled for this agent)
            return False

        memory.scope = target_scope

        if target_scope == "private":
            if self._scope_config.agent_name:
                memory.scope_ref = f"agent:{self._scope_config.agent_name}"
        else:
            # Non-private scopes use project_id as scope_ref when available
            project_id = ctx.get("project_id")
            if project_id:
                memory.scope_ref = project_id
            else:
                logger.debug(
                    "No project_id in context for scope '%s', key=%s — "
                    "memory will be unscoped (visible to all %s queries)",
                    target_scope,
                    memory.key,
                    target_scope,
                )

        # Enforce allowed_write boundary
        if memory.scope not in self._scope_config.allowed_write:
            logger.warning(
                "Scope '%s' not in allowed_write %s, dropping memory key=%s",
                memory.scope,
                self._scope_config.allowed_write,
                memory.key,
            )
            return False

        memory.source = ctx.get("source")
        return True

    async def _is_duplicate(self, mem: Memory) -> bool:
        """Return True if the store already has this exact key+value."""
        existing = await self._store.get_by_key(mem.key, include_archived=False)
        return any(e.value.strip().lower() == mem.value.strip().lower() for e in existing)

    @staticmethod
    def _is_task_runner_mode() -> bool:
        """Detect if running in task_runner/single-shot mode."""
        return bool(os.environ.get("SR2_TASK_SOURCE"))

    def _build_prompt(
        self, conversation_turn: str, existing_keys: list[str] | None = None,
    ) -> str:
        """Build the extraction prompt, scoped to project, task, or private context."""
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

        # Key reuse hints: pass existing keys so the LLM reuses them
        # instead of inventing synonyms
        key_hints_text = ""
        if existing_keys:
            key_hints_text = (
                "EXISTING KEYS (reuse these when the topic matches — "
                "do NOT create synonyms or near-duplicates):\n"
            )
            for k in existing_keys:
                key_hints_text += f"  - {k}\n"

        # Shared rule: never extract own execution errors
        do_not_extract_errors = """\
- Errors, failures, or issues encountered during YOUR OWN execution (tool failures, validation errors, task processing errors, file not found errors). These are transient operational issues, not durable knowledge.
- Debugging steps you took to resolve your own errors
- Task metadata or dispatcher behavior observations"""

        is_project_scope = (
            self._scope_config is not None and self._scope_config.default_write == "project"
        )
        is_private_task_runner = not is_project_scope and self._is_task_runner_mode()

        if is_project_scope:
            return f"""Extract technical findings, decisions, patterns, constraints, and reusable knowledge from this conversation.
Output ONLY a JSON array. No markdown, no explanation.
Each object: {{"key": "...", "value": "...", "memory_type": "identity|semi_stable|dynamic", "confidence_source": "explicit_statement|direct_answer|contextual_mention|inferred|offhand"}}
{schema_text}{key_hints_text}
EXTRACT: Research conclusions, API specifications, architectural decisions, implementation patterns, limitations discovered, and recommendations.
DO NOT EXTRACT:
- Raw data dumps, step-by-step reasoning, or tool call details
- Function names, search queries, or raw JSON/code
- Transient stats (GitHub stars/forks, API counts, prices that change)
- Empty, null, or placeholder values
- Information that only makes sense in the context of this specific task
- Restatements of something already obvious from context
{do_not_extract_errors}

Max {self._max} memories. If nothing durable to extract, return [].

Conversation turn:
{conversation_turn}"""
        elif is_private_task_runner:
            return f"""Extract reusable implementation patterns and technical decisions from this task.
Output ONLY a JSON array. No markdown, no explanation.
Each object: {{"key": "...", "value": "...", "memory_type": "identity|semi_stable|dynamic", "confidence_source": "explicit_statement|direct_answer|contextual_mention|inferred|offhand"}}
{schema_text}{key_hints_text}
EXTRACT: Focus on:
- Patterns that would help with FUTURE tasks (not just this one)
- Coding conventions discovered in the codebase
- Architectural patterns that affect how changes should be made
- Gotchas or non-obvious behaviors found during implementation

DO NOT EXTRACT:
- File paths, class names, or function names from this specific task
- Repository names or filesystem locations (the task already provides these)
- Generic language or framework facts (e.g., 'codebase uses Python')
- Project names or ownership information
- Anything that's only relevant to THIS task and won't help future tasks
{do_not_extract_errors}

Max {self._max} memories. If nothing durable to extract, return [].

Conversation turn:
{conversation_turn}"""
        else:
            return f"""Extract durable, personal facts from this conversation turn.
Output ONLY a JSON array. No markdown, no explanation.
Each object: {{"key": "...", "value": "...", "memory_type": "identity|semi_stable|dynamic", "confidence_source": "explicit_statement|direct_answer|contextual_mention|inferred|offhand"}}
{schema_text}{key_hints_text}
EXTRACT: Personal facts, preferences, decisions, goals, relationships, and stable context about the user.
DO NOT EXTRACT:
- Tool call details, function names, search queries, or raw JSON/code
- Transient stats (GitHub stars/forks, API counts, prices that change)
- Empty, null, or placeholder values
- Temporary task statuses that will be irrelevant next session (e.g. "task is in progress")
- Restatements of something already obvious from context
{do_not_extract_errors}

Max {self._max} memories. If nothing durable to extract, return [].

Conversation turn:
{conversation_turn}"""

    @staticmethod
    def _find_last_json_array(text: str) -> str | None:
        """Find the last balanced JSON array in text by bracket matching.

        LLMs typically place JSON output at the end of their response,
        after any commentary. Searching from the end avoids picking up
        quoted array literals (e.g. "[]") embedded in commentary text.
        """
        end = text.rfind("]")
        if end == -1:
            return None
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "]":
                depth += 1
            elif text[i] == "[":
                depth -= 1
            if depth == 0:
                return text[i : end + 1]
        return None

    def _parse_response(
        self,
        raw: str,
    ) -> list[Memory]:
        """Parse LLM JSON response into Memory objects, filtering noise."""
        if not raw:
            logger.debug("Memory extraction: empty LLM response")
            return []

        cleaned = self._normalizer.normalize(raw)
        cleaned = re.sub(r"</?json>", "", cleaned).strip()

        items = None

        # Strategy 1: direct parse (ideal case — cleaned text is valid JSON)
        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract the last JSON array in the text.
        # LLMs often emit commentary before the JSON; searching from the
        # end avoids confusing quoted "[]" in prose with the real output.
        if items is None:
            last_arr = self._find_last_json_array(cleaned)
            if last_arr:
                try:
                    items = json.loads(last_arr)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: greedy first-to-last bracket match (legacy fallback)
        if items is None:
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if items is None:
            logger.warning("Memory extraction: LLM returned unparseable response: %.200s", cleaned)
            return []

        if not isinstance(items, list):
            logger.warning("Memory extraction: expected JSON array, got %s", type(items).__name__)
            return []

        memories = []
        filtered_count = 0
        candidate_count = min(len(items), self._max)
        for item in items[: self._max]:
            if not isinstance(item, dict):
                filtered_count += 1
                continue
            key = str(item.get("key", "")).strip()
            value = str(item.get("value", "")).strip()

            # Drop empty or missing key/value
            if not key or not value:
                filtered_count += 1
                continue

            # Drop tool/system artifact keys
            if _TOOL_ARTIFACT_KEYS.match(key):
                filtered_count += 1
                continue

            # Drop keys that are always transient/low-value
            if _TRANSIENT_KEYS.match(key):
                filtered_count += 1
                continue

            # Drop task-specific keys (e.g. files_to_modify.*)
            if _TASK_SPECIFIC_KEYS.match(key):
                filtered_count += 1
                continue

            # Drop error/failure keys whose values reference operational artifacts
            if _ERROR_ARTIFACT_KEYS.search(key) and _ERROR_ARTIFACT_VALUES.search(value):
                filtered_count += 1
                continue

            # Drop values that are raw JSON blobs
            if _JSON_VALUE.match(value):
                filtered_count += 1
                continue

            # Enforce key_schema: reject keys not matching any allowed prefix
            if self._key_schema:
                allowed_prefixes = [s["prefix"] for s in self._key_schema]
                if not any(key.startswith(prefix) for prefix in allowed_prefixes):
                    logger.debug(
                        "Memory extraction: key %r rejected (no matching key_schema prefix)",
                        key,
                    )
                    filtered_count += 1
                    continue

            mem_type = item.get("memory_type", "semi_stable")
            if mem_type not in STABILITY_DEFAULTS:
                logger.warning(
                    "Memory extraction: invalid memory_type %r from LLM, defaulting to 'semi_stable'",
                    mem_type,
                )
                mem_type = "semi_stable"

            conf_source = item.get("confidence_source", "contextual_mention")
            if conf_source not in CONFIDENCE_SCORES:
                logger.warning(
                    "Memory extraction: invalid confidence_source %r from LLM, defaulting to 'contextual_mention'",
                    conf_source,
                )
                conf_source = "contextual_mention"

            memories.append(
                Memory(
                    key=key,
                    value=value,
                    memory_type=mem_type,
                    stability_score=STABILITY_DEFAULTS.get(mem_type, 0.7),
                    confidence=CONFIDENCE_SCORES[conf_source],
                    confidence_source=conf_source,
                    raw_text=None,
                )
            )

        if filtered_count:
            logger.warning(
                "Memory extraction: filtered %d/%d candidates (artifact/transient/task-specific/JSON)",
                filtered_count,
                candidate_count,
            )
        return memories
