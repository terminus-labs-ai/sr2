"""Automatic scope detection for memory system.

Determines the correct scope_ref for a session by analyzing context signals
(system prompt, user message) against known scope_refs in the memory store.
Private scopes are resolved deterministically; non-private scopes use an LLM
to match against existing scope_refs.
"""

import json
import logging

from sr2.config.models import MemoryScopeConfig
from sr2.memory.store import MemoryStore

logger = logging.getLogger(__name__)


class ScopeDetector:
    """Detects the correct scope_ref for a session based on context signals."""

    def __init__(
        self,
        store: MemoryStore,
        llm_callable,  # async (prompt: str) -> str
        scope_config: MemoryScopeConfig,
    ):
        self._store = store
        self._llm = llm_callable
        self._scope_config = scope_config
        self._cache: dict[str, dict[str, str | None]] = {}

    async def detect(
        self,
        system_prompt: str | None,
        user_message: str | None,
        session_id: str | None = None,
    ) -> dict[str, str | None]:
        """Return a dict of {scope: scope_ref} for each scope in allowed_read.

        Results are cached per session_id. Returns empty dict if scope_config
        has no non-private scopes or store has no existing scope_refs.
        Private scope refs are determined deterministically (agent_name),
        not via LLM -- they are included in the result automatically.
        """
        # Build deterministic result (private scope)
        deterministic: dict[str, str | None] = {}
        if "private" in self._scope_config.allowed_read and self._scope_config.agent_name:
            deterministic["private"] = f"agent:{self._scope_config.agent_name}"

        # Identify non-private scopes needing detection
        non_private_scopes = [
            s for s in self._scope_config.allowed_read if s != "private"
        ]

        # Short-circuit: no non-private scopes
        if not non_private_scopes:
            return deterministic

        # Check cache
        if session_id is not None and session_id in self._cache:
            return self._cache[session_id]

        # No LLM callable: return deterministic only
        if self._llm is None:
            return deterministic

        # Query store for known scope_refs (non-private only)
        try:
            known_pairs = await self._store.list_scope_refs(
                scope_filter=non_private_scopes,
            )
        except Exception:
            logger.error("Failed to query scope_refs from store", exc_info=True)
            return deterministic

        # Group by scope
        scope_refs_by_scope: dict[str, list[str]] = {}
        for scope, scope_ref in known_pairs:
            if scope_ref is not None:
                scope_refs_by_scope.setdefault(scope, []).append(scope_ref)

        # No existing non-private scope_refs: nothing to match against
        if not scope_refs_by_scope:
            return deterministic

        # Build LLM prompt
        prompt = self._build_prompt(system_prompt, user_message, scope_refs_by_scope)

        # Call LLM
        try:
            raw_response = await self._llm(prompt)
        except Exception:
            logger.warning("Scope detection LLM call failed", exc_info=True)
            return deterministic

        # Parse response
        detected = self._parse_response(raw_response, non_private_scopes)

        # Merge deterministic results
        result = {**deterministic, **detected}

        # Cache
        if session_id is not None:
            self._cache[session_id] = result

        return result

    def invalidate(self, session_id: str) -> None:
        """Clear cached detection for a session (e.g., on topic shift)."""
        self._cache.pop(session_id, None)

    @staticmethod
    def _build_prompt(
        system_prompt: str | None,
        user_message: str | None,
        scope_refs_by_scope: dict[str, list[str]],
    ) -> str:
        """Build the LLM prompt for scope detection."""
        parts: list[str] = [
            "Given the following context, determine which project/scope "
            "this conversation belongs to."
        ]

        if system_prompt:
            excerpt = system_prompt[:500]
            parts.append(f"\nSystem context (truncated):\n{excerpt}")

        if user_message:
            excerpt = user_message[:500]
            parts.append(f"\nUser message (truncated):\n{excerpt}")

        parts.append("\nKnown scope references:")
        for scope, refs in scope_refs_by_scope.items():
            parts.append(f'  Scope "{scope}":')
            for ref in refs:
                parts.append(f"    - {ref}")

        parts.append(
            "\nFor each scope listed above, return the scope_ref that best "
            "matches this conversation's context, or null if no match is clear."
        )
        parts.append("\nOutput ONLY a JSON object. No markdown, no explanation.")

        # Build example from actual scope names
        example_keys = list(scope_refs_by_scope.keys())
        if example_keys:
            example_parts = [f'"{k}": null' for k in example_keys]
            example = "{" + ", ".join(example_parts) + "}"
            parts.append(f"Example: {example}")

        return "\n".join(parts)

    @staticmethod
    def _find_last_json_object(text: str) -> str | None:
        """Find the last balanced JSON object in the text."""
        if not text:
            return None
        end = text.rfind("}")
        if end == -1:
            return None
        depth = 0
        for i in range(end, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
            if depth == 0:
                return text[i : end + 1]
        return None

    @classmethod
    def _parse_response(
        cls,
        raw_response: str | None,
        expected_scopes: list[str],
    ) -> dict[str, str | None]:
        """Parse the LLM response into a scope -> scope_ref mapping."""
        if not raw_response:
            return {}

        json_str = cls._find_last_json_object(raw_response)
        if not json_str:
            logger.warning("Scope detection: no JSON object found in LLM response")
            return {}

        try:
            parsed = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Scope detection: failed to parse JSON from LLM response")
            return {}

        if not isinstance(parsed, dict):
            logger.warning("Scope detection: LLM response is not a JSON object")
            return {}

        result: dict[str, str | None] = {}
        for key, value in parsed.items():
            if key not in expected_scopes:
                continue
            if isinstance(value, str) and value:
                result[key] = value
            else:
                result[key] = None

        return result
