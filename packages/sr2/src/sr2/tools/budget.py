"""Tool schema compression and token budget enforcement.

Extracted from sr2_runtime.agent so the logic is reusable outside the runtime
and testable without the full agent stack.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def estimate_schema_tokens(schema: dict) -> int:
    """Estimate token count for a tool schema using a character-based heuristic.

    Uses 1 token ≈ 4 characters on the compact JSON representation.
    Returns at least 1.
    """
    try:
        s = json.dumps(schema, separators=(",", ":"))
        return max(1, len(s) // 4)
    except Exception:
        return 100  # conservative fallback


@dataclass
class ToolBudgetResult:
    """Result of ToolSchemaBudget.enforce()."""

    schemas: list[dict]
    original_count: int
    original_tokens: int
    final_count: int
    final_tokens: int
    truncated: bool


class ToolSchemaBudget:
    """Compress and budget-enforce tool schemas before sending to an LLM.

    Two independent operations:

    - ``compress(schemas)`` — strips verbose/redundant fields that do not
      affect LLM behaviour (~20 % token savings with no functional loss).
    - ``enforce(schemas, max_tokens)`` — progressive degradation strategy:
      fit as many tools as possible within *max_tokens*, degrading each
      schema from full → no-description → minimal → dropped.
    """

    # Fields that are always removed from property definitions.
    _STRIP_ALWAYS = frozenset({"title", "examples", "minLength", "maxLength", "pattern", "default"})

    def compress(self, schemas: list[dict]) -> list[dict]:
        """Return a new list of schemas with verbose fields removed.

        Preserves:
        - Top-level ``name`` and ``description``
        - ``parameters.type``, ``parameters.required``
        - Per-property: ``type``, ``enum``
        - Per-property ``description`` *only* for required parameters

        Strips from every property:
        - ``title``, ``examples``, ``minLength``, ``maxLength``, ``pattern``, ``default``
        - ``description`` on non-required parameters
        """
        return [self._compress_one(s) for s in schemas]

    def _compress_one(self, schema: dict) -> dict:
        if not schema:
            return schema

        compressed: dict = {}

        if "name" in schema:
            compressed["name"] = schema["name"]
        if "description" in schema:
            compressed["description"] = schema["description"]

        params = schema.get("parameters")
        if params:
            compressed_params: dict = {}
            if "type" in params:
                compressed_params["type"] = params["type"]

            properties = params.get("properties", {})
            required = params.get("required", [])

            if properties:
                compressed_props: dict = {}
                for prop_name, prop_def in properties.items():
                    cp: dict = {}
                    if "type" in prop_def:
                        cp["type"] = prop_def["type"]
                    if "enum" in prop_def:
                        cp["enum"] = prop_def["enum"]
                    if prop_name in required and "description" in prop_def:
                        cp["description"] = prop_def["description"]
                    compressed_props[prop_name] = cp
                compressed_params["properties"] = compressed_props

            if required:
                compressed_params["required"] = required

            compressed["parameters"] = compressed_params

        return compressed

    def enforce(self, schemas: list[dict], max_tokens: int | None) -> ToolBudgetResult:
        """Fit schemas within *max_tokens* using progressive degradation.

        If *max_tokens* is falsy (None or 0), returns all schemas unchanged.

        Degradation order per schema:
        1. Full schema fits → include as-is.
        2. Drop top-level description → include name + parameters.
        3. Drop parameters (keep name + empty params) — only when remaining > 200.
        4. Drop the schema entirely.

        Returns a ``ToolBudgetResult`` with accounting fields.
        """
        original_count = len(schemas)
        original_tokens = sum(estimate_schema_tokens(s) for s in schemas)

        if not max_tokens or not schemas:
            return ToolBudgetResult(
                schemas=schemas,
                original_count=original_count,
                original_tokens=original_tokens,
                final_count=original_count,
                final_tokens=original_tokens,
                truncated=False,
            )

        total_tokens = original_tokens
        if total_tokens <= max_tokens:
            return ToolBudgetResult(
                schemas=schemas,
                original_count=original_count,
                original_tokens=original_tokens,
                final_count=original_count,
                final_tokens=total_tokens,
                truncated=False,
            )

        logger.info(
            "Tool schemas exceed max_tokens budget: %d > %d, truncating",
            total_tokens,
            max_tokens,
        )

        result: list[dict] = []
        remaining = max_tokens

        for schema in schemas:
            schema_tokens = estimate_schema_tokens(schema)

            if schema_tokens <= remaining:
                result.append(schema)
                remaining -= schema_tokens
                continue

            # Level 2: drop top-level description
            no_desc = {k: v for k, v in schema.items() if k != "description"}
            no_desc_tokens = estimate_schema_tokens(no_desc)

            if no_desc_tokens <= remaining:
                result.append(no_desc)
                remaining -= no_desc_tokens
                logger.debug("  Dropped description for %s", schema.get("name"))
                continue

            # Level 3: minimal schema — name + empty params (only when headroom > 200)
            if remaining > 200:
                minimal: dict = {"name": schema.get("name")}
                params = schema.get("parameters", {})
                minimal["parameters"] = {
                    "type": params.get("type", "object"),
                    "properties": {},
                }
                minimal_tokens = estimate_schema_tokens(minimal)
                if minimal_tokens <= remaining:
                    result.append(minimal)
                    remaining -= minimal_tokens
                    logger.debug("  Dropped parameters for %s", schema.get("name"))
                    continue

            # Level 4: drop entirely
            logger.debug("  Dropped schema for %s", schema.get("name"))

        if len(result) < original_count:
            logger.warning(
                "Tool schema truncation: %d/%d tools fit in %d tokens",
                len(result),
                original_count,
                max_tokens,
            )

        final_tokens = sum(estimate_schema_tokens(s) for s in result)

        return ToolBudgetResult(
            schemas=result,
            original_count=original_count,
            original_tokens=original_tokens,
            final_count=len(result),
            final_tokens=final_tokens,
            truncated=True,
        )
