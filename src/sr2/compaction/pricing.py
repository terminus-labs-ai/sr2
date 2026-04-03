"""Pricing resolution for cache cost calculations.

Resolves per-token pricing from LiteLLM's model_cost map with custom overrides
and fail-open fallback.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CachePricing:
    """Per-token pricing for cache cost calculations."""

    input_cost: float  # uncached input, per token
    cache_write_cost: float  # cache creation, per token
    cache_read_cost: float  # cache read, per token
    source: str  # "custom", "litellm:<model_name>", or "fail_open"


def _resolve_from_litellm(model_name: str) -> CachePricing | None:
    """Attempt to resolve pricing from LiteLLM's model_cost map."""
    try:
        from litellm import model_cost
    except ImportError:
        logger.warning("LiteLLM not available for pricing lookup")
        return None

    info = model_cost.get(model_name)
    if info is None:
        return None

    input_cost = info.get("input_cost_per_token", 0.0)
    cache_write = info.get("cache_creation_input_token_cost", 0.0)
    cache_read = info.get("cache_read_input_token_cost", 0.0)

    # Warn if model claims caching support but has zero/missing cache costs
    if info.get("supports_prompt_caching", False):
        if not cache_write:
            logger.warning(
                "Model '%s' supports caching but cache_creation_input_token_cost is %s "
                "in LiteLLM — pricing may be wrong. Consider setting custom_pricing.",
                model_name,
                cache_write,
            )
        if not cache_read:
            logger.warning(
                "Model '%s' supports caching but cache_read_input_token_cost is %s "
                "in LiteLLM — pricing may be wrong. Consider setting custom_pricing.",
                model_name,
                cache_read,
            )

    return CachePricing(
        input_cost=input_cost,
        cache_write_cost=cache_write,
        cache_read_cost=cache_read,
        source=f"litellm:{model_name}",
    )


def _resolve_from_custom(custom_pricing: dict[str, float]) -> CachePricing:
    """Resolve pricing from custom config (values in $/MTok, converted to per-token)."""
    return CachePricing(
        input_cost=custom_pricing.get("input", 0.0) / 1_000_000,
        cache_write_cost=custom_pricing.get("cache_write", 0.0) / 1_000_000,
        cache_read_cost=custom_pricing.get("cache_read", 0.0) / 1_000_000,
        source="custom",
    )


_FAIL_OPEN = CachePricing(
    input_cost=0.0,
    cache_write_cost=0.0,
    cache_read_cost=0.0,
    source="fail_open",
)


def resolve_pricing(
    model_hint: str | None = None,
    fallback_model: str | None = None,
    custom_pricing: dict[str, float] | None = None,
) -> CachePricing:
    """Resolve cache pricing using the fallback chain.

    1. custom_pricing dict (if provided)
    2. litellm.model_cost[model_hint]
    3. litellm.model_cost[fallback_model]
    4. fail-open (all zeros, source="fail_open")
    """
    # 1. Custom pricing takes priority
    if custom_pricing is not None:
        return _resolve_from_custom(custom_pricing)

    # 2. LiteLLM lookup by model_hint
    if model_hint is not None:
        result = _resolve_from_litellm(model_hint)
        if result is not None:
            return result

    # 3. Fallback model
    if fallback_model is not None:
        result = _resolve_from_litellm(fallback_model)
        if result is not None:
            return result

    # 4. Fail open
    logger.warning(
        "No pricing data for model_hint=%r, fallback_model=%r — failing open (allowing compaction)",
        model_hint,
        fallback_model,
    )
    return _FAIL_OPEN
