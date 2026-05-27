"""Shared pipeline utilities.

Provides helpers used across resolvers and transformers to avoid duplication.
"""

from __future__ import annotations

from sr2.config.models import EventSubscriptionConfig
from sr2.pipeline.events import EventPhase, EventSubscription

# Canonical phase map shared by all resolvers and transformers.
PHASE_MAP: dict[str, EventPhase] = {
    "starting": EventPhase.STARTING,
    "completed": EventPhase.COMPLETED,
    "failed": EventPhase.FAILED,
}


def build_subscriptions(
    config_subs: list[EventSubscriptionConfig] | None,
    phase_map: dict[str, EventPhase] | None,
    defaults: list[EventSubscription],
) -> list[EventSubscription]:
    """Build a list of EventSubscription from config or fall back to defaults.

    Args:
        config_subs: Subscriptions from ResolverConfig/TransformerConfig.
                     If None or empty, defaults are returned.
        phase_map:   Mapping from phase string to EventPhase enum.
                     If None, PHASE_MAP is used.
        defaults:    Subscriptions to return when config_subs is absent.
                     A copy is returned to prevent mutation of caller's list.

    Returns:
        List of EventSubscription instances.
    """
    if not config_subs:
        return list(defaults)

    _map = phase_map if phase_map is not None else PHASE_MAP

    return [
        EventSubscription(
            event_name=sub.event,
            phase=_map[sub.phase] if sub.phase is not None else None,
        )
        for sub in config_subs
    ]


def extract_user_input_text(events: list) -> str:
    """Extract text from user_input events.

    Walks events looking for 'user_input'; extracts text from ContentBlock
    lists or returns data directly if it is already a string.

    Args:
        events: List of Event objects.

    Returns:
        Concatenated text from the first matching user_input event,
        or an empty string if none found.
    """
    for event in events:
        if event.name == "user_input":
            data = event.data
            if isinstance(data, list):
                texts = [block.text for block in data if hasattr(block, "text")]
                return " ".join(texts)
            elif isinstance(data, str):
                return data
    return ""
