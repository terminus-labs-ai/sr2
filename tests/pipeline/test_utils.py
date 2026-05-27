"""Tests for sr2.pipeline.utils shared utilities.

Covers:
  - build_subscriptions() returns correct EventSubscription list from config_subs
  - build_subscriptions() returns defaults when config_subs is None or empty
"""

import pytest

from sr2.config.models import EventSubscriptionConfig
from sr2.pipeline.events import EventPhase, EventSubscription
from sr2.pipeline.utils import build_subscriptions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_sub_config(event: str, phase: str | None = None) -> EventSubscriptionConfig:
    return EventSubscriptionConfig(event=event, phase=phase)


# ---------------------------------------------------------------------------
# build_subscriptions — custom config_subs
# ---------------------------------------------------------------------------


class TestBuildSubscriptionsCustom:
    def test_returns_list_of_event_subscriptions(self):
        """build_subscriptions always returns a list of EventSubscription."""
        phase_map = {"starting": EventPhase.STARTING}
        subs = [make_sub_config("user_input", "starting")]
        result = build_subscriptions(subs, phase_map=phase_map, defaults=[])
        assert isinstance(result, list)
        assert all(isinstance(s, EventSubscription) for s in result)

    def test_maps_event_name_from_config(self):
        """event_name on returned subscriptions matches config sub.event."""
        subs = [make_sub_config("my_event", "completed")]
        result = build_subscriptions(subs, phase_map=None, defaults=[])
        assert result[0].event_name == "my_event"

    def test_maps_phase_from_phase_map(self):
        """phase is resolved via the provided phase_map."""
        phase_map = {
            "starting": EventPhase.STARTING,
            "completed": EventPhase.COMPLETED,
            "failed": EventPhase.FAILED,
        }
        subs = [make_sub_config("turn_start", "completed")]
        result = build_subscriptions(subs, phase_map=phase_map, defaults=[])
        assert result[0].phase == EventPhase.COMPLETED

    def test_none_phase_stays_none(self):
        """When sub.phase is None, EventSubscription.phase is also None."""
        phase_map = {
            "starting": EventPhase.STARTING,
        }
        subs = [make_sub_config("some_event", None)]
        result = build_subscriptions(subs, phase_map=phase_map, defaults=[])
        assert result[0].phase is None

    def test_multiple_subs_all_converted(self):
        """All items in config_subs are converted, not just the first."""
        phase_map = {
            "starting": EventPhase.STARTING,
            "completed": EventPhase.COMPLETED,
        }
        subs = [
            make_sub_config("user_input", "starting"),
            make_sub_config("assistant_response", "completed"),
        ]
        result = build_subscriptions(subs, phase_map=phase_map, defaults=[])
        assert len(result) == 2
        assert result[0].event_name == "user_input"
        assert result[1].event_name == "assistant_response"


# ---------------------------------------------------------------------------
# build_subscriptions — defaults when config_subs is None or empty
# ---------------------------------------------------------------------------


class TestBuildSubscriptionsDefaults:
    def test_none_config_subs_returns_defaults(self):
        """When config_subs is None, the defaults list is returned."""
        default = EventSubscription(event_name="user_input", phase=EventPhase.STARTING)
        result = build_subscriptions(None, phase_map={}, defaults=[default])
        assert result == [default]

    def test_empty_config_subs_returns_defaults(self):
        """When config_subs is an empty list, the defaults list is returned."""
        default = EventSubscription(event_name="user_input", phase=EventPhase.STARTING)
        result = build_subscriptions([], phase_map={}, defaults=[default])
        assert result == [default]

    def test_defaults_are_not_mutated(self):
        """Returned defaults must be a copy, not the same object."""
        default = EventSubscription(event_name="user_input", phase=EventPhase.STARTING)
        defaults = [default]
        result = build_subscriptions(None, phase_map={}, defaults=defaults)
        assert result is not defaults

    def test_multiple_defaults_all_returned(self):
        """All defaults are returned when config_subs is absent."""
        defaults = [
            EventSubscription(event_name="user_input", phase=EventPhase.STARTING),
            EventSubscription(event_name="assistant_response", phase=EventPhase.COMPLETED),
        ]
        result = build_subscriptions(None, phase_map={}, defaults=defaults)
        assert len(result) == 2
        assert result[0].event_name == "user_input"
        assert result[1].event_name == "assistant_response"
