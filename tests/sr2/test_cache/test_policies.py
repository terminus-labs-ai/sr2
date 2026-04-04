import pytest

from sr2.cache.registry import CachePolicyRegistry, PipelineState
from sr2.cache.policies import (
    ImmutablePolicy,
    RefreshOnTopicShiftPolicy,
    RefreshOnStateChangePolicy,
    AppendOnlyPolicy,
    create_default_cache_registry,
)


class TestImmutablePolicy:
    """ImmutablePolicy: returns True on first call (previous=None), False after."""

    def test_first_call_returns_true(self):
        policy = ImmutablePolicy()
        current = PipelineState()
        assert policy.should_recompute("core", current, None) is True

    def test_subsequent_call_returns_false(self):
        policy = ImmutablePolicy()
        current = PipelineState()
        previous = PipelineState()
        assert policy.should_recompute("core", current, previous) is False


class TestRefreshOnTopicShiftPolicy:
    """RefreshOnTopicShiftPolicy: returns True when intent changes, False when same."""

    def test_first_call_returns_true(self):
        policy = RefreshOnTopicShiftPolicy()
        current = PipelineState(current_intent="greet")
        assert policy.should_recompute("memory", current, None) is True

    def test_same_intent_returns_false(self):
        policy = RefreshOnTopicShiftPolicy()
        current = PipelineState(current_intent="greet")
        previous = PipelineState(current_intent="greet")
        assert policy.should_recompute("memory", current, previous) is False

    def test_different_intent_returns_true(self):
        policy = RefreshOnTopicShiftPolicy()
        current = PipelineState(current_intent="farewell")
        previous = PipelineState(current_intent="greet")
        assert policy.should_recompute("memory", current, previous) is True


class TestRefreshOnStateChangePolicy:
    """RefreshOnStateChangePolicy: returns True when state_hash changes."""

    def test_first_call_returns_true(self):
        policy = RefreshOnStateChangePolicy()
        current = PipelineState(state_hash="abc123")
        assert policy.should_recompute("state", current, None) is True

    def test_same_hash_returns_false(self):
        policy = RefreshOnStateChangePolicy()
        current = PipelineState(state_hash="abc123")
        previous = PipelineState(state_hash="abc123")
        assert policy.should_recompute("state", current, previous) is False

    def test_different_hash_returns_true(self):
        policy = RefreshOnStateChangePolicy()
        current = PipelineState(state_hash="def456")
        previous = PipelineState(state_hash="abc123")
        assert policy.should_recompute("state", current, previous) is True


class TestAppendOnlyPolicy:
    """AppendOnlyPolicy: always returns True."""

    def test_always_returns_true_no_previous(self):
        policy = AppendOnlyPolicy()
        current = PipelineState()
        assert policy.should_recompute("history", current, None) is True

    def test_always_returns_true_with_previous(self):
        policy = AppendOnlyPolicy()
        current = PipelineState()
        previous = PipelineState()
        assert policy.should_recompute("history", current, previous) is True


class TestCreateDefaultCacheRegistry:
    """create_default_cache_registry() has all 7 policies registered."""

    def test_all_seven_policies_registered(self):
        registry = create_default_cache_registry()
        expected = [
            "immutable",
            "refresh_on_topic_shift",
            "refresh_on_state_change",
            "append_only",
            "always_new",
            "per_invocation",
            "template_reuse",
        ]
        assert sorted(registry.registered_policies) == sorted(expected)

    def test_each_policy_is_retrievable(self):
        registry = create_default_cache_registry()
        for name in registry.registered_policies:
            policy = registry.get(name)
            assert policy is not None


class TestRefreshOnStateChangeMultiTurn:
    """Multi-turn state transitions for RefreshOnStateChangePolicy."""

    def test_state_cycle_a_b_a(self):
        """State returns to original hash — policy correctly detects the return as no-change.

        Sequence: A (first) -> B (changed) -> A (returned).
        The policy compares current vs. previous only (not full history),
        so A->A when previous was B should trigger recompute.
        """
        policy = RefreshOnStateChangePolicy()

        state_a = PipelineState(state_hash="hash_a")
        state_b = PipelineState(state_hash="hash_b")
        state_a2 = PipelineState(state_hash="hash_a")

        # Turn 1: first call (no previous)
        assert policy.should_recompute("state", state_a, None) is True

        # Turn 2: A -> B (hash changed)
        assert policy.should_recompute("state", state_b, state_a) is True

        # Turn 3: B -> A (hash changed back — still a change relative to previous)
        assert policy.should_recompute("state", state_a2, state_b) is True

    def test_repeated_same_state_no_recompute(self):
        """Staying in the same state across multiple turns never triggers recompute."""
        policy = RefreshOnStateChangePolicy()

        state = PipelineState(state_hash="stable")
        prev = PipelineState(state_hash="stable")

        for _ in range(5):
            assert policy.should_recompute("state", state, prev) is False


class TestRefreshOnTopicShiftMultiTurn:
    """Multi-turn topic transitions for RefreshOnTopicShiftPolicy."""

    def test_topic_return_triggers_recompute(self):
        """Returning to a previous topic after a detour triggers recompute.

        Sequence: greet -> weather -> greet.
        Policy compares current intent to previous intent only, so
        returning to 'greet' from 'weather' is a change.
        """
        policy = RefreshOnTopicShiftPolicy()

        greet = PipelineState(current_intent="greet")
        weather = PipelineState(current_intent="weather")
        greet2 = PipelineState(current_intent="greet")

        # Turn 1: first call
        assert policy.should_recompute("memory", greet, None) is True

        # Turn 2: greet -> weather
        assert policy.should_recompute("memory", weather, greet) is True

        # Turn 3: weather -> greet (return to previous topic)
        assert policy.should_recompute("memory", greet2, weather) is True

    def test_same_topic_sustained_no_recompute(self):
        """Staying on the same topic across turns never triggers recompute."""
        policy = RefreshOnTopicShiftPolicy()

        prev = PipelineState(current_intent="coding")
        current = PipelineState(current_intent="coding")

        for _ in range(5):
            assert policy.should_recompute("memory", current, prev) is False

    def test_none_intent_transition(self):
        """Transition from a set intent to None counts as a topic shift."""
        policy = RefreshOnTopicShiftPolicy()

        prev = PipelineState(current_intent="greet")
        current = PipelineState(current_intent=None)

        assert policy.should_recompute("memory", current, prev) is True

    def test_none_to_none_no_recompute(self):
        """Both intents None — no topic shift."""
        policy = RefreshOnTopicShiftPolicy()

        prev = PipelineState(current_intent=None)
        current = PipelineState(current_intent=None)

        assert policy.should_recompute("memory", current, prev) is False


class TestRegistryGetUnknown:
    """Registry get() on unknown name raises KeyError."""

    def test_unknown_name_raises_key_error(self):
        registry = CachePolicyRegistry()
        with pytest.raises(KeyError, match="No cache policy registered"):
            registry.get("nonexistent")
