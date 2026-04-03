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


class TestRegistryGetUnknown:
    """Registry get() on unknown name raises KeyError."""

    def test_unknown_name_raises_key_error(self):
        registry = CachePolicyRegistry()
        with pytest.raises(KeyError, match="No cache policy registered"):
            registry.get("nonexistent")
