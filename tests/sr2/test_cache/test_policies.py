import pytest

from sr2.cache.registry import CachePolicyRegistry, PipelineState
from sr2.cache.policies import (
    ImmutablePolicy,
    RefreshOnTopicShiftPolicy,
    RefreshOnStateChangePolicy,
    AppendOnlyPolicy,
    AlwaysNewPolicy,
    PerInvocationPolicy,
    TemplateReusePolicy,
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


class TestAlwaysNewPolicy:
    """AlwaysNewPolicy: always returns True regardless of state."""

    def test_first_call(self):
        policy = AlwaysNewPolicy()
        assert policy.should_recompute("layer", PipelineState(), None) is True

    def test_with_previous(self):
        policy = AlwaysNewPolicy()
        assert policy.should_recompute("layer", PipelineState(), PipelineState()) is True


class TestPerInvocationPolicy:
    """PerInvocationPolicy: always returns True regardless of state."""

    def test_first_call(self):
        policy = PerInvocationPolicy()
        assert policy.should_recompute("layer", PipelineState(), None) is True

    def test_with_previous(self):
        policy = PerInvocationPolicy()
        assert policy.should_recompute("layer", PipelineState(), PipelineState()) is True


class TestTemplateReusePolicy:
    """TemplateReusePolicy: returns True on first call, False after."""

    def test_first_call_returns_true(self):
        policy = TemplateReusePolicy()
        assert policy.should_recompute("tmpl", PipelineState(), None) is True

    def test_subsequent_call_returns_false(self):
        policy = TemplateReusePolicy()
        assert policy.should_recompute("tmpl", PipelineState(), PipelineState()) is False


class TestMultiTurnPipelineSimulation:
    """Simulate a realistic multi-turn pipeline with multiple layers using different policies.

    Models a KV-cache-style scenario: stable layers (system prompt, template)
    should remain cached while volatile layers (conversation, intent-dependent
    memory) recompute as state evolves.
    """

    LAYERS = {
        "system_prompt": ImmutablePolicy(),       # never changes after first
        "template": TemplateReusePolicy(),        # never changes after first
        "memory": RefreshOnTopicShiftPolicy(),    # recomputes on topic shift
        "state": RefreshOnStateChangePolicy(),    # recomputes on state change
        "conversation": AppendOnlyPolicy(),       # always recomputes
        "response": AlwaysNewPolicy(),            # always recomputes
    }

    def _simulate_turn(self, current, previous):
        """Return dict of layer_name -> should_recompute for one turn."""
        return {
            name: policy.should_recompute(name, current, previous)
            for name, policy in self.LAYERS.items()
        }

    def test_first_turn_all_layers_compute(self):
        """On the very first turn every layer must compute."""
        state = PipelineState(turn_number=0, current_intent="greet", state_hash="s0")
        results = self._simulate_turn(state, None)
        assert all(results.values())

    def test_stable_turn_only_volatile_layers_recompute(self):
        """When intent and state are unchanged, only always-recompute layers fire."""
        prev = PipelineState(turn_number=0, current_intent="greet", state_hash="s0")
        curr = PipelineState(turn_number=1, current_intent="greet", state_hash="s0")
        results = self._simulate_turn(curr, prev)

        # Stable layers stay cached
        assert results["system_prompt"] is False
        assert results["template"] is False
        assert results["memory"] is False
        assert results["state"] is False
        # Volatile layers always recompute
        assert results["conversation"] is True
        assert results["response"] is True

    def test_topic_shift_triggers_memory_only(self):
        """A topic shift causes memory to recompute but not system_prompt/template/state."""
        prev = PipelineState(turn_number=1, current_intent="greet", state_hash="s0")
        curr = PipelineState(turn_number=2, current_intent="weather", state_hash="s0")
        results = self._simulate_turn(curr, prev)

        assert results["system_prompt"] is False
        assert results["template"] is False
        assert results["memory"] is True   # topic shifted
        assert results["state"] is False   # hash unchanged
        assert results["conversation"] is True
        assert results["response"] is True

    def test_state_change_triggers_state_only(self):
        """A state hash change causes state layer to recompute but not memory."""
        prev = PipelineState(turn_number=2, current_intent="weather", state_hash="s0")
        curr = PipelineState(turn_number=3, current_intent="weather", state_hash="s1")
        results = self._simulate_turn(curr, prev)

        assert results["system_prompt"] is False
        assert results["template"] is False
        assert results["memory"] is False  # same intent
        assert results["state"] is True    # hash changed
        assert results["conversation"] is True
        assert results["response"] is True

    def test_both_topic_and_state_change(self):
        """When both intent and state change, both conditional layers recompute."""
        prev = PipelineState(turn_number=3, current_intent="weather", state_hash="s1")
        curr = PipelineState(turn_number=4, current_intent="coding", state_hash="s2")
        results = self._simulate_turn(curr, prev)

        assert results["system_prompt"] is False
        assert results["template"] is False
        assert results["memory"] is True
        assert results["state"] is True
        assert results["conversation"] is True
        assert results["response"] is True

    def test_five_turn_sequence_immutable_never_recomputes_after_first(self):
        """Immutable and template_reuse layers stay False across an entire 5-turn conversation."""
        immutable = ImmutablePolicy()
        template = TemplateReusePolicy()

        states = [
            PipelineState(turn_number=i, current_intent=f"t{i}", state_hash=f"h{i}")
            for i in range(5)
        ]

        # Turn 0: both compute
        assert immutable.should_recompute("sys", states[0], None) is True
        assert template.should_recompute("tmpl", states[0], None) is True

        # Turns 1-4: never recompute
        for i in range(1, 5):
            assert immutable.should_recompute("sys", states[i], states[i - 1]) is False
            assert template.should_recompute("tmpl", states[i], states[i - 1]) is False

    def test_always_recompute_policies_across_ten_turns(self):
        """AlwaysNew, PerInvocation, and AppendOnly always return True."""
        policies = [AlwaysNewPolicy(), PerInvocationPolicy(), AppendOnlyPolicy()]
        states = [PipelineState(turn_number=i) for i in range(10)]

        for policy in policies:
            # First turn
            assert policy.should_recompute("layer", states[0], None) is True
            # Subsequent turns
            for i in range(1, 10):
                assert policy.should_recompute("layer", states[i], states[i - 1]) is True


class TestRapidAlternation:
    """Edge cases with rapidly alternating state."""

    def test_intent_flip_flop(self):
        """Rapidly alternating intents trigger recompute every turn."""
        policy = RefreshOnTopicShiftPolicy()
        intents = ["a", "b", "a", "b", "a", "b"]
        states = [PipelineState(current_intent=i) for i in intents]

        for i in range(1, len(states)):
            assert policy.should_recompute("mem", states[i], states[i - 1]) is True

    def test_state_hash_flip_flop(self):
        """Rapidly alternating state hashes trigger recompute every turn."""
        policy = RefreshOnStateChangePolicy()
        hashes = ["x", "y", "x", "y", "x"]
        states = [PipelineState(state_hash=h) for h in hashes]

        for i in range(1, len(states)):
            assert policy.should_recompute("st", states[i], states[i - 1]) is True

    def test_occasional_stability_window(self):
        """Mixed sequence: alternation then stability then alternation."""
        policy = RefreshOnTopicShiftPolicy()
        intents = ["a", "b", "c", "c", "c", "d", "e"]
        states = [PipelineState(current_intent=i) for i in intents]
        expected = [True, True, False, False, True, True]

        for i in range(1, len(states)):
            assert policy.should_recompute("mem", states[i], states[i - 1]) is expected[i - 1]


class TestRegistryMultiTurnOrchestration:
    """Use the default registry to orchestrate a multi-turn cache decision sequence."""

    def test_registry_round_trip_all_policies(self):
        """Every registered policy is callable through the registry interface."""
        registry = create_default_cache_registry()
        state = PipelineState(current_intent="test", state_hash="h")

        for name in registry.registered_policies:
            policy = registry.get(name)
            # First call always True for all policies
            result = policy.should_recompute(name, state, None)
            assert result is True

    def test_registry_second_turn_conditional_policies(self):
        """On second turn with identical state, conditional policies return False."""
        registry = create_default_cache_registry()
        state = PipelineState(current_intent="test", state_hash="h")

        always_true = {"append_only", "always_new", "per_invocation"}

        for name in registry.registered_policies:
            policy = registry.get(name)
            result = policy.should_recompute(name, state, state)
            if name in always_true:
                assert result is True, f"{name} should always recompute"
            else:
                assert result is False, f"{name} should not recompute with identical state"


class TestRegistryGetUnknown:
    """Registry get() on unknown name raises KeyError."""

    def test_unknown_name_raises_key_error(self):
        registry = CachePolicyRegistry()
        with pytest.raises(KeyError, match="No cache policy registered"):
            registry.get("nonexistent")
