"""Tests for tool state machine."""

from sr2.tools.models import (
    ToolDefinition,
    ToolManagementConfig,
    ToolStateConfig,
    ToolTransitionConfig,
)
from sr2.tools.state_machine import ToolStateMachine


def _make_config() -> ToolManagementConfig:
    return ToolManagementConfig(
        tools=[
            ToolDefinition(name="read_file"),
            ToolDefinition(name="write_file"),
            ToolDefinition(name="bash"),
            ToolDefinition(name="rm"),
        ],
        states=[
            ToolStateConfig(name="default", allowed_tools="all"),
            ToolStateConfig(name="planning", allowed_tools=["read_file"]),
            ToolStateConfig(name="executing", allowed_tools="all", denied_tools=["rm"]),
        ],
        transitions=[
            ToolTransitionConfig(
                from_state="default", to_state="planning",
                trigger="agent_intent", condition="intent == 'planning'",
            ),
            ToolTransitionConfig(
                from_state="planning", to_state="executing",
                trigger="agent_intent", condition="intent == 'executing'",
            ),
            ToolTransitionConfig(
                from_state="any", to_state="default",
                trigger="pipeline_signal",
            ),
        ],
        masking_strategy="allowed_list",
        initial_state="default",
    )


class TestToolStateMachine:

    def test_initial_state(self):
        """Initial state matches config.initial_state."""
        sm = ToolStateMachine(_make_config())
        assert sm.current_state_name == "default"

    def test_allowed_tools_respects_state(self):
        """get_allowed_tools() respects current state."""
        sm = ToolStateMachine(_make_config())
        sm.try_transition("agent_intent", {"intent": "planning"})
        assert sm.get_allowed_tools() == ["read_file"]

    def test_transition_matching_trigger_and_condition(self):
        """try_transition() with matching trigger and condition transitions."""
        sm = ToolStateMachine(_make_config())
        result = sm.try_transition("agent_intent", {"intent": "planning"})
        assert result is True
        assert sm.current_state_name == "planning"

    def test_transition_non_matching_trigger(self):
        """try_transition() with non-matching trigger returns False."""
        sm = ToolStateMachine(_make_config())
        result = sm.try_transition("user_confirmation", {"intent": "planning"})
        assert result is False
        assert sm.current_state_name == "default"

    def test_transition_non_matching_condition(self):
        """try_transition() with non-matching condition returns False."""
        sm = ToolStateMachine(_make_config())
        result = sm.try_transition("agent_intent", {"intent": "wrong"})
        assert result is False

    def test_transition_from_any(self):
        """try_transition() from 'any' works from any current state."""
        sm = ToolStateMachine(_make_config())
        sm.try_transition("agent_intent", {"intent": "planning"})
        assert sm.current_state_name == "planning"
        result = sm.try_transition("pipeline_signal")
        assert result is True
        assert sm.current_state_name == "default"

    def test_transition_to_unknown_state(self):
        """try_transition() to unknown state returns False."""
        config = _make_config()
        config.transitions.append(
            ToolTransitionConfig(
                from_state="default", to_state="nonexistent", trigger="agent_intent",
            )
        )
        sm = ToolStateMachine(config)
        result = sm.try_transition("agent_intent", {})
        assert result is False

    def test_state_history_tracks(self):
        """state_history tracks all transitions."""
        sm = ToolStateMachine(_make_config())
        sm.try_transition("agent_intent", {"intent": "planning"})
        sm.try_transition("agent_intent", {"intent": "executing"})
        assert sm.state_history == ["default", "planning", "executing"]

    def test_reset(self):
        """reset() returns to initial state and clears history."""
        sm = ToolStateMachine(_make_config())
        sm.try_transition("agent_intent", {"intent": "planning"})
        sm.reset()
        assert sm.current_state_name == "default"
        assert sm.state_history == ["default"]

    def test_get_masking_output(self):
        """get_masking_output() returns strategy-specific output."""
        sm = ToolStateMachine(_make_config())
        sm.try_transition("agent_intent", {"intent": "planning"})
        output = sm.get_masking_output()
        assert "allowed_tools" in output
        assert output["allowed_tools"] == ["read_file"]

    def test_nested_condition(self):
        """Nested condition evaluates correctly."""
        config = _make_config()
        config.transitions.append(
            ToolTransitionConfig(
                from_state="default", to_state="executing",
                trigger="agent_action",
                condition="last_tool_call.status == 'error'",
            ),
        )
        sm = ToolStateMachine(config)
        result = sm.try_transition(
            "agent_action",
            {"last_tool_call": {"status": "error"}},
        )
        assert result is True
        assert sm.current_state_name == "executing"

    def test_get_denied_tools(self):
        """get_denied_tools() returns correct list."""
        sm = ToolStateMachine(_make_config())
        sm.try_transition("agent_intent", {"intent": "planning"})
        sm.try_transition("agent_intent", {"intent": "executing"})
        denied = sm.get_denied_tools()
        assert denied == ["rm"]

    def test_denied_tools_precedence_over_allowed_all(self):
        """denied_tools takes precedence over allowed_tools='all'."""
        sm = ToolStateMachine(_make_config())
        # Transition to 'executing' which has allowed_tools="all", denied_tools=["rm"]
        sm.try_transition("agent_intent", {"intent": "planning"})
        sm.try_transition("agent_intent", {"intent": "executing"})
        allowed = sm.get_allowed_tools()
        assert "rm" not in allowed
        assert "read_file" in allowed
        assert "write_file" in allowed
        assert "bash" in allowed

    def test_state_history_tracks_all_visited_states(self):
        """State history tracks all states visited in order, including initial."""
        sm = ToolStateMachine(_make_config())
        assert sm.state_history == ["default"]
        sm.try_transition("agent_intent", {"intent": "planning"})
        assert sm.state_history == ["default", "planning"]
        sm.try_transition("agent_intent", {"intent": "executing"})
        assert sm.state_history == ["default", "planning", "executing"]
        # Reset via pipeline_signal back to default
        sm.try_transition("pipeline_signal")
        assert sm.state_history == ["default", "planning", "executing", "default"]
