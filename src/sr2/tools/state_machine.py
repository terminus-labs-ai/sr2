"""Tool state machine managing availability transitions during agent execution."""

from sr2.tools.masking import get_masking_strategy
from sr2.tools.models import ToolManagementConfig, ToolStateConfig


class ToolStateMachine:
    """Manages tool state transitions and masking during agent execution."""

    def __init__(self, config: ToolManagementConfig):
        self._config = config
        self._tools = {t.name: t for t in config.tools}
        self._states = {s.name: s for s in config.states}
        self._transitions = config.transitions
        self._current_state_name = config.initial_state
        self._strategy = get_masking_strategy(config.masking_strategy)
        self._history: list[str] = [config.initial_state]

    @property
    def current_state(self) -> ToolStateConfig:
        return self._states[self._current_state_name]

    @property
    def current_state_name(self) -> str:
        return self._current_state_name

    @property
    def state_history(self) -> list[str]:
        return list(self._history)

    def get_masking_output(self) -> dict:
        """Get the current masking output for the LLM call."""
        return self._strategy.apply(
            list(self._tools.values()),
            self.current_state,
        )

    def get_allowed_tools(self) -> list[str]:
        """Get list of currently allowed tool names."""
        return [name for name in self._tools if self.current_state.is_tool_allowed(name)]

    def get_denied_tools(self) -> list[str]:
        """Get list of currently denied tool names."""
        return [name for name in self._tools if not self.current_state.is_tool_allowed(name)]

    def try_transition(self, trigger: str, context: dict | None = None) -> bool:
        """Attempt a state transition.

        Returns True if transition occurred, False if no matching transition found.
        """
        context = context or {}

        for t in self._transitions:
            if t.trigger != trigger:
                continue
            if t.from_state != "any" and t.from_state != self._current_state_name:
                continue
            if t.to_state not in self._states:
                continue
            if t.condition and not self._evaluate_condition(t.condition, context):
                continue

            self._current_state_name = t.to_state
            self._history.append(t.to_state)
            return True

        return False

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_state_name = self._config.initial_state
        self._history = [self._config.initial_state]

    def _evaluate_condition(self, condition: str, context: dict) -> bool:
        """Evaluate a simple condition string against context.

        Supports simple key == value checks:
          "intent == 'planning'" -> context.get("intent") == "planning"
        """
        condition = condition.strip()
        if "==" not in condition:
            return False

        parts = condition.split("==", 1)
        if len(parts) != 2:
            return False

        key = parts[0].strip()
        expected = parts[1].strip().strip("'\"")

        actual = self._get_nested(context, key)
        return str(actual) == expected

    def _get_nested(self, data: dict, key: str):
        """Get a nested value using dot notation."""
        parts = key.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current
