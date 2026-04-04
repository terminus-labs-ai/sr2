"""Tool masking — state machines and dynamic tool visibility.

Demonstrates:
- Defining tool states (default, planning, executing)
- Transitions between states based on triggers and conditions
- Three masking strategies (allowed_list, prefill, logit_mask)
- State history tracking

No LLM needed — shows the state machine mechanics directly.

Run:
    pip install sr2
    python examples/04_tool_masking.py
"""

from sr2.tools.masking import get_masking_strategy
from sr2.tools.models import (
    ToolDefinition,
    ToolManagementConfig,
    ToolStateConfig,
    ToolTransitionConfig,
)
from sr2.tools.state_machine import ToolStateMachine


def main():
    # --- 1. Define tools ---
    tools = [
        ToolDefinition(name="read_file", description="Read a file from disk", category="read"),
        ToolDefinition(name="search", description="Search for files by content", category="read"),
        ToolDefinition(name="write_file", description="Write content to a file", category="write"),
        ToolDefinition(name="bash", description="Execute a shell command", category="write"),
        ToolDefinition(name="rm", description="Delete a file permanently", category="dangerous"),
    ]

    # --- 2. Define states ---
    states = [
        ToolStateConfig(
            name="default",
            description="All tools available",
            allowed_tools="all",
        ),
        ToolStateConfig(
            name="planning",
            description="Read-only tools for research and planning",
            allowed_tools=["read_file", "search"],
        ),
        ToolStateConfig(
            name="executing",
            description="Write tools enabled, destructive tools blocked",
            allowed_tools="all",
            denied_tools=["rm"],
        ),
    ]

    # --- 3. Define transitions ---
    transitions = [
        ToolTransitionConfig(
            from_state="default",
            to_state="planning",
            trigger="agent_intent",
            condition="intent == 'planning'",
        ),
        ToolTransitionConfig(
            from_state="planning",
            to_state="executing",
            trigger="agent_intent",
            condition="intent == 'executing'",
        ),
        ToolTransitionConfig(
            from_state="executing",
            to_state="planning",
            trigger="agent_action",
            condition="last_tool_call.status == 'error'",
        ),
        ToolTransitionConfig(
            from_state="any",  # matches any current state
            to_state="default",
            trigger="pipeline_signal",
        ),
    ]

    # --- 4. Create the state machine ---
    config = ToolManagementConfig(
        tools=tools,
        states=states,
        transitions=transitions,
        masking_strategy="allowed_list",
        initial_state="default",
    )
    sm = ToolStateMachine(config)

    # --- 5. Walk through a workflow ---
    print("=== Tool State Machine ===\n")

    # Initial state
    print(f"State: {sm.current_state_name}")
    print(f"Allowed: {sm.get_allowed_tools()}")
    print(f"Denied: {sm.get_denied_tools()}")
    print()

    # Agent decides to plan
    print("--- Trigger: agent starts planning ---")
    sm.try_transition("agent_intent", {"intent": "planning"})
    print(f"State: {sm.current_state_name}")
    print(f"Allowed: {sm.get_allowed_tools()}")
    print()

    # Agent ready to execute
    print("--- Trigger: agent starts executing ---")
    sm.try_transition("agent_intent", {"intent": "executing"})
    print(f"State: {sm.current_state_name}")
    print(f"Allowed: {sm.get_allowed_tools()}")
    print(f"Denied: {sm.get_denied_tools()}")
    print()

    # A tool call fails — fall back to planning
    print("--- Trigger: tool call error ---")
    sm.try_transition("agent_action", {"last_tool_call": {"status": "error"}})
    print(f"State: {sm.current_state_name}")
    print(f"Allowed: {sm.get_allowed_tools()}")
    print()

    # Pipeline signal resets everything
    print("--- Trigger: pipeline reset ---")
    sm.try_transition("pipeline_signal")
    print(f"State: {sm.current_state_name}")
    print()

    # Full history
    print(f"State history: {sm.state_history}")
    print()

    # --- 6. Compare masking strategies ---
    print("=== Masking Strategies ===\n")

    state = ToolStateConfig(name="planning", allowed_tools=["read_file", "search"])

    for strategy_name in ["allowed_list", "logit_mask"]:
        strategy = get_masking_strategy(strategy_name)
        output = strategy.apply(tools, state)
        print(f"Strategy: {strategy_name}")
        for key, value in output.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                print(f"  {key}: [{', '.join(t['name'] for t in value)}]")
            else:
                print(f"  {key}: {value}")
        print()

    # Prefill strategy (forces a single tool)
    from sr2.tools.masking import PrefillStrategy

    prefill = PrefillStrategy(forced_tool="search")
    output = prefill.apply(tools, state)
    print("Strategy: prefill (forced_tool='search')")
    print(f"  forced_tool: {output['forced_tool']}")
    print(f"  response_prefix: {output['response_prefix'][:60]}...")


if __name__ == "__main__":
    main()
