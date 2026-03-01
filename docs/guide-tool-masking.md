# Tool Masking

Tool masking controls which tools your agent can see at any point in a conversation. Instead of dumping all 50 tools into every LLM call, you define named states with allowed/denied lists and transition rules that move between them.

## Why It Matters

More tools in context = more tokens consumed + more confusion for the LLM. A coding agent in "planning" mode doesn't need `write_file` or `bash`. A research agent doesn't need deployment tools. Tool masking lets you expose only what's relevant, reducing cost and improving tool selection accuracy.

## States

A tool state defines which tools are visible. Every pipeline starts in an initial state (default: `"default"`).

```yaml
tool_states:
  - name: default
    allowed_tools: all       # everything visible

  - name: planning
    allowed_tools:
      - read_file
      - search
      - list_dir             # read-only tools

  - name: executing
    allowed_tools: all
    denied_tools:
      - rm
      - drop_table           # everything except destructive tools
```

### State Rules

- `allowed_tools: all` — every tool is visible (minus any in `denied_tools`)
- `allowed_tools: [list]` — only these tools are visible
- `denied_tools` takes precedence over `allowed_tools` — if a tool is in both, it's denied

### Default Configuration

If you don't configure tool states, you get a single `default` state with all tools allowed:

```yaml
tool_masking:
  strategy: allowed_list
  initial_state: default

tool_states:
  - name: default
    allowed_tools: all
```

## Transitions

Transitions define rules for moving between states. Each transition has a trigger (what event causes it) and an optional condition (when to fire).

```yaml
tool_transitions:
  - from_state: default
    to_state: planning
    trigger: agent_intent
    condition: "intent == 'planning'"

  - from_state: planning
    to_state: executing
    trigger: agent_intent
    condition: "intent == 'executing'"

  - from_state: any          # matches any current state
    to_state: default
    trigger: pipeline_signal
```

### Transition Fields

| Field | Required | Description |
|-------|----------|-------------|
| `from_state` | yes | Current state this transition fires from. Use `"any"` to match all states. |
| `to_state` | yes | State to transition to. Must reference a defined state name. |
| `trigger` | yes | Event type that activates this transition (e.g., `agent_intent`, `agent_action`, `pipeline_signal`) |
| `condition` | no | Expression evaluated against the trigger context. Supports dotted access (e.g., `last_tool_call.status == 'error'`). |

### Trigger Types

| Trigger | When It Fires | Typical Use |
|---------|--------------|-------------|
| `agent_intent` | When intent detection classifies the agent's goal | Switch between planning/executing modes |
| `agent_action` | After a tool call completes | React to tool results (e.g., errors) |
| `pipeline_signal` | System event from the pipeline | Reset to default state |

### Condition Evaluation

Conditions are evaluated against a context dict. Simple equality checks and dotted access are supported:

```yaml
# Simple
condition: "intent == 'planning'"

# Nested
condition: "last_tool_call.status == 'error'"
```

If no condition is specified, the transition fires whenever the trigger matches.

## Masking Strategies

The masking strategy determines *how* tool visibility is enforced at the LLM API level. Set via `tool_masking.strategy`:

### `allowed_list` (default)

Filters the tool schemas sent to the LLM. Only allowed tools appear in the API call.

```yaml
tool_masking:
  strategy: allowed_list
```

**Output:** `{"allowed_tools": ["read_file", "search"], "tool_schemas": [...]}`

Best for: Claude, GPT-4, and any API that accepts a `tools` parameter. This is the simplest and most compatible strategy.

### `prefill`

Uses assistant response prefill to force the LLM to call a specific tool. Only supported by APIs that accept response prefill (e.g., Claude).

```yaml
tool_masking:
  strategy: prefill
```

**Output:** `{"response_prefix": "{\"tool\": \"bash\"...", "forced_tool": "bash"}`

Best for: forcing a specific tool call in a controlled workflow. Less flexible than `allowed_list` — only forces one tool at a time.

### `logit_mask`

Returns token-level masks for allowed vs denied tool names. Used with self-hosted inference (vLLM, TGI) where you can apply logit biases.

```yaml
tool_masking:
  strategy: logit_mask
```

**Output:** `{"allowed_tool_tokens": ["read_file", "bash"], "denied_tool_tokens": ["write_file", "rm"]}`

Best for: self-hosted models where you control the decoding layer.

### `none`

All tools visible regardless of state. Disables masking entirely.

```yaml
tool_masking:
  strategy: none
```

## State Machine Behavior

The `ToolStateMachine` tracks:

- **Current state** — which state the agent is in right now
- **State history** — ordered list of all states visited (e.g., `["default", "planning", "executing"]`)
- **Allowed/denied tools** — derived from the current state's configuration

### API

```python
# Check current state
sm.current_state_name  # "planning"

# Get allowed tools for the current state
sm.get_allowed_tools()  # ["read_file", "search"]

# Get denied tools for the current state
sm.get_denied_tools()   # ["rm"]

# Try a transition
sm.try_transition("agent_intent", {"intent": "executing"})  # True if matched

# Get masking output (strategy-specific format)
sm.get_masking_output()  # {"allowed_tools": [...], "tool_schemas": [...]}

# View transition history
sm.state_history  # ["default", "planning", "executing"]

# Reset to initial state
sm.reset()
```

### Transition Evaluation

When `try_transition()` is called:

1. Find transitions matching the trigger and current state (or `from_state: any`)
2. Evaluate the condition against the context dict
3. If condition passes (or no condition), transition to the target state
4. If the target state doesn't exist, the transition is rejected (returns `False`)
5. State history is updated

## MCP Tool Management

For agents using MCP servers, SR2 provides an `MCPToolManager` with three strategies for controlling which MCP tools appear in context:

| Strategy | Behavior |
|----------|----------|
| `curated` | Only tools listed in `curated_tools` per server |
| `curated_with_discovery` | Curated tools + a `discover_mcp_tool` meta-tool for finding more |
| `all_in_context` | Every tool from every connected MCP server |

```python
from sr2.tools.mcp_manager import MCPToolManager, MCPToolConfig

manager = MCPToolManager(
    strategy="curated_with_discovery",
    mcp_configs=[
        MCPToolConfig("filesystem", curated_tools=["read_file", "write_file"]),
        MCPToolConfig("database", curated_tools=None),  # all tools from this server
    ],
    all_available_tools=discovered_tools,
)

tools = manager.get_context_tools()
# -> [read_file, write_file, query_db, insert_row, discover_mcp_tool]
```

The `discover_mcp_tool` meta-tool lets the agent search for tools it doesn't currently have in context:

```python
results = await manager.discover("directory listing")
# -> [ToolDefinition(name="list_dir", ...)]
```

## Full Example

A coding agent with three modes:

```yaml
tool_masking:
  strategy: allowed_list
  initial_state: default

tool_states:
  - name: default
    allowed_tools: all

  - name: planning
    allowed_tools:
      - read_file
      - search
      - list_dir
      - discover_mcp_tool

  - name: executing
    allowed_tools: all
    denied_tools:
      - rm
      - drop_database

tool_transitions:
  - from_state: default
    to_state: planning
    trigger: agent_intent
    condition: "intent == 'planning'"

  - from_state: planning
    to_state: executing
    trigger: agent_intent
    condition: "intent == 'executing'"

  - from_state: executing
    to_state: default
    trigger: agent_action
    condition: "last_tool_call.status == 'error'"

  - from_state: any
    to_state: default
    trigger: pipeline_signal
```

This agent starts in `default` (all tools). When it starts planning, it transitions to `planning` (read-only). When it begins executing, it gets write access but not destructive tools. If a tool call errors, it falls back to `default`. A pipeline signal (e.g., session end) resets everything.
