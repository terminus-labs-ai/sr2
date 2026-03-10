# Heartbeat System

Heartbeats let agents schedule future callbacks to themselves — for monitoring async work, retrying failed tasks, or timed reminders. The agent calls `schedule_heartbeat` during a conversation, and later receives a new session with context from the original conversation.

## How It Works

```
Agent session                   HeartbeatScanner
    │                                │
    ├─ calls schedule_heartbeat()    │
    │   delay=300, prompt="check X"  │
    │   key="monitor-x"             │
    │                                │
    │   Store: pending heartbeat     │
    │                                │
    ◄────── 5 minutes pass ─────────►│
    │                                ├─ polls store every 30s
    │                                ├─ finds due heartbeat
    │                                ├─ fires TriggerContext
    │                                │
    ├─ new ephemeral session ────────┤
    │   prompt: "check X"            │
    │   context: last 10 turns       │
    │   from original session        │
    └────────────────────────────────┘
```

## Configuration

Enable heartbeats in your `agent.yaml` under `runtime`:

```yaml
runtime:
  heartbeat:
    enabled: true                    # default: false
    poll_interval_seconds: 30        # how often scanner checks (min: 5)
    max_context_turns: 10            # turns carried from source session
    session_lifecycle: ephemeral     # ephemeral or persistent
    pipeline: interfaces/heartbeat.yaml  # optional, custom pipeline for heartbeat sessions
    max_pending_per_agent: 100       # max queued heartbeats
```

### Pipeline Override

If `pipeline` is set, heartbeat sessions use that pipeline config (typically with reduced token budget and disabled retrieval). If unset, heartbeats use the agent's default pipeline.

Example `interfaces/heartbeat.yaml`:
```yaml
extends: ../../defaults.yaml

token_budget: 3000

compaction:
  enabled: false

summarization:
  enabled: false

retrieval:
  enabled: false
```

## Tools

When heartbeats are enabled, two tools are automatically registered:

### `schedule_heartbeat`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `delay_seconds` | integer | yes | Seconds from now until the heartbeat fires |
| `prompt` | string | yes | What the agent should do when it wakes up |
| `key` | string | no | Idempotent key — if a heartbeat with this key exists, it's updated instead of duplicated |

**Returns:** Confirmation with the scheduled fire time.

**Idempotent keys:** If you schedule a heartbeat with `key="check-task-42"` and one already exists with that key, the existing heartbeat is updated (new delay, new prompt, new context). This prevents duplicate heartbeats for the same purpose.

### `cancel_heartbeat`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `key` | string | yes | Key of the heartbeat to cancel |

**Returns:** Confirmation or "not found". Only cancels `pending` heartbeats.

## Context Carry-Over

When a heartbeat fires, the last N turns (configured by `max_context_turns`) from the original session are injected into the new heartbeat session. This gives the agent context about what it was doing when it scheduled the callback.

The context turns appear as the session's initial history, so the agent can reference previous tool results, decisions, and conversation context.

## Persistence

- **With database:** Heartbeats survive restarts (PostgreSQL via `PostgresHeartbeatStore`)
- **Without database:** In-memory only — heartbeats are lost on restart

The store is selected automatically based on whether a database connection is configured.

## Lifecycle

Each heartbeat goes through these states:

```
pending → firing → completed
                 → failed
pending → cancelled
```

- **pending**: Waiting for `fire_at` time
- **firing**: Scanner picked it up, executing callback
- **completed**: Callback succeeded
- **failed**: Callback raised an exception
- **cancelled**: Agent called `cancel_heartbeat`

## Example Use Cases

**Monitor an async task:**
```
Agent: "I've kicked off a deployment. Let me check back in 10 minutes."
→ schedule_heartbeat(delay_seconds=600, prompt="Check deployment status for PR #42", key="deploy-42")
```

**Retry a failed operation:**
```
Agent: "API returned 503. I'll retry in 2 minutes."
→ schedule_heartbeat(delay_seconds=120, prompt="Retry the API call to /api/users", key="retry-users-api")
```

**Timed reminder:**
```
User: "Remind me to review the PR in an hour"
Agent: → schedule_heartbeat(delay_seconds=3600, prompt="Remind the user to review PR #15", key="remind-pr-15")
```

**Supervisor polling:**
```
Agent: "I'll check the task board every 10 minutes for stuck tasks."
→ schedule_heartbeat(delay_seconds=600, prompt="Check Galaxy Map for error/stuck tasks and take action", key="supervisor-check")
```
