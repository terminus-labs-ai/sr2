# Architecture

## High-Level Overview

```
                    ┌──────────────────────────────┐
                    │        Agent Runtime          │
                    │   (your code / framework)     │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     InterfaceRouter           │
                    │  user_message | heartbeat |   │
                    │  a2a_inbound                  │
                    └──────────┬───────────────────┘
                               │ PipelineConfig
                    ┌──────────▼───────────────────┐
                    │      PipelineEngine           │
                    │  Layer 1 → Layer 2 → Layer N  │
                    │  (resolve, cache, budget)     │
                    └──────────┬───────────────────┘
                               │ CompiledContext
                    ┌──────────▼───────────────────┐
                    │         LLM Call              │
                    │    (handled by runtime)       │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │    PostLLMProcessor           │
                    │  memory | compaction |        │
                    │  summarization | conflicts    │
                    └──────────────────────────────┘
```

## Pipeline Flow

1. **Trigger arrives** — User message, heartbeat timer, or A2A request
2. **InterfaceRouter** selects the PipelineConfig for this trigger type
3. **PipelineEngine.compile()** processes each layer:
   - Check cache policy — should we recompute?
   - Resolve content items via registered resolvers
   - Enforce token budget (trim from last layers first)
4. **CompiledContext** returned — content string + token count + metrics
5. **Agent runtime** sends compiled context to LLM
6. **PostLLMProcessor** runs async post-processing:
   - Memory extraction from the conversation turn
   - Conflict detection and resolution
   - Compaction of older turns
   - Summarization when thresholds are met

## Three-Zone Conversation Management

```
┌─────────────────────────────────────────────┐
│              Conversation Window             │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │Summarized│  │ Compacted │  │    Raw    │  │
│  │  Zone    │  │   Zone    │  │   Zone    │  │
│  │         │  │           │  │           │  │
│  │ Running  │  │ Rule-based│  │ Verbatim  │  │
│  │ summary  │  │ compacted │  │ recent    │  │
│  │ of old   │  │ turns     │  │ turns     │  │
│  │ content  │  │           │  │           │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│                                             │
│  ◄── oldest ──────────────── newest ──►     │
└─────────────────────────────────────────────┘
```

- **Raw zone** — Last N turns (configurable via `raw_window`), kept verbatim
- **Compacted zone** — Older turns processed by compaction rules (tool results summarized, code output truncated, etc.)
- **Summarized zone** — Oldest content collapsed into a structured summary with key decisions, unresolved issues, and user preferences

## Multi-Agent with A2A

```
┌──────────────┐     A2A Protocol      ┌──────────────┐
│  Agent A     │◄─────────────────────►│  Agent B     │
│              │                       │              │
│ A2AClient ──►│   POST /a2a/message   │◄── A2AServer │
│              │                       │              │
│ AgentCard    │   GET /agent.json     │   AgentCard  │
└──────────────┘                       └──────────────┘
```

- **AgentCardGenerator** — Auto-generates A2A Agent Cards from config
- **A2AServerAdapter** — Routes inbound A2A messages through the pipeline
- **A2AClientTool** — Tool that agents call to send requests to other agents
- **A2AClientRegistry** — Manages multiple remote agent connections

## Interface Types

| Interface | Budget | Features | Use Case |
|---|---|---|---|
| `user_message` | Full (48k) | All features enabled | Chat with user |
| `heartbeat` | Minimal (3k) | Compaction/summarization/retrieval disabled | Scheduled callbacks, async monitoring |
| `a2a_inbound` | Medium (8k) | Stateless, per-invocation | Called by other agents |

Each interface type gets its own PipelineConfig, allowing different token budgets, layer layouts, and feature toggles for different trigger types.

### Heartbeat System

Agents can schedule future callbacks via `schedule_heartbeat` / `cancel_heartbeat` tools. The `HeartbeatScanner` polls the store for due heartbeats and fires them as synthetic triggers through `_handle_trigger`, creating ephemeral sessions with context from the original conversation.

```
Agent calls schedule_heartbeat(delay=300, prompt="check X", key="k")
  → Heartbeat stored (pending)
  → HeartbeatScanner polls every 30s
  → When fire_at <= now: fires TriggerContext(interface="heartbeat", input=prompt)
  → New ephemeral session with last N turns from original session
  → Agent processes prompt with carried context
```

Heartbeats support idempotent keys (same key updates instead of duplicating), cancellation, and DB persistence. See [Heartbeat Guide](guide-heartbeats.md).

## Bridge Mode

The bridge is a separate runtime mode that applies SR2's context optimization to *external* LLM callers without owning the LLM loop.

```
External Caller (Claude Code, LangChain, etc.)
    │  Full message history
    ▼
┌──────────────────────────────────────────────┐
│              SR2 Bridge Server               │
│                                              │
│  ┌────────────────┐  ┌───────────────────┐   │
│  │ProtocolAdapter │  │  SessionTracker   │   │
│  │ (Anthropic)    │  │  (system_hash)    │   │
│  └───────┬────────┘  └────────┬──────────┘   │
│          │                    │               │
│  ┌───────▼────────────────────▼──────────┐   │
│  │           BridgeEngine                │   │
│  │  CompactionEngine + ConversationMgr   │   │
│  │  + SummarizationEngine (optional)     │   │
│  └───────────────────┬───────────────────┘   │
│                      │  Optimized messages    │
│  ┌───────────────────▼───────────────────┐   │
│  │         BridgeForwarder (httpx)       │   │
│  └───────────────────┬───────────────────┘   │
└──────────────────────┼───────────────────────┘
                       │  Original auth headers
                       ▼
              Upstream API (api.anthropic.com)
```

Key differences from the agent runtime:
- **Doesn't own the LLM loop** — the external caller drives the conversation
- **Full history on every request** — Claude Code sends all messages each time, so the bridge compares to last known count and only processes new messages
- **Auth passthrough** — preserves the caller's OAuth/API key headers for upstream forwarding
- **Session tracking by system prompt hash** — Claude Code's system prompt is session-specific, so hashing it naturally groups related requests

The bridge reuses the same CompactionEngine, ConversationManager, and SummarizationEngine as the agent runtime, just wired differently. See [Bridge Guide](guide-bridge.md).
