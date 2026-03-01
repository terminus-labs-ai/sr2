# Architecture

## High-Level Overview

```
                    ┌──────────────────────────────┐
                    │        Agent Harness          │
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
                    │    (handled by harness)       │
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
5. **Agent harness** sends compiled context to LLM
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
| `heartbeat` | Minimal (3k) | Compaction/summarization/retrieval disabled | Polling checks |
| `a2a_inbound` | Medium (8k) | Stateless, per-invocation | Called by other agents |

Each interface type gets its own PipelineConfig, allowing different token budgets, layer layouts, and feature toggles for different trigger types.
