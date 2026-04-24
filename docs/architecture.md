# Architecture

## High-Level Overview

```
                    ┌──────────────────────────────┐
                    │      Your Application         │
                    │   (your code / framework)     │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     InterfaceRouter           │
                    │  user_message | heartbeat |   │
                    │  webhook | ...                │
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
                    │    (handled by your code)     │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │    PostLLMProcessor           │
                    │  memory | compaction |        │
                    │  summarization | conflicts    │
                    └──────────────────────────────┘
```

## Pipeline Flow

1. **Trigger arrives** — User message, webhook, or scheduled callback
2. **InterfaceRouter** selects the PipelineConfig for this trigger type
3. **PipelineEngine.compile()** processes each layer:
   - Check cache policy — should we recompute?
   - Resolve content items via registered resolvers
   - Enforce token budget (trim from last layers first)
4. **CompiledContext** returned — content string + token count + metrics
5. **Your code** sends compiled context to LLM
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

## Interface Types

| Interface | Budget | Features | Use Case |
|---|---|---|---|
| `user_message` | Full (48k) | All features enabled | Chat with user |
| `heartbeat` | Minimal (3k) | Compaction/summarization/retrieval disabled | Scheduled callbacks, background tasks |

Each interface type gets its own PipelineConfig, allowing different token budgets, layer layouts, and feature toggles for different trigger types.

## Pipeline Inspection

Every pipeline invocation can be traced via the **TraceCollector** subsystem. Trace events are emitted at each stage (resolve, retrieve, compact, cache) with timing, token counts, and diagnostics.

```
┌─────────────────────────────────────────────────────────────┐
│                     TraceCollector                          │
│                                                             │
│  Components emit:                                           │
│    PipelineEngine  ──► emit("resolve", {layers, tokens})    │
│    HybridRetriever ──► emit("retrieve", {candidates, k})   │
│    CompactionEngine──► emit("compact", {original, result})  │
│    Your code       ──► emit("cache", {hit_rate, prefix})    │
│                                                             │
│  Ring buffer (100 turns) ─► on_turn_complete callbacks      │
│                              │                              │
│                     ┌────────▼────────┐                     │
│                     │  TraceRenderer  │                     │
│                     │  default|full|  │                     │
│                     │  brief          │                     │
│                     └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

- **Always available** — TraceCollector activates when a collector is attached
- **Zero overhead when unused** — events are only collected if a collector is attached
- **`--inspect` flag** — enables CLI rendering (`default`, `full`, or `brief` modes)
- **Programmatic access** — `trace_collector.get_session_traces(session_id)` returns structured `TurnTrace` objects
