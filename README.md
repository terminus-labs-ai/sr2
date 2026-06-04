# SR2 — A Context Runtime for AI Agents

SR2 manages the full lifecycle of what goes into an LLM's context window:
layer assembly, compaction, summarization, memory, token budgeting, KV-cache
alignment, and graceful degradation.

**Version:** 2.0.0-alpha (`revamp` branch — clean-slate rebuild)

---

## What SR2 is

SR2 is **not a library you call once, and not a harness.** It's a distinct
middle tier in the agent stack: a **context runtime** that sits between your
agent harness and the LLM provider.

```
┌──────────────────────────────────────────────────────────┐
│ HARNESS              your agent / Claude Code / LangGraph   │
│   owns: I/O, session identity, agent personality,           │
│         tool *implementations*, config                       │
└───────────────────────────┬─────────────────────────────────┘
                            │  turn(user_input)
                            ▼
┌──────────────────────────────────────────────────────────┐
│ SR2                  the context runtime                     │
│   owns: WHAT fills the window + how it evolves               │
│   drives the multi-iteration tool loop:                       │
│     compile → call LLM → run tools → repeat → post-process    │
│   layers · resolvers · transformers · memory · compaction     │
│   summarization · budgeting · cache alignment · degradation   │
└───────────────────────────┬─────────────────────────────────┘
                            │  CompletionRequest      (injected ↓)
                            ▼
┌──────────────────────────────────────────────────────────┐
│ LLMCallable          LiteLLMCallable (default) / your own     │
│   owns: the raw provider HTTP call (Anthropic, OpenAI, …)     │
└──────────────────────────────────────────────────────────┘
```

### Why it owns the LLM call

Context management is a **closed loop**, not a one-shot transform:

> compile context → call the model → observe the response → post-process
> (memory extraction, compaction, summarization, cache tracking)

A pure "compile-only" library forces the harness into a two-call dance —
compile, call the LLM yourself, then hand the response back — and every
guarantee SR2 makes (cache-stable prefixes, correct conversation-zone
transitions, reliable post-processing) depends on the harness getting that
dance exactly right. So SR2 **owns the turn loop** instead.

It does *not* own the provider HTTP call or tool execution. Both are
**injected dependencies** (inversion of control):

| Dependency | Type | Provided by | SR2's use |
|---|---|---|---|
| `llm` | `LLMCallable` (`complete` / `stream`) | harness | called each LLM iteration |
| `tool_executor` | `async (ToolUseBlock) -> ToolResultBlock` | harness | called per tool request; **SR2 never implements tools** |
| `token_counter` | `TokenCounter` | harness | budget enforcement |
| `memory_store`, `memory_extractor`, `tracer`, `provenance_store` | protocols | harness (optional) | memory, observability |

This is how SR2 owns the *lifecycle* without becoming a provider client or a
harness. The harness supplies adapters + I/O + personality; SR2 drives the
turn.

### Two delivery modes

The same SR2 core ships two ways:

- **Embedded** — import SR2, inject `LiteLLMCallable`, call `turn()` in-process.
  This is the primary mode (see Quick Start).
- **Gateway** — SR2 behind an HTTP endpoint, for foreign harnesses (Claude
  Code, Cursor, OpenAI-SDK clients) that can't embed it as a Python library.
  This is what `sr2-relay` provides; it's a thin wire-format adapter over the
  same `turn()`.

## The turn loop

`SR2.turn(user_input) -> AsyncIterator[StreamEvent]` is the entry point. Per
turn it:

1. Compiles context from the configured layers (resolvers → transformers → budget)
2. Streams an LLM response via the injected `LLMCallable`
3. If the model requested tools: runs them through the injected `tool_executor`
   (concurrently, order-preserving), appends results, and loops
4. Repeats until the model stops requesting tools (capped by `max_tool_iterations`)
5. Emits a single `end` event, then `await`s `post_process`

Callers see a clean stream: `text` and `usage` events, loop-progress events
(`tool_use_emitted`, `tool_result_received`, `iteration_complete`), and exactly
one `end`. At most one `error` event may appear per phase, always before `end`.

## Quick start

```python
import asyncio
from sr2 import SR2
from sr2.config.models import PipelineConfig
from sr2.integrations.litellm import LiteLLMCallable
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.models import TextBlock

config = PipelineConfig(
    layers=[
        {"name": "system", "target": "system",
         "resolvers": [{"type": "static",
                        "config": {"text": "You are a helpful assistant."}}]},
        {"name": "conversation", "target": "messages",
         "resolvers": [{"type": "session"}, {"type": "input"}]},
    ]
)

sr2 = SR2(
    config,
    llm=LiteLLMCallable("anthropic/claude-haiku-4-5-20251001"),
    token_counter=CharacterTokenCounter(),
)

async def main():
    async for event in sr2.turn([TextBlock(text="Hello")]):
        if event.type == "text":
            print(event.text, end="")

asyncio.run(main())
```

Or run a config straight from the CLI:

```bash
sr2 configs/minimal.yaml
```

See [QUICKSTART.md](QUICKSTART.md) for the YAML config shape, tool wiring, and
multi-turn sessions.

## How context is built

### Event-driven pipeline

The pipeline runs on an **EventBus**, not a fixed sequence. Components subscribe
to event names (`turn_start`, `user_input`, `assistant_response`, …); the engine
drives a drain-process loop each turn until no component produces new work.

### Layers

A **layer** is the unit of context. Each layer holds **resolvers** (produce
content), **transformers** (reshape content), and optional **tool providers**
(produce tool definitions). Layers compile to one of three targets — `system`,
`messages`, or `tools` — and enforce a per-layer token budget.

Layer order drives KV-cache efficiency: stable layers (system prompt, project
context) go first and stay cached; dynamic layers (conversation) go last.

### Extension model

Four extension points, each a `runtime_checkable` Protocol, discovered lazily
via Python entry points — nothing activates unless your config names it:

| Group | Protocol | Built-ins |
|---|---|---|
| `sr2.resolvers` | `Resolver` | `static`, `input`, `session`, `event_payload`, `memory` |
| `sr2.transformers` | `Transformer` | `summarize`, `memory_extraction`, `compaction` |
| `sr2.extractors` | `MemoryExtractor` | `rule_based` |
| `sr2.tool_providers` | `ToolProvider` | *(extension point; no built-ins yet)* |

Add a capability by registering an entry point — never by modifying the engine
(Open/Closed).

## Subsystems

- **Memory** — `MemoryStore` / `MemoryExtractor` protocols; `MemoryResolver`
  injects relevant memories on input, `MemoryExtractionTransformer` saves them
  from responses. In-memory store ships with core.
- **Compaction** — rule-based block compression (`schema_and_sample`,
  `result_summary`) with an optional cost gate (cache-invalidation economics).
- **Summarization** — LLM-powered compression of older content.
- **Degradation** — per-provider `CircuitBreaker` (closed/open/half-open) plus a
  5-level `DegradationLadder`.
- **Provenance & tracing** — every content block's origin is recorded
  (`InMemoryProvenanceStore` or durable `SQLiteProvenanceStore`); `Tracer`
  captures per-component firing records with a human-readable timeline.

## Design principles

- **Context runtime, not a library or a harness** — owns the turn loop;
  delegates the provider call and tool execution via dependency injection.
- **Event-driven** — components react to events; no hardcoded execution order.
- **Config-driven** — the pipeline is declared in YAML/Pydantic; no context
  logic in agent code.
- **Open/Closed** — new resolvers, transformers, extractors, stores plug in via
  entry points without touching the engine.
- **Single package** — v2 is one clean package (`src/sr2/`). The v1 multi-package
  monorepo, runtime, bridge, and premium split are gone.
- **Async throughout** — LLM calls and provenance I/O are async. Memory stores use a sync protocol (dict-backed `InMemoryMemoryStore`); async persistence backends can be added via the `MemoryStore` protocol.

## Project structure

The authoritative module map and architecture reference lives in
[CLAUDE.md](CLAUDE.md). High level:

```
src/sr2/
  __init__.py       # public API — `from sr2 import SR2`, plus core models
  sr2.py            # facade re-export of the SR2 orchestrator
  orchestrator.py   # SR2 class: the turn loop
  models.py         # core data models (TextBlock, Message, ToolDefinition, …)
  protocols/llm.py  # LLMCallable, CompletionRequest/Response, StreamEvent
  integrations/     # LiteLLMCallable
  config/           # Pydantic config models
  pipeline/         # engine, event bus, layers, resolvers, transformers, tracing
  memory/           # stores, extractors, resolver
  compaction/       # rule engine + cost gate
  degradation/      # circuit breaker, ladder
  tokenization/     # tiktoken counter
```

## Status

v2.0.0-alpha on `revamp`. Turn loop, event-driven pipeline, layers,
resolvers/transformers, memory, compaction, degradation, provenance, and tracing
are in place. `post_process` is a wired no-op pending the extraction/maintenance
hookup.

## Documentation

- [QUICKSTART.md](QUICKSTART.md) — config shape, tools, multi-turn
- [CLAUDE.md](CLAUDE.md) — full architecture reference and module map
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow and standards

## License

TBD
