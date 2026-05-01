# SR2 — Context Engineering Library for AI Agents

SR2 manages the full lifecycle of what goes into an LLM's context window:
compaction, caching, summarization, graceful degradation, and KV-cache optimization.

**Version:** 2.0.0-alpha (complete redesign from v1)

**Repository:** `revamp` branch — clean-slate rebuild following the [v2 redesign plan](/data/obsidian/projects/sr2/PLAN-sr2-v2-redesign.md).

---

## What it does

SR2 sits between your agent harness and the LLM. The harness:

1. Calls `sr2.process(config, inputs)` to get compiled context
2. Makes the LLM call (provider-specific — Anthropic, OpenAI, etc.)
3. Feeds the response back as `TurnResult`
4. Calls `sr2.post_process(turn_result)` for memory extraction and maintenance

SR2 never makes the LLM call. It only manages context.

```
┌─────────────────────────────────────┐
│         Public API (facade)         │  <-- only thing harness touches
│  config in -> process() -> context out│
│            post_process(turn_result) │
├─────────────────────────────────────┤
│         Plugin Registry             │  <-- entry-point discovery
│  providers, reducers, stores,       │
│  exporters                          │
├─────────────────────────────────────┤
│         Pipeline Engine             │  <-- interprets YAML config
│  layers -> resolve -> budget -> reduce
├─────────────────────────────────────┤
│         Subsystems (internal)       │
│  memory, compaction, summarization, │
│  degradation, cache, metrics        │
└─────────────────────────────────────┘
```

## Core principles

- **Declarative-first** — YAML config is the API. Users describe what they want, SR2 executes it.
- **Open/Closed Principle** — New capabilities added via plugins, never by modifying existing code.
- **Hard public/internal boundary** — Harness gets a facade with defined inputs/outputs. Cannot reach into SR2 internals.
- **Plugin model (open-core)** — Free plugins ship with core. Paid plugins register via entry points. Core never imports paid code.

## Design principles

SOLID, OCP, and DRY are enforced at every level:

- **SRP** — Each protocol has one responsibility. Each module owns one concept.
- **OCP** — Extension points via `runtime_checkable` protocols + Python entry points. New providers, reducers, stores, and exporters plug in without touching core.
- **DRY** — Shared types defined once, reused everywhere. Generic `PluginRegistry[T]` serves all extension points.

## Architecture

### Facade Boundary

Two entry points — that's it:

| Entry point | Timing | Returns |
|---|---|---|
| `process(config, inputs)` | Pre-LLM | `CompiledContext` with assembled layers + metrics |
| `post_process(turn_result)` | Post-LLM | `PostProcessResult` with memory extraction + maintenance |

### Layer Model

Layers are containers with named content providers. Each provider has its own config and token budget. Example:

```yaml
layers:
  - name: system_prompt
    cache: static

  - name: conversation
    window: 10
    max_tokens: 8000
    session_history:
      max_tokens: 8000
    memory:
      read:
        scope: [private]
        max_tokens: 2000
      write:
        scope: [private]
    compaction:
      rules: [schema_and_sample, result_summary]
```

Layer order determines KV-cache efficiency. Stable layers (system prompt, project context, tools) go first — they stay cached across turns. Dynamic layers (conversation) go last.

### Memory System

Two placement modes, same retrieval engine:

- **Standalone layer** — stable memories, can be cached. Changes rarely.
- **Conversation-interleaved** — dynamic memories injected per-turn based on topic. Swept by summarization.

Read and write are independent toggles with independent scopes: `private`, `project`, `team`, `shared`.

### Degradation

Degradation is *emergent from config*, not a hardcoded ladder:

1. **Circuit breakers** — per-provider. 3 failures -> open -> cooldown -> half-open test
2. **Priority shedding** — over budget? Shed lowest-priority layers first
3. **Fallback content** — provider failed? Use cached, static, or skip

No special-cased layer names. The user's priority assignments define their own degradation path.

### Plugin System

Four extension points, each defined by a runtime-checkable Protocol:

| Extension | Protocol | Examples |
|---|---|---|
| Content providers | `ContentProvider` | session_history, memory, tools |
| Reducers | `ContentReducer` | compaction, summarization, tool_summarizer |
| Memory stores | `MemoryStore` | in_memory, sqlite, postgres |
| Exporters | `MetricExporter` | OTel, Prometheus |

Plugins register via Python packaging entry points. Lazy discovery — nothing activates unless YAML config names it.

### Metrics

Always-on, always embedded in the response. No extra calls or setup:

```python
result = sr2.process(config, inputs)
result.context   # compiled messages
result.metrics   # what SR2 did and why
```

Export plugins (OTel, Prometheus) consume the same snapshots — they subscribe, they don't collect.

## Quick start

See [QUICKSTART.md](QUICKSTART.md) for getting running in 5 minutes.

## Project structure

```
src/sr2/
  __init__.py           # Public API exports
  sr2.py                # SR2 facade (process / post_process)

  protocols/            # Extension point contracts (runtime_checkable)
    __init__.py         # ContentProvider, ContentReducer, MemoryStore, MetricExporter

  core/                 # Domain models & errors
    models.py           # Memory, TurnResult, ToolCall, TokenUsage, enums
    errors.py           # SR2Error hierarchy

  config/               # Declarative YAML config
    models.py           # PipelineConfig, LayerConfig, provider configs
    loader.py           # YAML parsing with extends inheritance

  pipeline/             # Context compilation engine
    engine.py           # Layer resolution, budget enforcement, caching
    result.py           # CompiledContext, PostProcessResult, PipelineMetrics

  plugins/              # Plugin infrastructure
    registry.py         # Generic PluginRegistry[T] with lazy entry-point discovery

  degradation/          # Graceful failure handling
    circuit_breaker.py  # Per-provider circuit breaker (closed/open/half-open)

  tokenization/         # Token counting
    counting.py         # tiktoken-based count_tokens() and truncate_to_tokens()

  memory/               # Memory subsystem (in progress)
  compaction/           # Content compaction (in progress)
  summarization/        # Conversation summarization (in progress)
  cache/                # KV-cache policies (in progress)
  metrics/              # Metric collection (in progress)
  resolvers/            # Content resolvers (in progress)
  tools/                # Tool management (in progress)
```

## Status

v2.0.0-alpha. Scaffolding and core contracts complete. Memory, compaction, summarization, and cache subsystems in progress.

## License

TBD
