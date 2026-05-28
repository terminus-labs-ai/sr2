# SR2 Project Instructions

## Project Overview

**SR2** is a context engineering library for AI agents. It manages what goes into an LLM's context window via an event-driven pipeline: resolvers populate layers, transformers reshape them, and the engine compiles everything into a `CompletionRequest`.

**Status**: v2 revamp (single-package, event-driven pipeline). Active development on the `revamp` branch.

**Tech stack**: Python 3.12+, Pydantic, LiteLLM, aiosqlite, tiktoken (optional), ULID.

**Single package**: Everything lives in `src/sr2/`. No multi-package monorepo — that was v1.

## Repository Structure

```
/home/shepard/git/sr2/
  src/sr2/
    sr2.py                      # Public facade — re-exports SR2 from orchestrator.py
    orchestrator.py             # SR2 class — top-level entry point
    cli.py                      # `sr2` CLI entry point (YAML config → chat loop)
    models.py                   # Core data models (TextBlock, Message, ToolDefinition, etc.)
    config/
      models.py                 # Pydantic config models (PipelineConfig, LayerConfig, etc.)
    protocols/
      llm.py                    # LLMCallable protocol, CompletionRequest/Response, StreamEvent
    integrations/
      litellm.py                # LiteLLMCallable — implements LLMCallable via litellm
    pipeline/
      engine.py                 # PipelineEngine — drain-process loop, compile, metrics
      layer.py                  # Layer — accumulates content, enforces budget, compiles
      event_bus.py              # EventBus — pub/sub, async drain, concurrent dispatch
      events.py                 # Event, EventPhase, EventSubscription models
      protocols.py              # Resolver, Transformer, ToolProvider, TokenCounter protocols
      models.py                 # CompilationTarget, ResolvedContent, TransformationResult, metrics
      compilation.py            # PositionStrategy: AppendStrategy, PrefixStrategy
      dependencies.py           # Dependencies — immutable DI container passed to components
      provenance.py             # Entry, EntryOrigin, ProvenanceStore protocol, InMemoryProvenanceStore
      tracing.py                # FiringRecord, Tracer protocol, CollectingTracer, render_trace()
      token_counting.py         # CharacterTokenCounter (fallback, no deps)
      utils.py                  # build_subscriptions(), extract_user_input_text(), PHASE_MAP
      resolvers/
        static.py               # StaticResolver — fixed text from config
        input.py                # InputResolver — wraps user_input event as Message
        session.py              # SessionResolver — accumulates conversation history across turns
        event_payload.py        # EventPayloadResolver — surfaces event.data as provenance entries
      transformers/
        summarization.py        # SummarizationTransformer — LLM-powered content compression
      stores/
        sqlite.py               # SQLiteProvenanceStore — aiosqlite-backed ProvenanceStore
    plugins/
      registry.py               # PluginRegistry[T] — lazy entry-point discovery
      errors.py                 # PluginNotFoundError, PluginCollisionError
    memory/
      schema.py                 # Memory, MemoryScope, MemorySearchResult, ExtractionResult
      store.py                  # InMemoryMemoryStore — dict-backed MemoryStore
      protocol.py               # MemoryStore, MemoryExtractor protocols
      extraction.py             # RuleBasedExtractor — heuristic memory extraction
      extraction_transformer.py # MemoryExtractionTransformer — extracts from assistant_response events
      memory_resolver.py        # MemoryResolver — injects relevant memories into context
      extractor_registry.py     # EXTRACTORS registry (PluginRegistry for sr2.extractors)
    tokenization/
      counting.py               # TiktokenTokenCounter (tiktoken, cl100k_base)
    compaction/
      engine.py                 # CompactionEngine — applies rule pipeline to Messages
      rules.py                  # Compaction rules: schema_and_sample, result_summary, etc.
      transformer.py            # CompactionTransformer — plugin wrapper around CompactionEngine
      llm_strategy.py           # LLM-powered compaction strategy
      cost_gate.py              # CompactionCostGate — economics check before compacting
    degradation/
      circuit_breaker.py        # CircuitBreaker — CLOSED/OPEN/HALF_OPEN state machine
      ladder.py                 # DegradationLadder — 5 levels from FULL to SYSTEM_PROMPT_ONLY
      registry.py               # Degradation policy registry
      fallback.py               # Fallback helpers
      shedding.py               # Load-shedding logic
  tests/                        # Test suite
  configs/                      # Example YAML configs
  pyproject.toml                # Single-package build (uv)
```

## Key Architecture Concepts

### Entry Points (pyproject.toml)

```toml
[project.scripts]
sr2 = "sr2.cli:main"

[project.entry-points."sr2.resolvers"]
static = "sr2.pipeline.resolvers.static:StaticResolver"
input = "sr2.pipeline.resolvers.input:InputResolver"
session = "sr2.pipeline.resolvers.session:SessionResolver"
event_payload = "sr2.pipeline.resolvers.event_payload:EventPayloadResolver"
memory = "sr2.memory.memory_resolver:MemoryResolver"

[project.entry-points."sr2.transformers"]
summarize = "sr2.pipeline.transformers.summarization:SummarizationTransformer"
memory_extraction = "sr2.memory.extraction_transformer:MemoryExtractionTransformer"
compaction = "sr2.compaction.transformer:CompactionTransformer"

[project.entry-points."sr2.extractors"]
rule_based = "sr2.memory.extraction:RuleBasedExtractor"
```

All component types are discovered lazily at runtime via `PluginRegistry` scanning these entry-point groups.

### SR2 Orchestrator

`SR2` (`orchestrator.py`) is the single top-level entry point. It:
1. Accepts a `PipelineConfig`, `llm: dict[str, LLMCallable]`, `TokenCounter`, and optional deps (provenance store, tracer, memory store/extractor).
2. Builds a `Dependencies` container and instantiates layers from config — each `LayerConfig` gets its resolvers, transformers, and tool providers resolved via `PluginRegistry`.
3. Creates a `PipelineEngine` that owns a shared `EventBus`.
4. Exposes `async turn(user_input)` → `AsyncIterator[StreamEvent]`.

```python
sr2 = SR2(pipeline_config, llm={"default": LiteLLMCallable("gpt-4o")}, token_counter=CharacterTokenCounter())
async for event in sr2.turn([TextBlock(text="hello")]):
    if event.type == "text":
        print(event.text, end="")
```

`post_process()` is a hook for post-turn work (memory extraction, summarization). Currently a no-op; called fire-and-forget after streaming completes.

### Event-Driven Pipeline

The pipeline runs on an **EventBus** (pub/sub). Components subscribe to event names; the engine drives a drain-process loop per turn.

**Lifecycle events emitted by engine per turn:**
- `turn_start` — fires at the start of each turn (triggers most resolvers)
- `user_input` — carries `list[ContentBlock]` as payload
- `turn_end` — fires after content drain (triggers post-turn transformers)
- `assistant_response` — queued by orchestrator after LLM stream completes; carries `CompletionResponse`

**Cascade events** (emitted by components):
- `overflow` — layer exceeded token budget
- `token_threshold` — layer crossed `token_threshold_pct`
- `truncation` — layer force-truncated content
- `summarization_complete` — SummarizationTransformer finished

**Drain-process loop** (`engine._run_loop()`):
1. Drain the bus (dispatch async callbacks concurrently)
2. Process all layers' pending events (fire resolvers, transformers, tool providers)
3. Repeat until bus is empty and no layer produced new work
4. Max 50 cycles per turn to prevent infinite cascades

### Layers

A `Layer` is the core pipeline unit. It:
- Holds `resolvers`, `transformers`, `tool_providers` (all implementing their respective protocols)
- Subscribes components to the shared bus; accumulates content as events arrive
- Enforces a `token_budget` (per-layer cap); emits `overflow` then `force_truncate` if exceeded
- Compiles to one of three `CompilationTarget`s: `SYSTEM`, `MESSAGES`, `TOOLS`
  - Target inferred from layer name (`"system"` → SYSTEM, `"tool"` → TOOLS, else MESSAGES)
  - Or overridden explicitly via `LayerConfig.target`
- Content placement: `append` (default) or `prefix` via `PositionStrategy`

### Protocols

All pipeline components implement runtime-checkable protocols from `pipeline/protocols.py`:

| Protocol | Method | Description |
|---|---|---|
| `Resolver` | `async resolve(events) -> ResolvedContent` | Produces content blocks |
| `Transformer` | `async transform(content, events) -> TransformationResult` | Reshapes layer content |
| `ToolProvider` | `async provide(events) -> list[ToolDefinition]` | Produces tool definitions |
| `TokenCounter` | `count(content) -> int` | Counts tokens in content blocks |

Each component also exposes `subscriptions: list[EventSubscription]`, `max_executions: int`, `execution_count: int`, and a `build(config, deps) -> Self` classmethod.

### Built-in Resolvers

| Name | Entry point key | Trigger | Output |
|---|---|---|---|
| `StaticResolver` | `static` | `turn_start` | Fixed `TextBlock` from config |
| `InputResolver` | `input` | `user_input` | `Message(role='user')` wrapping event data |
| `SessionResolver` | `session` | `user_input` + `assistant_response` | Prior turn history as `list[Message]` |
| `EventPayloadResolver` | `event_payload` | configurable | Event payload blocks as provenance entries |
| `MemoryResolver` | `memory` | `user_input` | Matched memories as `TextBlock` |

### Built-in Transformers

| Name | Entry point key | Trigger | Effect |
|---|---|---|---|
| `SummarizationTransformer` | `summarize` | configurable | LLM-compresses older blocks; keep_last_n or keep_within_tokens strategy |
| `MemoryExtractionTransformer` | `memory_extraction` | `assistant_response` | Extracts memories from response text, saves to MemoryStore; pass-through (content=None) |
| `CompactionTransformer` | `compaction` | configurable | Applies rule-based compaction (schema_and_sample, result_summary) to layer content |

### Config Models (`config/models.py`)

```python
class PipelineConfig(BaseModel):
    layers: list[LayerConfig]
    max_iterations: int = 100
    token_budget: int = 200_000
    enable_overflow_detection: bool = True

class LayerConfig(BaseModel):
    name: str
    cache: Literal["static", "ephemeral", "append_only"] | None = None
    token_budget: int | None = None
    token_threshold_pct: float | None = None
    resolvers: list[ResolverConfig]
    transformers: list[TransformerConfig] | None = None
    tool_providers: list[ToolProviderConfig] | None = None
    target: str | None = None    # explicit CompilationTarget override
    position: str = "append"     # "append" | "prefix"

class ResolverConfig(BaseModel):
    type: str                    # entry-point key (e.g. "static", "session")
    config: dict = {}            # resolver-specific options (live dict, not copied)
    subscriptions: list[EventSubscriptionConfig] = []
    max_executions: int = 1
```

`config` dicts use `_LiveDict` — Pydantic does not copy them, so hot-reload works correctly.

### LLM Integration

`LLMCallable` protocol (`protocols/llm.py`):
```python
class LLMCallable(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]: ...
```

`LiteLLMCallable` (`integrations/litellm.py`) implements this via `litellm.acompletion`. Handles tool calls, streaming, OpenAI-compatible endpoints (auto-adds `openai/` prefix for bare model names with a `base_url`).

The orchestrator takes `llm: dict[str, LLMCallable]` — must include `"default"` key. Named entries can be used by transformers that specify `config.model`.

### Token Counting

Two implementations:
- `CharacterTokenCounter` (`pipeline/token_counting.py`): `chars // 4`, zero-dep, default for CLI
- `TiktokenTokenCounter` (`tokenization/counting.py`): `cl100k_base` encoding, accurate, requires `tiktoken`

Injected into `PipelineEngine` and `Layer` at init time. Swappable.

### Provenance Tracking

`Entry` (`pipeline/provenance.py`): immutable record of a content block's origin (resolver or transformer), layer, session, and source entries (for transformer-derived entries).

- `InMemoryProvenanceStore`: dict-backed, no persistence (default)
- `SQLiteProvenanceStore` (`pipeline/stores/sqlite.py`): aiosqlite-backed, durable; supports `get_lineage()` BFS traversal

Entries are buffered in `layer._pending_writes` and flushed to the store after each `process_pending()` call.

### Plugin Registry

`PluginRegistry[T]` (`plugins/registry.py`): lazy entry-point scanner. Scans on first `get()` or `names()` call; detects collisions across distributions.

Three registries in the orchestrator:
- `_RESOLVERS`: `sr2.resolvers` group
- `_TRANSFORMERS`: `sr2.transformers` group
- `_TOOL_PROVIDERS`: `sr2.tool_providers` group

### Memory Subsystem

`MemoryStore` protocol (`memory/protocol.py`): `save()`, `search()`, `get_by_tag()`, `delete()`, `get_all()`.

`InMemoryMemoryStore` (`memory/store.py`): dict-backed, no persistence. Injected via `Dependencies.memory_store`.

`MemoryExtractor` protocol: `extract(turn_text) -> ExtractionResult`. Default impl: `RuleBasedExtractor` (regex patterns, up to 5 memories per turn).

`MemoryResolver`: retrieves relevant memories on `user_input` events and injects them as context. `MemoryExtractionTransformer`: extracts and saves memories on `assistant_response` events (side-effectful, does not modify layer content).

### Compaction

`CompactionEngine` (`compaction/engine.py`): applies an ordered list of block-level rules to `list[Message]`. Rules are `Callable[[ContentBlock], ContentBlock | None]`; first match wins.

Built-in rules (`compaction/rules.py`): `schema_and_sample` (tool output → schema + sample), `result_summary` (code results → exit code + truncated output).

`CompactionTransformer`: registered plugin wrapping CompactionEngine. `CompactionCostGate`: optional economics gate before compacting (estimates cache invalidation cost vs. token savings).

### Degradation

`CircuitBreaker` (`degradation/circuit_breaker.py`): CLOSED → OPEN (after N consecutive failures) → HALF_OPEN (after recovery timeout) → CLOSED. Per-provider isolation.

`DegradationLadder` (`degradation/ladder.py`): 5 levels — FULL, REDUCED_MEMORY, TOOLS_DISABLED, MEMORY_DISABLED, SYSTEM_PROMPT_ONLY. Monotonic step-down via `step_down()`; `reset()` returns to FULL. Reports `active_providers()` at each level.

### Tracing

`Tracer` protocol (`pipeline/tracing.py`): `on_firing(FiringRecord)`, `on_compile(CompletionRequest)`.

`CollectingTracer`: in-memory buffer, `get_trace() -> list[FiringRecord]`, `clear()`.

`render_trace(records)`: human-readable timeline grouped by turn_seq and firing_seq. `render_compiled_request(request)`: human-readable dump of the compiled request.

`FiringRecord`: immutable snapshot per component firing — component name, layer, trigger events, content before/after, token delta, duration_ms, status, error.

### Core Data Models (`models.py`)

```python
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock  # discriminated union
Message(role, content, turn_index, timestamp, zone)
ToolDefinition(name, description, input_schema)
TokenUsage(input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens)
Zone: RAW | COMPACTED | SUMMARIZED  # attached to Messages for conversation zone tracking
```

## CLI

```bash
sr2 configs/my_agent.yaml
```

Config YAML shape:
```yaml
models:
  default:
    model: gpt-4o
    base_url: http://localhost:8008  # optional, for local endpoints
pipeline:
  token_budget: 100000
  layers:
    - name: system
      resolvers:
        - type: static
          config:
            text: "You are a helpful assistant."
    - name: history
      resolvers:
        - type: session
    - name: input
      resolvers:
        - type: input
```

## Development Commands

```bash
# Install dev environment
uv sync

# Run unit tests
pytest tests/ -v

# Run with async test support
pytest tests/ -v --asyncio-mode=auto

# Lint and format
ruff check src/
ruff format src/

# Run CLI
sr2 configs/example.yaml
```

## Testing

- All tests in `tests/`
- Async tests use `pytest-asyncio`
- Always run tests before committing. New features = new tests.
- Test behavior through public APIs. Mock at system boundaries (LLM, stores).

## Design Philosophy

- **Event-driven, not sequential**: the pipeline reacts to events, not a fixed execution order. Resolvers and transformers fire when their subscribed events arrive.
- **Config-driven**: everything is YAML-declarable. No context management code in agent logic.
- **Protocol-based extensibility**: new resolvers, transformers, tool providers, token counters, and stores are addable without modifying engine code — just register via entry points.
- **Single package**: v2 is a clean single package. No sr2-runtime, sr2-bridge, sr2-pro, or multi-package workspace.
- **Async throughout**: all I/O (LLM, stores, provenance) is async.
- **Testability**: inject dependencies; mock at boundaries.

## What Is NOT In This Repo (v1 relics, removed)

The following v1 concepts **do not exist** on the `revamp` branch:
- Multi-package monorepo (sr2, sr2-runtime, sr2-bridge, sr2-pro)
- FastAPI runtime / HTTP server / Telegram / A2A protocol
- PostgreSQL/pgvector stores
- OTel / Prometheus exporters
- 50+ metrics collector
- Three-zone conversation (summarized/compacted/raw) as a first-class system
- Heartbeat scheduler
- LangGraph integration
- Bridge proxy
- sr2-pro / license enforcement
- sr2-agent / sr2-bridge / sr2-license CLI commands

## Preferences

- **BLUF first**: Lead with the outcome, then details
- **Testable before committed**: All code changes must pass tests
- **Explicit over implicit**: If behavior is unclear, ask
