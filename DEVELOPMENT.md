# SR2 Development Guide

How to write code that belongs in this project. Read this before contributing — it covers architecture patterns, module boundaries, extensibility, and anti-patterns.

For setup instructions, see [CONTRIBUTING.md](CONTRIBUTING.md). For testing guidelines, see [TESTING.md](TESTING.md). For review checklists, see [CODE_REVIEW.md](CODE_REVIEW.md).

---

## Module Boundaries

### Package Responsibilities

| Package | Responsibility | Depends On |
|---------|---------------|------------|
| `sr2` (core) | Context compilation library — pipeline, resolvers, memory, compaction, summarization, degradation, tools, metrics, config | Nothing (standalone) |
| `sr2-relay` *(separate repo)* | Context-aware LLM API — library + HTTP server, wraps sr2 | `sr2` |
| `sr2-spectre` *(separate repo)* | Full agent runtime — tools, plugins, sessions, heartbeats | `sr2-relay` |

### Dependency Direction

```
sr2-spectre → sr2-relay → sr2 (core)
```

**Rules:**
- Core (`sr2`) must **never** import from `sr2-relay` or `sr2-spectre`.
- Out-of-tree extensions (premium, third-party) integrate via **entry points** — never direct imports into core.

### Where Does This Code Go?

| If it... | Put it in... |
|----------|-------------|
| Compiles, transforms, or manages context | `sr2` (core) |
| Defines a protocol for extensibility | `sr2` (core) |
| Makes LLM calls, manages SR2 sessions | `sr2-relay` |
| Runs an agent loop, executes tools | `sr2-spectre` |
| Exposes HTTP/WebSocket/CLI interfaces | `sr2-spectre` (plugins) |
| Implements a production-scale backend | Out-of-tree extension via entry points |
| Adds observability exporters | Out-of-tree extension via entry points |

### Cross-Package Extension via Entry Points

Core provides registries that discover extensions at runtime via `importlib.metadata` entry points. This is how out-of-tree packages add memory stores, metric exporters, or degradation policies without modifying core.

**Entry point groups:**
- `sr2.stores` — Memory store backends (e.g., PostgreSQL)
- `sr2.exporters` — Metric exporters (e.g., OpenTelemetry, Prometheus)
- `sr2.alerts` — Alert rule engines

**How it works:**
1. Extension package declares entry points in `pyproject.toml`
2. Core's registry calls `importlib.metadata.entry_points(group="sr2.stores")` on first miss
3. Extension's registration function is called, which registers the implementation
4. Core uses the registered implementation via `get_store("postgres")`, etc.

**Registration function pattern:**
```python
# In the extension package
def register_stores():
    from sr2.memory.registry import register_store
    from .postgres import PostgresMemoryStore
    register_store("postgres", PostgresMemoryStore)
```

---

## Architecture Invariants

These must never be violated. If a change breaks one of these, it's a bug.

### 1. Config Drives Behavior
All pipeline behavior is determined by YAML configuration. Agent code must never contain context management logic — that's SR2's job.

```yaml
# Correct: behavior is in config
compaction:
  enabled: true
  raw_window: 5
  strategy: rule_based
```

```python
# Wrong: hardcoded behavior in agent code
if len(messages) > 10:
    messages = compact(messages)  # NO — this belongs in SR2 config
```

### 2. Resolvers Are Stateless
Resolvers receive everything they need via `ResolverContext`. They must not maintain internal state between calls, cache results internally, or depend on call order.

```python
# Correct: stateless resolver
class ConfigResolver:
    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        value = context.agent_config[key]
        return ResolvedContent(key=key, content=str(value), tokens=estimate_tokens(str(value)))
```

### 3. All I/O Is Async
Every function that performs I/O (LLM calls, database, HTTP, filesystem) must be `async`. No blocking I/O anywhere in the codebase.

### 4. Protocols Over Inheritance
Extensibility uses `Protocol` classes, not abstract base classes. This enables duck typing and avoids tight coupling.

### 5. Pipeline Is the Only Path
All context must flow through the pipeline engine. No side-channel injection of content into LLM calls.

### 6. Layers Are Ordered and Deterministic
Layer resolution order must be deterministic for KV-cache prefix stability. Layers are resolved in config order, serialized deterministically, and hashed for cache comparison.

### 7. Budget Is Always Enforced
The pipeline engine must never produce output exceeding `token_budget`. Overflow handling (compaction -> summarization -> truncation) is mandatory.

### 8. Scope Isolation
Memories are scoped (private, project, shared). A resolver must never return memories outside its configured `allowed_read` scopes. Writes must respect `allowed_write`.

---

## Extensibility Patterns

### Adding a New Resolver

1. Create `packages/sr2/src/sr2/resolvers/my_resolver.py`
2. Implement the `ContentResolver` protocol:

```python
from sr2.resolvers.registry import ContentResolver, ResolverContext, ResolvedContent, estimate_tokens

class MyResolver:
    """Resolves content from [describe source]."""

    async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
        # config contains per-item settings from the YAML content item
        # context contains agent_config, trigger_input, session_id, etc.
        content = ...  # fetch or compute content
        return ResolvedContent(
            key=key,
            content=content,
            tokens=estimate_tokens(content),
            metadata={"source": "my_resolver"},
        )
```

3. Register in the resolver registry (done at initialization time in `sr2.py` or by the runtime)
4. Add tests in `tests/sr2/test_resolvers/test_my_resolver.py`
5. Reference in pipeline config:

```yaml
layers:
  - name: my_layer
    contents:
      - key: my_content
        source: my_resolver  # matches registration name
```

### Adding a New Compaction Rule

1. Create a rule class in `packages/sr2/src/sr2/compaction/rules.py` (or a new file)
2. Implement the rule interface:

```python
class MyRule:
    """Compacts [describe what content type]."""

    def matches(self, turn: ConversationTurn) -> bool:
        """Return True if this rule applies to the turn."""
        return turn.content_type == "my_type"

    def compact(self, turn: ConversationTurn) -> str:
        """Return the compacted version of the turn content."""
        return f"-> {summarize(turn.content)}"
```

3. Register via compaction config:

```yaml
compaction:
  rules:
    - type: my_type
      strategy: my_rule
```

### Adding a New Cache Policy

1. Implement the `CachePolicy` protocol in `packages/sr2/src/sr2/cache/policies.py`:

```python
class MyPolicy:
    def should_recompute(self, layer_name: str, current: PipelineState, previous: PipelineState | None) -> bool:
        """Return True if the layer should be recomputed."""
        ...
```

2. Register in the cache policy registry

### Adding a New Memory Store

1. Implement the `MemoryStore` protocol from `sr2.memory.store`
2. Register via entry points in your package's `pyproject.toml`:

```toml
[project.entry-points."sr2.stores"]
my_store = "my_package.store:register_stores"
```

3. All methods must be async. Required methods: `save`, `get`, `get_by_key`, `search_vector`, `search_keyword`, `delete`, `archive`, `count`, `list_scope_refs`

### Adding a New Agent Plugin

Agent plugins (HTTP, Telegram, timer, etc.) live in `sr2-spectre` (separate repo). See `sr2-spectre` for the plugin protocol and registry.

---

## SOLID Principles in SR2

### Single Responsibility (SRP)

Each resolver does one thing: fetch content from one source. Each compaction rule handles one content type. Each cache policy answers one question (`should_recompute?`).

**If a class needs a conjunction to describe what it does ("fetches memories AND compacts them"), split it.**

### Open/Closed (OCP)

New features are added by implementing protocols and registering — not by modifying existing engine code.

- New content source? Add a resolver. Don't modify `PipelineEngine`.
- New memory backend? Implement `MemoryStore`. Don't modify retrieval logic.
- New metric export? Implement an exporter. Don't modify `MetricCollector`.

### Liskov Substitution (LSP)

All `MemoryStore` implementations must behave identically for the same inputs. `InMemoryMemoryStore`, `SQLiteMemoryStore`, and `PostgresMemoryStore` are interchangeable — the pipeline doesn't know or care which one is active.

### Interface Segregation (ISP)

The `ContentResolver` protocol has exactly one method: `resolve()`. Not `resolve_and_cache()` or `resolve_with_metrics()`. Caching and metrics are handled by the pipeline engine around the resolver, not by the resolver itself.

### Dependency Inversion (DIP)

The pipeline engine depends on the `ContentResolver` protocol, not on `ConfigResolver` or `SessionResolver` concretely. Resolvers are injected via the registry at initialization time.

```python
# PipelineEngine depends on abstractions
engine = PipelineEngine(resolver_registry, cache_registry)

# Concrete resolvers registered externally
registry.register("config", ConfigResolver())
registry.register("session", SessionResolver())
```

---

## Error Handling

### When to Raise

- **Config validation errors**: Raise immediately at load time. Invalid config should never reach the pipeline.
- **Missing required content**: If a resolver can't produce content for a non-optional item, raise `KeyError`.
- **Protocol violations**: If a store method receives invalid arguments, raise `ValueError`.

### When to Degrade

- **Resolver failures during pipeline**: The circuit breaker catches exceptions, records failures, and skips the layer after threshold. Core layers (system prompt) are exempt — their failures always propagate.
- **Memory extraction failures**: Log and continue. Extraction is post-LLM and must never block the response.
- **Compaction/summarization failures**: Log and continue. The raw conversation is always available as fallback.

### Never Do This

```python
# Wrong: silently swallowing errors
try:
    result = await resolver.resolve(...)
except Exception:
    pass  # NO — at minimum log, ideally let circuit breaker handle it

# Wrong: catching too broadly
try:
    config = load_config(path)
except Exception:
    config = {}  # NO — config errors should fail fast
```

---

## Naming Conventions

### Modules and Files

| Type | Convention | Example |
|------|-----------|---------|
| Module directory | `snake_case/` | `memory/`, `compaction/` |
| Module file | `snake_case.py` | `circuit_breaker.py`, `cost_gate.py` |
| Test file | `test_<module>.py` | `test_engine.py`, `test_store.py` |
| Test directory | `test_<module>/` | `test_resolvers/`, `test_memory/` |

### Classes and Functions

| Type | Convention | Example |
|------|-----------|---------|
| Protocol | `PascalCase` (noun) | `ContentResolver`, `MemoryStore`, `CachePolicy` |
| Implementation | `PascalCase` (descriptive) | `ConfigResolver`, `SQLiteMemoryStore`, `ImmutablePolicy` |
| Engine/manager | `PascalCase` + `Engine`/`Manager` | `PipelineEngine`, `ConversationManager` |
| Data model | `PascalCase` (noun) | `Memory`, `CompiledContext`, `PipelineResult` |
| Config model | `PascalCase` + `Config` | `PipelineConfig`, `CompactionConfig` |
| Async function | `snake_case` | `async def resolve()`, `async def extract()` |
| Helper function | `_snake_case` (prefixed) | `def _serialize_layer()` |

### Config Keys

YAML config keys use `snake_case`:
```yaml
token_budget: 32000
kv_cache:
  compaction_timing: post_llm_async
  memory_refresh_interval: 10
```

### Metric Names

Metrics use `sr2_` prefix with `snake_case`:
```python
"sr2_cache_hit_rate"
"sr2_pipeline_total_tokens"
"sr2_retrieval_latency_ms"
```

---

## Config-Driven Design

### Config Inheritance

Three levels of config, merged deepest-first:

```
defaults.yaml        # Global defaults (token_budget, standard rules)
    ↓ overrides
agent.yaml           # Agent-specific (system prompt, tools, memory)
    ↓ overrides
interfaces/*.yaml    # Interface-specific (heartbeat uses less budget)
```

Use `extends:` to reference parent configs:
```yaml
# interfaces/heartbeat.yaml
extends: agent        # inherits from agent.yaml
token_budget: 3000    # override just what's different
compaction:
  enabled: false
```

### Adding New Config Fields

1. Add field to the appropriate Pydantic model in `config/models.py`:

```python
class CompactionConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable conversation compaction")
    raw_window: int = Field(default=5, description="Number of recent turns kept uncompacted")
    my_new_field: int = Field(default=10, description="Clear description of what this controls")
```

2. Always provide a default value
3. Always provide a `description` (used for auto-generated docs)
4. Regenerate config docs: `sr2-config-docs --format md > docs/configuration.md`
5. Add validation in `config/validation.py` if the field has constraints

---

## KV-Cache Awareness

SR2's pipeline is designed to maximize KV-cache reuse across LLM invocations. Changes that break prefix stability hurt performance.

### What Preserves Prefix Stability

- Immutable layers at the top of the layer stack (system prompt, tools)
- Append-only layers that only add content, never modify existing content
- Deterministic serialization (same input -> same output string)
- Deferred memory touch (access counts updated after LLM call, not before)

### What Breaks Prefix Stability

- Changing content in early layers between calls (forces full recompute)
- Non-deterministic serialization (timestamps, random IDs in layer content)
- Eager memory touch (updates retrieval scores before LLM call, changing retrieval results mid-prefix)
- Resolver side effects that modify shared state

### Rules for Resolvers

- Resolvers in Layer 1 (core) must produce **identical output** across calls within a session (use `immutable` cache policy)
- Resolvers in Layer 2 (memory) should produce **stable output** unless retrieval scores change (use `refresh_on_topic_shift`)
- Resolvers in Layer 3 (conversation) change every turn — this is expected (use `append_only`)

---

## Anti-Patterns

### Don't Reach Across Package Boundaries

```python
# Wrong: runtime importing core internals
from sr2.pipeline.engine import _serialize_layer  # NO — private function

# Right: use public API
from sr2.pipeline.engine import PipelineEngine
```

### Don't Put State in Resolvers

```python
# Wrong: resolver caches results internally
class BadResolver:
    def __init__(self):
        self._cache = {}  # NO — state belongs in the pipeline/cache layer

    async def resolve(self, key, config, context):
        if key in self._cache:
            return self._cache[key]
        ...
```

### Don't Hardcode Behavior

```python
# Wrong: hardcoded compaction window
if len(turns) > 5:  # NO — this should be config.compaction.raw_window
    compact(turns[:-5])

# Right: read from config
if len(turns) > config.compaction.raw_window:
    compact(turns[:-config.compaction.raw_window])
```

### Don't Bypass the Pipeline

```python
# Wrong: injecting content directly into LLM messages
messages.append({"role": "system", "content": extra_context})  # NO

# Right: add a resolver and configure it in a layer
# Then the pipeline handles budget, caching, degradation
```

### Don't Mock Internal Classes in Tests

```python
# Wrong: mocking ConversationManager internals
with patch.object(ConversationManager, '_compact_zone'):  # NO

# Right: test through public API
manager = ConversationManager(compaction_engine=engine, raw_window=3)
for i in range(10):
    manager.add_turn(make_turn(i))
result = manager.run_compaction()
assert len(manager.zones().compacted) > 0
```

### Don't Swallow Memory Scope

```python
# Wrong: ignoring scope on retrieval
results = await store.search_keyword(query)  # NO — missing scope filter

# Right: always pass scope
results = await store.search_keyword(query, scope_filter=scope_config.allowed_read)
```
