# Changelog

All notable changes to SR2 v2.

## [Unreleased] — 2025-04-30

### Added — v2 scaffolding (clean slate)

- **Protocols** — 4 runtime-checkable extension point contracts:
  - `ContentProvider` — fetches content for a layer
  - `ContentReducer` — transforms/compresses content
  - `MemoryStore` — persistence backend for memories
  - `MetricExporter` — ships metrics to external systems
- **Core models** — `Memory`, `TurnResult`, `ToolCall`, `TokenUsage`
- **Memory enums** — `MemoryScope` (private/project/team/shared), `MemoryType` (identity/knowledge/preference/task/ephemeral), `MemoryStatus` (active/stale/archived/merged), `CachePolicy` (static/ephemeral/none)
- **Config system** — Pydantic models for full declarative YAML API:
  - `PipelineConfig`, `LayerConfig`, `MemoryConfig`, `CompactionConfig`, `SummarizationConfig`, `CircuitBreakerConfig`, `ProviderConfig`
  - YAML loader with `extends` inheritance
- **Pipeline engine** — Layer resolution with:
  - Cross-layer dependency topological sort (summarization scope refs)
  - Per-layer token budget enforcement
  - Total pipeline budget with priority-based shedding
  - Static cache layer support
- **Plugin registry** — Generic `PluginRegistry[T]`:
  - Lazy entry-point discovery
  - Instance caching
  - License gating via `require_license()`
- **Circuit breaker** — Per-provider failure tracking:
  - closed -> open -> half-open state machine
  - Configurable threshold and cooldown
  - Automatic recovery testing
- **Tokenization** — tiktoken-based `count_tokens()` and `truncate_to_tokens()`
- **SR2 Facade** — Two-entry-point public API:
  - `process(config, inputs) -> CompiledContext`
  - `post_process(turn_result) -> PostProcessResult`
- **Error hierarchy** — `SR2Error` base with typed subclasses
- **Result types** — `CompiledContext`, `LayerResult`, `PipelineMetrics`, `PostProcessResult`, `MaintenanceAction`
- **Documentation** — README.md, QUICKSTART.md, CONTRIBUTING.md

### Design principles

- SOLID enforced at every level
- OCP: all capabilities extendable via protocols + entry points
- DRY: shared types defined once, generic registry
- Hard facade boundary: harness only touches `process()` and `post_process()`
- Declarative-first: YAML config is the API
