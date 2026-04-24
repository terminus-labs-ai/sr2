# CLAUDE.md — SR2 Project Context

## What is SR2?

SR2 is a **context engineering library for AI agents** that manages the full lifecycle of what goes into an LLM's context window. It prevents context blowout (token budget overruns, KV-cache invalidation, agent degradation over long conversations) through a config-driven pipeline of caching, compaction, summarization, and memory management.

- **Library** (`src/sr2/`): ~5,900 LOC, pip-installable, minimal dependencies

## Architecture

### Three-Layer Pipeline
```
Layer 1: Core (immutable)     — System prompt, tools, static context
Layer 2: Memory (append-only) — Retrieved memories, semantic search
Layer 3: Conversation (append-only) — Session history, compacted/summarized
```

### Three-Zone Conversation Management
- **Raw Zone**: Recent N turns (verbatim)
- **Compacted Zone**: Older turns (tool outputs → samples, files → paths)
- **Summarized Zone**: Oldest content (structured LLM digest)

### Key Components
- **PipelineEngine** (`src/sr2/pipeline/engine.py`) — Core compilation, token budgets, cache tracking
- **InterfaceRouter** (`src/sr2/pipeline/router.py`) — Routes triggers to pipeline configs
- **ContentResolverRegistry** (`src/sr2/resolvers/registry.py`) — Pluggable content fetchers (15 built-in)
- **CompactionEngine** (`src/sr2/compaction/engine.py`) — 5 strategies: schema_and_sample, reference, result_summary, supersede, collapse; optional cost gate (`cost_gate.py`, `pricing.py`) blocks compaction when cache invalidation cost exceeds token savings
- **SummarizationEngine** (`src/sr2/summarization/engine.py`) — LLM-powered structured digests
- **Memory System** (`src/sr2/memory/`) — Extraction, hybrid retrieval (semantic+keyword), conflict resolution, pluggable store backends via registry
- **Extension Registries** (`memory/registry.py`, `metrics/registry.py`, `degradation/registry.py`) — Entry-point-based plugin discovery for stores, exporters, and degradation policies
- **ToolStateMachine** (`src/sr2/tools/state_machine.py`) — Named states with dynamic tool masking
- **CircuitBreaker** (`src/sr2/degradation/circuit_breaker.py`) — Per-layer graceful degradation

### Config Inheritance
```
configs/defaults.yaml → agent.yaml → interfaces/*.yaml
```
All behavior is config-driven (YAML). Zero hardcoded context logic.

## Fresh Environment Setup

- **Python 3.12+**, uv for package management (`uv sync --all-extras`)
- **Embeddings:** `OPENAI_API_KEY` for `text-embedding-3-small` (optional, needed for memory retrieval)
- **MCP servers:** Optional. Require Node.js 20+ for stdio transport.

## Build & Run

```bash
# Install (all packages + dev deps)
uv sync --all-extras

# Tests
pytest tests/ --ignore=tests/integration/ -v
pytest tests/ --ignore=tests/integration/ --cov=sr2 --cov-report=term-missing

# Integration tests (requires PostgreSQL with pgvector)
docker compose -f docker-compose.test.yaml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v

# Lint (enforced by pre-commit)
ruff check src/
ruff format src/
```

## Developer Guides

Read these before making changes:

- **[DEVELOPMENT.md](DEVELOPMENT.md)** — Architecture patterns, module boundaries, extensibility (how to add resolvers, rules, policies, plugins), SOLID principles, dependency rules, naming conventions, anti-patterns
- **[TESTING.md](TESTING.md)** — Test philosophy (behavior over implementation), mock patterns, async testing, fixture conventions, integration test setup, anti-patterns
- **[CODE_REVIEW.md](CODE_REVIEW.md)** — Review checklist: architecture, design patterns, KV-cache safety, memory scope isolation, test quality, config conventions

For the core/pro boundary definition (what goes in sr2 vs sr2-pro), see `~/git/sr2-pro/CLAUDE.md`.

## Code Conventions

- **Python 3.12+**, full type hints on all public signatures
- **Async-first**: All I/O is async/await with asyncio
- **Pydantic v2** for config models with `Field(description=...)`
- **Protocol-based** abstractions (not ABC) for pluggability
- **`@dataclass`** for value objects (CompiledContext, ResolvedContent, PipelineResult, etc.)
- **Naming**: CamelCase classes, snake_case functions, `_private` methods, suffixes like `_resolver`, `_engine`, `_policy`, `_manager`
- **Logging**: `logger = logging.getLogger(__name__)`, structured messages
- **Error handling**: Graceful degradation over crashes. Circuit breakers skip failing layers (except core). Optional content items can fail silently.
- **Token counting**: Character heuristic (`len(text) // 4`) to avoid tokenizer dependencies

## Testing

- **1,487 tests** across test files in `tests/`
- `pytest-asyncio` in auto mode (fixtures auto-marked)
- Unit tests cover: pipeline, compaction, summarization, memory, config, resolvers, tools
- Integration tests in `tests/integration/` require PostgreSQL + sr2-pro
- Premium code (PostgresMemoryStore, OTel/Prometheus exporters, AlertRuleEngine) lives in sr2-pro repo (`~/git/sr2-pro`)
- Pre-commit hooks: `ruff check` + `ruff format --check`

## Key Design Decisions

1. **Library-only** — SR2 core has no framework dependencies; agent runtime lives in sr2-spectre (separate repo)
2. **Config-driven everything** — All pipeline behavior in YAML with inheritance
3. **KV-cache aware** — Layers ordered by stability (immutable → append → dynamic) to maximize prefix reuse
4. **Post-LLM processing** — Memory extraction, compaction, summarization run async after LLM response
5. **Graceful degradation** — Per-layer circuit breakers; core layer never skipped
6. **Memory conflict resolution** — Detects conflicting memories with configurable resolution strategies (latest-wins-archive, keep-both-tagged)

## Directory Structure

```
packages/
  sr2/src/sr2/             # Core library (PyPI: sr2)
    config/                # Pydantic config models, YAML loader
    pipeline/              # Engine, router, conversation manager, prefix tracker
    resolvers/             # 15 content resolver implementations
    cache/                 # 7 cache policy classes
    compaction/            # Rule-based content compaction
    summarization/         # LLM-powered summarization
    memory/                # Extraction, retrieval, storage, conflicts
    tools/                 # Tool definitions, state machine, masking
    degradation/           # Circuit breaker
    metrics/               # Collector, Prometheus/OTel exporters, alerts
    normalization/         # Response processing (thinking blocks, etc.)

configs/                   # Example YAML configurations
  defaults.yaml            # Library defaults

examples/                  # Runnable examples
  01-04_*.py               # Core library demos (no API key needed)
  integrations/            # Framework examples (OpenAI Agents, LangChain, Pydantic AI, CrewAI)

benchmarks/                # Reproducible benchmarks
  token_savings.py         # No LLM needed, --json flag for CI
  coherence.py             # LLM-powered recall scoring
  cost.py                  # Real API cost measurement

docs/                      # MkDocs site (mkdocs serve to preview)
  index.md                 # Home page
  getting-started.md       # 5-minute quickstart
  configuration.md         # Auto-generated config reference
  guide-*.md               # Feature guides (memory, compaction, tools, etc.)
  pro.md                   # SR2 Pro features

tests/                     # Core library tests
  sr2/                     # Core library tests
  integration/             # PostgreSQL integration tests
```
