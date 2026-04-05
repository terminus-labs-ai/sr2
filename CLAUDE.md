# CLAUDE.md — SR2 Project Context

## What is SR2?

SR2 is a **context engineering library for AI agents** that manages the full lifecycle of what goes into an LLM's context window. It prevents context blowout (token budget overruns, KV-cache invalidation, agent degradation over long conversations) through a config-driven pipeline of caching, compaction, summarization, and memory management.

- **Library** (`packages/sr2/`): ~5,900 LOC, pip-installable, minimal dependencies
- **Runtime** (`packages/sr2-runtime/`): ~5,000 LOC, optional agent runtime with HTTP, Telegram, MCP, A2A plugins
- **Bridge** (`packages/sr2-bridge/`): Context optimization proxy for external LLM callers (Claude Code, LangChain, etc.)

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
- **PipelineEngine** (`packages/sr2/src/sr2/pipeline/engine.py`) — Core compilation, token budgets, cache tracking
- **InterfaceRouter** (`packages/sr2/src/sr2/pipeline/router.py`) — Routes triggers to pipeline configs
- **ContentResolverRegistry** (`packages/sr2/src/sr2/resolvers/registry.py`) — Pluggable content fetchers (15 built-in)
- **CompactionEngine** (`packages/sr2/src/sr2/compaction/engine.py`) — 5 strategies: schema_and_sample, reference, result_summary, supersede, collapse; optional cost gate (`cost_gate.py`, `pricing.py`) blocks compaction when cache invalidation cost exceeds token savings
- **SummarizationEngine** (`packages/sr2/src/sr2/summarization/engine.py`) — LLM-powered structured digests
- **Memory System** (`packages/sr2/src/sr2/memory/`) — Extraction, hybrid retrieval (semantic+keyword), conflict resolution, pluggable store backends via registry
- **Extension Registries** (`memory/registry.py`, `metrics/registry.py`, `degradation/registry.py`) — Entry-point-based plugin discovery for stores, exporters, and degradation policies
- **ToolStateMachine** (`packages/sr2/src/sr2/tools/state_machine.py`) — Named states with dynamic tool masking
- **CircuitBreaker** (`packages/sr2/src/sr2/degradation/circuit_breaker.py`) — Per-layer graceful degradation
- **Heartbeat System** (`packages/sr2-runtime/src/sr2_runtime/heartbeat/`) — Scheduled future agent callbacks with DB persistence, idempotent keys, context carry-over
- **BridgeEngine** (`packages/sr2-bridge/src/sr2_bridge/engine.py`) — Context optimization proxy using CompactionEngine + ConversationManager + SummarizationEngine for external LLM callers

### Config Inheritance
```
configs/defaults.yaml → agent.yaml → interfaces/*.yaml
```
All behavior is config-driven (YAML). Zero hardcoded context logic.

## Fresh Environment Setup

For setting up EDI on a new machine, see **SETUP.md**. Quick summary:
- **Docker (AMD GPU):** `docker compose up` then `docker exec ollama ollama pull llama3.1:8b llama3.2:3b`
- **Docker (NVIDIA/CPU):** `docker compose -f docker-compose.nvidia.yaml up` then pull models
- **Local dev:** Python 3.12+, PostgreSQL with pgvector, Ollama with `llama3.1:8b` + `llama3.2:3b`
- **Embeddings:** `OPENAI_API_KEY` in `.env` for `text-embedding-3-small` (optional, needed for memory retrieval)
- **MCP servers:** Optional. Require Node.js 20+ for stdio transport.

## Build & Run

```bash
# Install (all packages + dev deps)
uv sync --all-extras

# Tests
pytest tests/ --ignore=tests/integration/ -v
pytest tests/ --ignore=tests/integration/ --cov=sr2 --cov=sr2_bridge --cov=sr2_runtime --cov-report=term-missing

# Integration tests (requires PostgreSQL with pgvector)
docker compose -f docker-compose.test.yml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v

# Lint (enforced by pre-commit)
ruff check packages/
ruff format packages/

# Run example agent
sr2-agent configs/agents/edi --http --port 8008

# Run bridge proxy (for Claude Code, LangChain, etc.)
sr2-bridge                           # zero-config
sr2-bridge bridge.yaml --port 9200   # custom config

# Single-shot mode (fire one message through an interface, print response, exit)
sr2-agent configs/agents/edi --single-shot task_runner "implement auth"
echo "long prompt" | sr2-agent configs/agents/edi --single-shot task_runner

# Run with Claude Code provider (uses claude CLI for tool execution)
sr2-agent configs/agents/claude-code --http --port 8741

# Run with Claude Code + proxy (CLIProxyAPI, etc.) for fast model
PROXY_URL=http://localhost:9090/v1 sr2-agent configs/agents/claude-code-proxy --http --port 8741

# Docker (full stack: agent + Ollama + Prometheus + Grafana + PostgreSQL)
docker compose up                                    # AMD GPU
docker compose -f docker-compose.nvidia.yaml up      # NVIDIA GPU / CPU
docker compose -f docker-compose.claude-code.yaml up # Claude Code + proxy
```

## Claude Code Provider

SR2-runtime can use Claude Code CLI (`claude -p`) as the main LLM
provider. Claude Code handles tool execution (Bash, Edit, Read, MCPs, etc.)
internally while SR2 retains ownership of context engineering (compaction,
summarization, memory) and session management.

```yaml
# In agent.yaml:
runtime:
  llm:
    claude_code:
      enabled: true
      path: claude
      allowed_tools: [Read, Glob, Grep, Agent, WebSearch, WebFetch]
```

**Note:** When using Claude Code, SR2's registered MCP tools are not available
to Claude Code (which has its own MCP config). Configure MCP servers in Claude
Code's own settings if needed.

## OpenAI-Compatible Proxy

Any OpenAI-compatible proxy (CLIProxyAPI, LiteLLM proxy, etc.) can serve SR2's
internal LLM tasks by setting `api_base` on model configs. Useful for routing
summarization, memory extraction, or embedding calls through a local proxy.

```yaml
# In agent.yaml:
runtime:
  llm:
    fast_model:
      name: claude-haiku-4-5-20251001
      api_base: http://localhost:9090/v1  # Your proxy endpoint
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

1. **Library vs Runtime separation** — SR2 core has no framework dependencies; runtime is optional
2. **Config-driven everything** — All pipeline behavior in YAML with inheritance
3. **KV-cache aware** — Layers ordered by stability (immutable → append → dynamic) to maximize prefix reuse
4. **Post-LLM processing** — Memory extraction, compaction, summarization run async after LLM response
5. **Graceful degradation** — Per-layer circuit breakers; core layer never skipped
6. **Memory conflict resolution** — Detects conflicting memories with configurable resolution strategies (latest-wins-archive, keep-both-tagged)
7. **Dynamic heartbeats** — Agents can schedule future callbacks to themselves via `schedule_heartbeat`/`cancel_heartbeat` tools, with context carry-over and idempotent keys
8. **Bridge proxy** — Reverse proxy mode applies SR2 context optimization to external callers (Claude Code, LangChain) without requiring modifications to the caller

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
    a2a/                   # Agent-to-Agent protocol

  sr2-runtime/src/sr2_runtime/  # Agent runtime (PyPI: sr2-runtime)
    llm/                   # LiteLLM wrapper, agentic loop, streaming
    mcp/                   # MCP client, transports (stdio/HTTP/SSE)
    plugins/               # HTTP, Telegram, timer, A2A, single-shot plugins
    session/               # Session lifecycle management
    heartbeat/             # Scheduled future agent callbacks

  sr2-bridge/src/sr2_bridge/    # Bridge proxy (PyPI: sr2-bridge)

configs/                   # Example YAML configurations
  defaults.yaml            # Library defaults
  agents/edi/              # Example agent config

examples/                  # Runnable examples
  01-04_*.py               # Core library demos (no API key needed)
  runtime/                 # Agent runtime demos
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

tests/                     # 1,487 tests
  sr2/                     # Core library tests
  runtime/                 # Runtime tests
  bridge/                  # Bridge tests
  integration/             # PostgreSQL integration tests
```
