# CLAUDE.md — SR2 Project Context

## What is SR2?

SR2 is a **context engineering library for AI agents** that manages the full lifecycle of what goes into an LLM's context window. It prevents context blowout (token budget overruns, KV-cache invalidation, agent degradation over long conversations) through a config-driven pipeline of caching, compaction, summarization, and memory management.

- **Library** (`src/sr2/`): ~5,900 LOC, pip-installable, minimal dependencies
- **Runtime** (`src/runtime/`): ~5,000 LOC, optional agent runtime with HTTP, Telegram, MCP, A2A plugins
- **Bridge** (`src/bridge/`): Context optimization proxy for external LLM callers (Claude Code, LangChain, etc.)

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
- **ContentResolverRegistry** (`src/sr2/resolvers/registry.py`) — Pluggable content fetchers (13 built-in)
- **CompactionEngine** (`src/sr2/compaction/engine.py`) — 5 strategies: schema_and_sample, reference, result_summary, supersede, collapse
- **SummarizationEngine** (`src/sr2/summarization/engine.py`) — LLM-powered structured digests
- **Memory System** (`src/sr2/memory/`) — Extraction, hybrid retrieval (semantic+keyword), conflict resolution
- **ToolStateMachine** (`src/sr2/tools/state_machine.py`) — Named states with dynamic tool masking
- **CircuitBreaker** (`src/sr2/degradation/circuit_breaker.py`) — Per-layer graceful degradation
- **Heartbeat System** (`src/runtime/heartbeat/`) — Scheduled future agent callbacks with DB persistence, idempotent keys, context carry-over
- **BridgeEngine** (`src/bridge/engine.py`) — Context optimization proxy using CompactionEngine + ConversationManager + SummarizationEngine for external LLM callers

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
# Install
pip install -e ".[dev]"          # Development (includes ruff, pytest)
pip install -e ".[runtime]"      # With runtime plugins
pip install -e ".[all]"          # Everything

# Tests
pytest tests/ --ignore=tests/integration/ -v
pytest tests/ --ignore=tests/integration/ --cov=sr2 --cov-report=term-missing

# Integration tests (requires PostgreSQL with pgvector)
docker compose -f docker-compose.test.yml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v

# Lint (enforced by pre-commit)
ruff check src/
ruff format src/

# Run example agent
sr2-agent configs/agents/edi --http --port 8008

# Run bridge proxy (for Claude Code, LangChain, etc.)
sr2-bridge                           # zero-config
sr2-bridge bridge.yaml --port 9200   # custom config

# Single-shot mode (fire one message through an interface, print response, exit)
sr2-agent configs/agents/edi --single-shot task_runner "implement auth"
echo "long prompt" | sr2-agent configs/agents/edi --single-shot task_runner

# Docker (full stack: agent + Ollama + Prometheus + Grafana + PostgreSQL)
docker compose up                                    # AMD GPU
docker compose -f docker-compose.nvidia.yaml up      # NVIDIA GPU / CPU
```

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

- **844 tests** across 77 test files in `tests/`
- `pytest-asyncio` in auto mode (fixtures auto-marked)
- Unit tests cover: pipeline, compaction, summarization, memory, config, resolvers, tools
- Integration tests in `tests/integration/` require PostgreSQL
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
src/sr2/           # Core library
  config/          # Pydantic config models, YAML loader
  pipeline/        # Engine, router, conversation manager, prefix tracker
  resolvers/       # 13 content resolver implementations
  cache/           # 7 cache policy classes
  compaction/      # Rule-based content compaction
  summarization/   # LLM-powered summarization
  memory/          # Extraction, retrieval, storage, conflicts
  tools/           # Tool definitions, state machine, masking
  degradation/     # Circuit breaker
  metrics/         # Collector, Prometheus/OTel exporters, alerts
  normalization/   # Response processing (thinking blocks, etc.)
  a2a/             # Agent-to-Agent protocol

src/runtime/       # Optional agent runtime
  llm/             # LiteLLM wrapper, agentic loop, streaming
  mcp/             # MCP client, transports (stdio/HTTP/SSE)
  plugins/         # HTTP, Telegram, timer, A2A, single-shot plugins
  session/         # Session lifecycle management
  heartbeat/       # Scheduled future agent callbacks (model, store, tools, scanner)
  bridge/          # Context optimization proxy (sr2-bridge CLI, adapters, engine)

configs/           # Example YAML configurations
  defaults.yaml    # Library defaults
  agents/edi/      # Example agent config

tests/             # 844 tests
  integration/     # PostgreSQL integration tests
```
