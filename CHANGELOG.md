# Changelog

All notable changes to this project will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025-XX-XX

Initial public release.

### Added

**Pipeline Engine**
- Config-driven context compilation with layered architecture
- Three cache policies: `immutable`, `append_only`, `always_new`
- Token budget enforcement with per-layer trimming
- Prefix stability tracking for KV-cache optimization
- Interface router for per-trigger pipeline configs
- Config inheritance with deep merge (`defaults.yaml` -> `agent.yaml` -> `interfaces/*.yaml`)
- Config validation with actionable error messages

**Compaction**
- Five compaction strategies: `schema_and_sample`, `reference`, `result_summary`, `supersede`, `collapse`
- Three-zone conversation management: raw -> compacted -> summarized
- Configurable raw window and minimum content size
- Recovery hints for just-in-time re-fetching
- Async post-LLM compaction timing

**Summarization**
- LLM-powered structured summarization of old context
- Configurable preserve/discard categories (decisions, issues, preferences)
- Token threshold triggers
- Structured bullet and prose output formats

**Memory System**
- LLM-powered memory extraction from conversation turns
- Four memory types with stability scores: `identity`, `semi_stable`, `dynamic`, `ephemeral`
- Conflict detection (same key, different value)
- Conflict resolution strategies: `latest_wins_archive`, `latest_wins_discard`, `keep_both`
- Hybrid retrieval engine (keyword + semantic + recency/frequency boosts)
- Dimensional matching (`best_fit`, `exact`, `fallback_to_generic`)
- In-memory and PostgreSQL storage backends

**Tool Masking**
- Tool state machine with named states and conditional transitions
- Three masking strategies: `allowed_list`, `prefill`, `logit_mask`
- MCP tool manager with curated, discovery, and all-in-context strategies

**Degradation**
- Per-layer circuit breakers (configurable threshold and cooldown)
- Five-level degradation ladder: `full` -> `skip_summarization` -> `skip_intent` -> `raw_context` -> `system_prompt_only`

**Metrics & Observability**
- 28 pipeline metrics (cache hit rate, token usage, stage timing, degradation events)
- Prometheus text exposition exporter
- OpenTelemetry OTLP exporter
- Alert rule engine with configurable thresholds

**Content Resolvers**
- Built-in resolvers: `config`, `input`, `session`, `runtime`, `static_template`, `retrieval`, `mcp_resource`, `mcp_prompt`
- Pluggable resolver registry for custom sources

**Agent Runtime** (optional, in `src/runtime/`)
- Agentic LLM loop with LiteLLM
- Plugin system: HTTP/FastAPI, Telegram, timer, A2A
- Session management with three lifecycle policies: `persistent`, `ephemeral`, `rolling`
- MCP client with stdio/HTTP/SSE transport
- A2A protocol support (agent cards, discovery, message exchange)

**Infrastructure**
- Docker Compose development stack (agent, Ollama, Prometheus, Grafana, PostgreSQL)
- GitHub Actions CI (unit tests, linting, integration tests, schema validation)
- Ruff linting and formatting
- 688 tests across 71 test files
