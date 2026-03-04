# Roadmap

A high-level view of where SR2 is headed. Priorities shift based on community feedback — if something here matters to you, open an issue or start a discussion.

## v0.1.0 — Initial Release

The foundation: a config-driven context engineering library for AI agents.

- Pipeline engine with layered context compilation and KV-cache optimization
- Five compaction strategies with recovery hints
- LLM-powered summarization with preserve/discard categories
- Memory system: extraction, conflict resolution, hybrid retrieval
- Tool state machine with three masking strategies
- Per-layer circuit breakers and graceful degradation
- Prometheus and OpenTelemetry metrics export
- Agent runtime with HTTP, Telegram, timer, and A2A plugins
- MCP client with stdio/HTTP/SSE transport

## v0.1.x — Polish

### Done

- **CI pipeline hardening** — fix ruff lint failures, remove dead imports, complete missing wiring ✅
- **StreamContentConfig documentation** — add to auto-generated config reference ✅
- **Observability overhaul** — 14 new metrics across 5 categories (cache, retrieval, compaction, summarization, pipeline), Grafana dashboard creation and fixes ✅
- **Session metrics accuracy** — use actual session state for turn count and duration metrics ✅
- **Response normalization module** — strip thinking blocks, markdown fences, JSON extraction, multi-model compatibility ✅
- **Memory improvements** — key schema enforcement from config, empty embedding fix, confidence scoring, pgvector format, score clamping, retrieval persistence ✅
- **Telegram agent commands** — user-facing commands through Telegram interface ✅
- **Benchmarks** — performance benchmark suite ✅
- **Bug fix sweep** — fixes across compaction (log format, content truncation, input mutation), summarization (enabled flag, preserve_recent_turns, token budget), circuit breaker (config values for threshold/cooldown), pipeline (consistency issues), cache (warn on unknown policies, remove dead code), resolvers (standardize token counting, fix state mutation), normalization (multi-model compat) ✅
- **Docker Compose demo stack** — `docker compose up` with example agent + Prometheus + Grafana + Postgres + Open-WebUI + Ollama ✅

### Remaining

- **HTTP interface parity** — route HTTP/OpenAI-compatible requests through the interface plugin system instead of bypassing `handle_user_message()`. Currently the agent-level `pipeline:` config is used as a fallback, but proper per-interface routing (session lifecycle enforcement, dedicated pipeline config) requires an explicit `api` interface entry.
- Docs site (GitHub Pages)
- Demo video / GIF for the README

## v0.2.0 — Evaluation & Quality

Know whether your context engineering is working — and prove it.

- **Eval harness** — replay conversation transcripts against a pipeline config, output a quality scorecard (token efficiency, cache hit rate, compaction ratio, memory recall accuracy)
- **Regression detection** — compare scorecards across config versions, flag regressions before they ship
- **Transcript fixtures** — record and replay real sessions as repeatable test cases
- **Config diff** — `sr2 diff config-a/ config-b/` with human-readable impact summary

### Premium

- Hosted eval runs with historical comparison
- A/B testing between pipeline configs with statistical significance
- Regression alerts (CI integration, webhook notifications)

## v0.3.0 — Cost Intelligence

Turn SR2's instrumentation into a cost story.

- **Per-session cost tracking** — attribute token spend to layers, resolvers, and cache policies
- **Cost-per-layer breakdown** — new metrics and Grafana panels showing where budget goes
- **Budget alerts** — configurable thresholds that warn or degrade gracefully when cost targets are exceeded

### Premium

- Cost dashboards with trend analysis and forecasting
- Optimization recommendations (e.g., "switching layer X to immutable cache saves ~18%")
- Team-level usage quotas and chargeback reports

## v0.4.0 — Security & Compliance

Enterprise table stakes for running agents with real user data.

- **PII detection and redaction** — configurable pipeline hook that scans content before it reaches the LLM
- **Audit logging** — structured log of messages, memory operations, tool calls, and config changes
- **Data retention policies** — auto-expire memories, session TTLs, right-to-forget support
- **Content filtering hooks** — pre/post LLM filter points for custom moderation

### Premium

- Managed audit log storage with search and export
- Compliance reporting (data residency, retention proof)
- SSO / RBAC for team access control

## v0.5.0 — Config Management

Production-grade config lifecycle for teams.

- **Config validation CLI** — `sr2 validate configs/agents/myagent/` with actionable suggestions
- **Config versioning** — version history and rollback for agent configurations
- **Agent creation wizard** — interactive CLI to scaffold a new agent config

### Premium

- Hosted config registry with deploy-to-environment workflows
- Canary deployments (route a percentage of sessions to a new config)
- Config drift detection across environments

## Future

- Additional storage backends (Redis, SQLite)
- Visual config editor
- Agent marketplace / registry

---

Have ideas? [Open an issue](https://github.com/terminus-labs-ai/sr2/issues).
