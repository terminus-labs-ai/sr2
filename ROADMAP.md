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

- **HTTP interface parity** — route HTTP/OpenAI-compatible requests through the interface plugin system instead of bypassing `handle_user_message()`. Currently the agent-level `pipeline:` config is used as a fallback, but proper per-interface routing (session lifecycle enforcement, dedicated pipeline config) requires an explicit `api` interface entry.
- Docs site (GitHub Pages)
- One-command demo stack: `docker compose up` with example agent + Prometheus + Grafana and pre-built dashboards
- Demo video / GIF for the README

## v0.2.0 — Developer Experience

- **Agent creation wizard** — interactive CLI to scaffold a new agent config
- **Framework integrations** — examples and adapters for LangChain, CrewAI, AutoGen, and other popular frameworks
- **Config validation CLI** — `sr2 validate configs/agents/myagent/` with actionable suggestions

## Future

- Managed service / hosted offering
- Enterprise tier with team features
- Additional storage backends (Redis, SQLite)
- Visual config editor
- Agent marketplace / registry

---

Have ideas? [Open an issue](https://github.com/sr2labs/sr2/issues).
