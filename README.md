# SR2

**Context engineering for AI agents.** Manages the full lifecycle of what goes into your LLM's context window — so you stop losing cache hits, blowing token budgets, and shipping agents that forget what happened 10 turns ago.

---

## The Problem

Your agent framework handles tool calling and orchestration. Nobody's managing the context window.

Every turn, your agent stuffs more conversation history, tool results, retrieved memories, and system prompts into a growing blob of text. Eventually you hit the token limit and start truncating from the top — destroying your KV-cache prefix, evicting the system prompt, and losing critical context. Your agent gets worse the longer the conversation goes.

SR2 treats the context window as a managed resource. It compiles context through a config-driven pipeline with caching, compaction, summarization, and graceful degradation — so your agent stays coherent at turn 200 the same way it was at turn 2.

## How It Works

```
  Trigger arrives (user message, heartbeat, A2A call)
         │
         ▼
  ┌─────────────────┐
  │ InterfaceRouter  │  ← picks the right pipeline config
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  PipelineEngine  │  ← resolves layers, checks cache, enforces budget
  │                  │
  │  Layer 1: core   │  immutable  (system prompt, tools)
  │  Layer 2: memory │  append     (retrieved memories, summaries)
  │  Layer 3: conv   │  append     (session history, user input)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ CompiledContext  │  ← content string + token count + cache metadata
  └────────┬────────┘
           │
           ▼
      Your LLM call
           │
           ▼
  ┌─────────────────┐
  │ PostLLMProcessor │  ← async: extract memories, compact, summarize
  └─────────────────┘
```

Layers are ordered most-stable to least-stable. The system prompt never changes, so it's always the KV-cache prefix. Conversation history changes every turn, so it's last. This means your LLM provider caches the expensive prefix and you only pay to process new tokens.

## Install

```bash
git clone https://github.com/terminus-labs-ai/sr2.git
cd sr2

# Install everything (recommended for development)
uv sync --all-extras

# Or install individual packages
pip install -e packages/sr2                # Core library only
pip install -e packages/sr2-runtime        # Agent runtime (depends on sr2)
pip install -e packages/sr2-bridge         # Bridge proxy (depends on sr2)
```

## Quick Start

```python
import asyncio
from sr2.config.models import PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.cache.policies import create_default_cache_registry


async def main():
    # 1. Register resolvers (they fetch content for each layer item)
    registry = ContentResolverRegistry()
    registry.register("config", ConfigResolver())
    registry.register("input", InputResolver())

    # 2. Create the engine
    engine = PipelineEngine(registry, create_default_cache_registry())

    # 3. Define your context layout
    config = PipelineConfig(
        token_budget=8000,
        layers=[
            {
                "name": "core",
                "cache_policy": "immutable",
                "contents": [
                    {"key": "system_prompt", "source": "config"},
                ],
            },
            {
                "name": "conversation",
                "cache_policy": "append_only",
                "contents": [
                    {"key": "user_input", "source": "input"},
                ],
            },
        ],
    )

    # 4. Compile context
    result = await engine.compile(
        config,
        ResolverContext(
            agent_config={"system_prompt": "You are a helpful assistant."},
            trigger_input="What's the weather like?",
        ),
    )

    print(result.content)
    # -> "You are a helpful assistant.\n\nWhat's the weather like?"
    print(f"Tokens: {result.tokens}")
    print(f"Pipeline status: {result.pipeline_result.overall_status}")


asyncio.run(main())
```

That's the minimal case — two layers, no caching tricks. In production you'd add memory retrieval, conversation history with compaction, summarization triggers, and per-interface configs. See the [configuration reference](docs/configuration.md) for the full surface area.

## Config-Driven Context Layout

Everything is YAML. No context management code in your agent logic.

```yaml
# configs/defaults.yaml — library defaults, all fields have defaults
token_budget: 32000
pre_rot_threshold: 0.25

compaction:
  enabled: true
  raw_window: 5
  rules:
    - type: tool_output
      strategy: schema_and_sample
      max_compacted_tokens: 80
      recovery_hint: true
    - type: file_content
      strategy: reference

summarization:
  enabled: true
  trigger: token_threshold
  threshold: 0.75
  preserve:
    - decisions_and_reasoning
    - unresolved_issues
    - user_preferences_expressed

layers:
  - name: core
    cache_policy: immutable          # never recompute mid-session
    contents:
      - key: system_prompt
        source: config
      - key: tools
        source: config

  - name: memory
    cache_policy: append_only
    contents:
      - key: retrieved_context
        source: retrieval
        optional: true

  - name: conversation
    cache_policy: append_only
    contents:
      - key: session_history
        source: session
      - key: user_input
        source: input
```

Config inheritance: `defaults.yaml` → `agent.yaml` → `interfaces/user_message.yaml`. Deep merge, more specific wins.

## Key Features

**Three-zone conversation management.** Raw turns (verbatim recent history) → compacted turns (tool outputs replaced with references) → summarized zone (structured summary of oldest context). Your agent never loses important decisions even at turn 200.

**KV-cache optimization.** Layers are ordered for prefix stability. Immutable content stays at the top so your LLM provider's KV-cache can reuse it across turns. The prefix tracker measures actual cache efficiency so you can see if your layout is working.

**Compaction rules.** Tool outputs get replaced with "→ 47 lines. Sample: ..." plus a recovery hint. File contents become path references. Code execution results become exit code + first 3 lines. The agent can re-fetch anything it needs.

**Automatic summarization.** When the compacted zone exceeds a threshold, an LLM call produces a structured summary preserving decisions, unresolved issues, and user preferences. Routine confirmations and dead-end explorations are discarded.

**Graceful degradation.** Per-layer circuit breakers. If retrieval fails 3 times in a row, the breaker opens and that layer is skipped — the agent keeps running with reduced context rather than crashing. The core layer (system prompt) is never skipped.

**Per-interface pipeline configs.** A Telegram chat gets 48k tokens with full compaction/summarization. A heartbeat gets 3k tokens with everything disabled. An A2A call gets 8k tokens, stateless. Same agent, different context strategies per trigger type.

**Dynamic heartbeats.** Agents can schedule future callbacks to themselves via `schedule_heartbeat` / `cancel_heartbeat` tools. Supports idempotent keys, context carry-over from the original session, and DB persistence. Use it for async monitoring, retries, or timed reminders. See [Heartbeat Guide](docs/guide-heartbeats.md).

**Memory system.** Extract structured memories from conversations, detect conflicts between new and existing memories, resolve them with configurable strategies (latest-wins-archive, keep-both-tagged), and retrieve with hybrid semantic + keyword search. Automatic scope detection assigns memories to the correct project/team context without manual configuration. Supports PostgreSQL and SQLite backends.

**Intent detection.** Classify user messages to detect topic shifts, enabling context-aware memory refresh and selective summarization. Foundation for LLM-based topic understanding.

**Pre-emptive context rotation.** Early warning system that monitors token budget pressure and triggers proactive rotation before cache invalidation. Prevents degradation from emergency truncation.

**Bridge proxy.** Optimize context for external LLM callers (Claude Code, LangChain, OpenCode) without modifying them. The bridge sits as a reverse proxy, applying compaction and summarization to requests in flight. Point Claude Code at `localhost:9200` and get 30-60% token reduction on long sessions with zero behavioral change.

**Pluggable tokenizers.** Choose between fast character heuristic (default) or accurate tiktoken counting with support for specific LLM encoding schemes.

**Tool state machine.** Dynamic tool masking with named states and transitions. Start in "default" (all tools), transition to "planning" (read-only tools), back to "execution" (write tools enabled). Supports allowed-list, prefill, and logit-mask strategies.

**Metrics and alerting.** Every pipeline run produces a `PipelineResult` with per-stage timing, token counts, cache hit rates, and degradation events. With [sr2-pro](https://sr2.dev/pricing): export to Prometheus, push to OpenTelemetry, and alert on low cache hit rates or circuit breaker activations.

## Benchmarks

Real multi-turn sessions against Claude Opus, side-by-side: SR2 pipeline vs naive concatenation. Run them yourself from `benchmarks/`.

### Coherence (50 turns, 8k token budget)

| Question | Naive | Managed |
|----------|-------|---------|
| What database did the team decide to use? | MISS | HIT |
| What authentication method was chosen? | MISS | HIT |
| What message queue system was selected? | HIT | HIT |
| What frontend framework did the team pick? | HIT | HIT |
| What container orchestration tool was decided on? | HIT | HIT |
| **Score** | **3/5** | **5/5** |

Naive used 7,122 tokens. Managed used 3,329 tokens. **3.6x more information per token.** 100% KV-cache prefix hit rate.

Context breakdown (managed):

| Zone | Tokens | % |
|------|--------|---|
| Raw (recent, verbatim) | 1,538 | 55.2% |
| Compacted (tool refs) | 1,249 | 44.8% |
| Summarized (LLM digest) | 0 | 0.0% |

### Cost (30 turns)

| | Naive | Managed | Saved |
|---|-------|---------|-------|
| Input tokens | 184,456 | 88,638 | 51.9% |
| Cost | $0.184 | $0.089 | $0.096 |

**52% token reduction per session.** Multiply by however many agent sessions you run per day.

## Architecture

SR2 is a **library**, not a framework. It compiles context — your code owns the LLM call, tool execution, and agent loop.

The repo includes an **agent runtime** (`packages/sr2-runtime/`) that wires the library into a working agent with an LLM loop, session management, Telegram/HTTP/timer plugins, and MCP tool integration. Use it as-is or as a reference for your own integration.

```
packages/
├── sr2/                   # Core context engineering library (PyPI: sr2)
│   └── src/sr2/
│       ├── config/        #   Config models, loader, validation, schema gen
│       ├── pipeline/      #   Engine, router, conversation manager, post-processor
│       ├── resolvers/     #   Content resolvers (config, input, session, retrieval, etc.)
│       ├── cache/         #   Cache policies and registry
│       ├── compaction/    #   Rule-based content compaction
│       ├── summarization/ #   LLM-powered conversation summarization
│       ├── memory/        #   Extraction, retrieval, conflicts, resolution
│       ├── degradation/   #   Circuit breaker and degradation ladder
│       ├── tools/         #   Tool definitions, state machine, masking strategies
│       ├── metrics/       #   Collector, exporter, alerts
│       ├── tokenization/  #   Pluggable tokenizers (heuristic, tiktoken)
│       ├── normalization/ #   LLM response cleaning (thinking blocks, markdown, JSON)
│       ├── eval/          #   Multi-turn evaluation framework and benchmarking
│       └── a2a/           #   Agent-to-Agent protocol support
│
├── sr2-runtime/           # Agent runtime (PyPI: sr2-runtime, depends on sr2)
│   └── src/sr2_runtime/
│       ├── agent.py       #   Main Agent class
│       ├── cli.py         #   CLI entry point (sr2-agent)
│       ├── llm/           #   LLM client, agentic loop, streaming
│       ├── mcp/           #   MCP client and transports
│       ├── plugins/       #   Interface plugins (http, telegram, timer, a2a, single-shot)
│       ├── session/       #   Session lifecycle management
│       └── heartbeat/     #   Scheduled agent callbacks
│
└── sr2-bridge/            # Context optimization proxy (PyPI: sr2-bridge, depends on sr2)
    └── src/sr2_bridge/
│
configs/               # Example configs
│   ├── defaults.yaml  #   Library defaults
│   └── agents/edi/    #   Example agent
│
tests/                 # 1,292 tests
│   ├── sr2/           #   Core library tests
│   ├── runtime/       #   Runtime tests
│   └── bridge/        #   Bridge tests
```

## Running the Bridge

The bridge optimizes context for external LLM callers like Claude Code.

```bash
# Install bridge
pip install -e packages/sr2-bridge

# Terminal 1: start the bridge (zero-config)
sr2-bridge

# Terminal 2: point Claude Code at the bridge
ANTHROPIC_BASE_URL=http://localhost:9200 claude
```

The bridge compacts tool outputs, summarizes old conversation turns, and forwards optimized requests to the real API. See the [Bridge Guide](docs/guide-bridge.md) for configuration and details.

## Running the Example Agent

The repo includes an example agent as a working reference.

```bash
# Install everything
uv sync --all-extras

# Run with HTTP API
sr2-agent configs/agents/edi --http --port 8008

# Open the chat UI
open http://localhost:8008

# Or talk to it via curl
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "session_id": "default"}'
```

The agent also exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`) so you can connect [Open WebUI](https://github.com/open-webui/open-webui) or any OpenAI-compatible client directly to it. See the [Quick Reference](docs/reference.md) for setup details.

The example agent requires Ollama running locally (see `configs/agents/edi/agent.yaml` for model config). Swap the model strings to use any LiteLLM-supported provider.

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
pytest tests/ --ignore=tests/integration/ -v

# Run integration tests (requires PostgreSQL)
docker compose -f docker-compose.test.yml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v
docker compose -f docker-compose.test.yml down

# Lint
ruff check packages/
ruff format packages/

# Generate config docs from Pydantic models
sr2-config-docs --format md > docs/configuration.md
```

## Framework Integrations

SR2 works with any LLM framework — it compiles context, your framework handles the rest.

| Framework | Example |
|-----------|---------|
| OpenAI Agents SDK | [examples/integrations/openai_agents_sdk.py](examples/integrations/openai_agents_sdk.py) |
| LangChain | [examples/integrations/langchain_example.py](examples/integrations/langchain_example.py) |
| Pydantic AI | [examples/integrations/pydantic_ai_example.py](examples/integrations/pydantic_ai_example.py) |
| CrewAI | [examples/integrations/crewai_example.py](examples/integrations/crewai_example.py) |
| LangGraph | [examples/runtime/langgraph_pipeline.py](examples/runtime/langgraph_pipeline.py) |

## Documentation

- **[Getting Started](docs/getting-started.md)** — Install to working pipeline in 5 minutes
- **[Quick Reference](docs/reference.md)** — CLI commands, config structure, key directories
- **[Configuration Reference](docs/configuration.md)** — Every config field, auto-generated from Pydantic models
- **[Architecture Overview](docs/architecture.md)** — Pipeline flow, three-zone conversation, multi-agent
- **[Memory System](docs/guide-memory.md)** — Extraction, conflict resolution, retrieval (SQLite + PostgreSQL backends)
- **[Compaction](docs/guide-compaction.md)** — Five strategies for compressing tool outputs
- **[Tool Masking](docs/guide-tool-masking.md)** — Dynamic tool visibility with state machines
- **[Custom Resolvers](docs/guide-custom-resolvers.md)** — Build pluggable content sources (5 patterns + examples)
- **[Circuit Breakers](docs/guide-circuit-breakers.md)** — Graceful degradation when layers fail
- **[Agent-to-Agent](docs/guide-a2a.md)** — Multi-agent workflows and service composition
- **[Bridge](docs/guide-bridge.md)** — Context optimization proxy for Claude Code and external LLM callers
- **[Heartbeats](docs/guide-heartbeats.md)** — Scheduling agent callbacks for async tasks and retries
- **[Evaluation Harness](docs/guide-eval-harness.md)** — Multi-turn benchmarking framework and evaluation
- **[Observability](docs/observability.md)** — Prometheus and OpenTelemetry setup
- **[Troubleshooting](docs/troubleshooting.md)** — Common errors, debugging, FAQ

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
