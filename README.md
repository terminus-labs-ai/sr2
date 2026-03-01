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
# Core library (pydantic + pyyaml + litellm, nothing else)
pip install sr2

# With the agent runtime (FastAPI + uvicorn)
pip install sr2[runtime]

# Everything
pip install sr2[all]

# Development
pip install sr2[dev]
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

**Per-interface pipeline configs.** A Telegram chat gets 48k tokens with full compaction/summarization. A heartbeat timer gets 3k tokens with everything disabled. An A2A call gets 8k tokens, stateless. Same agent, different context strategies per trigger type.

**Memory system.** Extract structured memories from conversations, detect conflicts between new and existing memories, resolve them with configurable strategies (latest-wins-archive, keep-both-tagged), and retrieve with hybrid semantic + keyword search.

**Tool state machine.** Dynamic tool masking with named states and transitions. Start in "default" (all tools), transition to "planning" (read-only tools), back to "execution" (write tools enabled). Supports allowed-list, prefill, and logit-mask strategies.

**Metrics and alerting.** Every pipeline run produces a `PipelineResult` with per-stage timing, token counts, cache hit rates, and degradation events. Export to Prometheus. Alert on low cache hit rates or circuit breaker activations.

## Architecture

SR2 is a **library**, not a framework. It compiles context — your code owns the LLM call, tool execution, and agent loop.

The repo includes an **agent runtime** (`src/runtime/`) that wires the library into a working agent with an LLM loop, session management, Telegram/HTTP/timer plugins, and MCP tool integration. Use it as-is or as a reference for your own integration.

```
src/
├── sr2/           # The library (this is what you pip install)
│   ├── config/        #   Config models, loader, validation, schema gen
│   ├── pipeline/      #   Engine, router, conversation manager, post-processor
│   ├── resolvers/     #   Content resolvers (config, input, session, retrieval, etc.)
│   ├── cache/         #   Cache policies and registry
│   ├── compaction/    #   Rule-based content compaction
│   ├── summarization/ #   LLM-powered conversation summarization
│   ├── memory/        #   Extraction, retrieval, conflicts, resolution
│   ├── degradation/   #   Circuit breaker and degradation ladder
│   ├── tools/         #   Tool definitions, state machine, masking strategies
│   ├── metrics/       #   Collector, exporter, alerts
│   └── a2a/           #   Agent-to-Agent protocol support
│
├── harness/           # Agent runtime (optional, uses the library)
│   ├── agent.py       #   Main Agent class
│   ├── cli.py         #   CLI entry point
│   ├── llm_client.py  #   LiteLLM wrapper
│   ├── loop.py        #   Agentic LLM loop
│   ├── plugins/       #   Interface plugins (telegram, http, timer, a2a)
│   └── session.py     #   Session management with lifecycle policies
│
configs/               # Example configs
│   ├── defaults.yaml  #   Library defaults
│   └── agents/edi/    #   Example agent
│
tests/                 # 688 tests
│   ├── test_config/
│   ├── test_pipeline/
│   ├── test_memory/
│   ├── test_compaction/
│   └── ...
```

## Running the Example Agent

The repo includes an example agent as a working reference.

```bash
# Install everything
pip install -e ".[all]"

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
pip install -e ".[dev]"

# Run tests
pytest tests/ --ignore=tests/integration/ -v

# Run integration tests (requires PostgreSQL)
docker compose -f docker-compose.test.yml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v
docker compose -f docker-compose.test.yml down

# Lint
ruff check src/
ruff format src/

# Generate config docs from Pydantic models
python -m schema_gen --format md > docs/configuration.md
```

## Documentation

- **[Getting Started](docs/getting-started.md)** — Install to working pipeline in 5 minutes
- **[Quick Reference](docs/reference.md)** — CLI commands, config structure, key directories
- **[Configuration Reference](docs/configuration.md)** — Every config field, auto-generated from Pydantic models
- **[Architecture Overview](docs/architecture.md)** — Pipeline flow, three-zone conversation, multi-agent
- **[Memory System](docs/guide-memory.md)** — Extraction, conflict resolution, retrieval
- **[Compaction](docs/guide-compaction.md)** — Five strategies for compressing tool outputs
- **[Tool Masking](docs/guide-tool-masking.md)** — Dynamic tool visibility with state machines
- **[Observability](docs/observability.md)** — Prometheus and OpenTelemetry setup
- **[Troubleshooting](docs/troubleshooting.md)** — Common errors, debugging, FAQ

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
