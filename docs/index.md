# SR2

**Context engineering for AI agents.** Manages the full lifecycle of what goes into your LLM's context window — compaction, caching, summarization, graceful degradation, and KV-cache optimization.

Your agent framework handles tool calling and orchestration. SR2 manages the context window.

**Requires Python 3.12+**

## Why SR2

Every turn, agents stuff more history, tool results, and memories into a growing blob of text. Hit the token limit and you start truncating from the top — destroying your KV-cache prefix, evicting the system prompt, losing critical context. The agent gets worse the longer the conversation goes.

SR2 treats the context window as a managed resource. Config-driven pipelines compile context with caching, compaction, and summarization so your agent stays coherent at turn 200 the same way it was at turn 2.

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

Layers are ordered most-stable to least-stable. The system prompt is always the KV-cache prefix. Conversation history changes every turn, so it's last. Your LLM provider caches the expensive prefix and you only pay to process new tokens.

## Quick Start

```bash
pip install sr2
```

```python
import asyncio
from sr2.config.models import PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.cache.policies import create_default_cache_registry


async def main():
    registry = ContentResolverRegistry()
    registry.register("config", ConfigResolver())
    registry.register("input", InputResolver())

    engine = PipelineEngine(registry, create_default_cache_registry())

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

    result = await engine.compile(
        config,
        ResolverContext(
            agent_config={"system_prompt": "You are a helpful assistant."},
            trigger_input="What's the weather like?",
        ),
    )

    print(result.content)
    print(f"Tokens: {result.tokens}")


asyncio.run(main())
```

That's the minimal case. In production you'd add memory retrieval, conversation history with compaction, summarization triggers, and per-interface configs. See the [Configuration Reference](configuration.md) for the full surface area.

## Key Features

- **Three-zone conversation** — Raw turns, compacted turns, summarized zone. Never lose important decisions.
- **KV-cache optimization** — Prefix stability keeps cache hit rates high across turns.
- **Compaction rules** — Tool outputs, file contents, and code results compressed automatically.
- **Automatic summarization** — LLM-powered summaries preserve decisions and discard routine items.
- **Graceful degradation** — Per-layer circuit breakers keep the agent running when layers fail.
- **Per-interface configs** — Different token budgets and strategies per trigger type (chat, heartbeat, A2A).
- **Memory system** — Extract, store, and retrieve structured memories with conflict resolution.
- **Tool state machine** — Dynamic tool masking with named states and transitions.
- **Pluggable tokenizers** — Heuristic (fast) or tiktoken (accurate).

## Benchmarks

Real multi-turn sessions against Claude Opus — SR2 pipeline vs naive concatenation.

| Metric | Naive | SR2 |
|--------|-------|-----|
| Coherence (50 turns, 8k budget) | 3/5 | **5/5** |
| Tokens used | 7,122 | **3,329** |
| Input tokens (30 turns) | 184,456 | **88,638** |
| Cost savings | — | **52%** |

100% KV-cache prefix hit rate. 3.6x more information per token.

## Packages

| Package | Description |
|---------|-------------|
| `sr2` | Core context engineering library |
| `sr2-pro` | PostgreSQL + pgvector, OpenTelemetry, Prometheus, alerts |

## Next Steps

- [Getting Started](getting-started.md) — Install to working pipeline in 5 minutes
- [Architecture](architecture.md) — Pipeline flow, three-zone conversation, multi-agent patterns
- [Configuration](configuration.md) — Every config field, auto-generated from Pydantic models
- [Guides](guide-memory.md) — Deep dives on memory, compaction, tool masking, and more
