# SR2

Context engineering library for AI agents — manage token budgets, optimize KV-cache, and maintain quality over long conversations.

- **Save money.** 50%+ token reduction through compaction, summarization, and cache-aware context layout. Same quality, half the cost.
- **Maintain quality.** Your agent stays coherent at turn 200. Three-zone conversation management preserves decisions, preferences, and unresolved issues while discarding noise.
- **Drop-in.** Works with any LLM framework. SR2 compiles context — you own the LLM call, tool execution, and agent loop.

## Benchmarks

Real multi-turn sessions against Claude Opus. SR2 pipeline vs naive concatenation.

| Metric                  | Naive    | SR2      | Savings |
|-------------------------|----------|----------|---------|
| Total tokens (30 turns) | 184,456  | 88,638   | 51.9%   |
| Coherence (50 turns)    | 3/5      | 5/5      | +67%    |
| KV-cache hit rate       | 0%       | 100%     | ---     |

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
    print(f"Tokens: {result.tokens}")
    print(f"Pipeline status: {result.pipeline_result.overall_status}")


asyncio.run(main())
```

Two layers, no caching tricks. In production you'd add memory retrieval, conversation history with compaction, summarization triggers, and per-interface configs.

## Features

- **Compaction** — Tool outputs replaced with references + recovery hints. File contents become path refs. Code results become exit code + first 3 lines.
- **Summarization** — LLM-powered structured summaries preserving decisions, unresolved issues, and user preferences. Routine confirmations discarded.
- **Memory** — Extract structured memories from conversations. Conflict detection and resolution. Hybrid semantic + keyword retrieval. PostgreSQL and SQLite backends.
- **KV-cache optimization** — Layers ordered most-stable to least-stable. Immutable prefix stays cached across turns.
- **Graceful degradation** — Per-layer circuit breakers. If retrieval fails, the breaker opens and the agent keeps running with reduced context instead of crashing.
- **Tool masking** — Dynamic tool visibility with named states and transitions. Show different tools at different stages of a workflow.
- **Metrics** — Every pipeline run produces per-stage timing, token counts, cache hit rates, and degradation events.
- **Per-interface configs** — Different token budgets and strategies per trigger type. Chat gets 48k with full compaction. Heartbeat gets 3k, stateless.

## Installation

```bash
pip install sr2
```

Optional extras:

```bash
pip install sr2[mcp]    # MCP tool integration
pip install sr2[a2a]    # Agent-to-Agent protocol
```

## Documentation

Full docs, architecture guides, and config reference: [github.com/terminus-labs-ai/sr2](https://github.com/terminus-labs-ai/sr2)

Premium features (Prometheus export, OpenTelemetry, advanced alerting): [sr2.dev](https://sr2.dev/pricing)

## License

Apache 2.0
