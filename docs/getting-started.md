# Getting Started

This guide takes you from zero to a working SR2 pipeline in under 5 minutes. No database, no LLM API key, no runtime — just the library.

## Install

```bash
git clone https://github.com/terminus-labs-ai/sr2.git
cd sr2
pip install -e .
```

This gives you the core library: config models, pipeline engine, resolvers, compaction, and caching. No heavy dependencies beyond Pydantic and PyYAML.

## Your First Pipeline

SR2 compiles context through a layered pipeline. Each layer has a cache policy and a list of content items. The engine resolves each item, enforces a token budget, and returns a compiled context string.

```python
import asyncio
from sr2.config.models import PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.cache.policies import create_default_cache_registry


async def main():
    # 1. Register resolvers — they fetch content for each layer item
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
    print(f"Status: {result.pipeline_result.overall_status}")


asyncio.run(main())
```

Save this as `hello_sr2.py` and run it:

```bash
python hello_sr2.py
```

That's a two-layer pipeline. The `core` layer is immutable (system prompt never changes mid-session), and the `conversation` layer is append-only (new input each turn). The engine compiles both into a single context string.

## What Just Happened

1. **Resolvers** fetched content. The `config` resolver pulled `system_prompt` from `agent_config`. The `input` resolver pulled from `trigger_input`.
2. **The engine** compiled layers in order, tracked token counts, and checked the budget (8000 tokens).
3. **Cache policies** were applied. `immutable` means the core layer string is cached and never recomputed. `append_only` means the conversation layer can grow but its prefix stays stable.

This ordering matters for KV-cache: your LLM provider caches the prefix (system prompt), so you only pay to process new tokens each turn.

## Adding a Third Layer

Real agents need more than two layers. Here's a three-layer setup with a memory/context layer in the middle:

```python
config = PipelineConfig(
    token_budget=16000,
    layers=[
        {
            "name": "core",
            "cache_policy": "immutable",
            "contents": [
                {"key": "system_prompt", "source": "config"},
            ],
        },
        {
            "name": "memory",
            "cache_policy": "append_only",
            "contents": [
                {"key": "relevant_context", "source": "config", "optional": True},
            ],
        },
        {
            "name": "conversation",
            "cache_policy": "append_only",
            "contents": [
                {"key": "session_history", "source": "session", "optional": True},
                {"key": "user_input", "source": "input"},
            ],
        },
    ],
)
```

Note `optional: True` — if a resolver fails or returns empty for an optional item, the pipeline continues. Required items (the default) cause the layer to fail.

## Using YAML Configs

In production, you define pipelines in YAML instead of Python dicts. SR2 supports config inheritance: `defaults.yaml` -> `agent.yaml` -> `interfaces/user_message.yaml`. More specific configs win in a deep merge.

```yaml
# my_pipeline.yaml
token_budget: 16000

layers:
  - name: core
    cache_policy: immutable
    contents:
      - key: system_prompt
        source: config

  - name: conversation
    cache_policy: append_only
    contents:
      - key: user_input
        source: input
```

Load it with the config loader:

```python
from sr2.config.loader import ConfigLoader

loader = ConfigLoader(defaults_path="configs/defaults.yaml")
config = loader.load("my_pipeline.yaml")
```

The loader merges your file with the defaults, so you only specify what you want to override.

## Next Steps

- **[Configuration Reference](configuration.md)** — every config field documented
- **[Architecture Overview](architecture.md)** — pipeline flow, three-zone conversation management
- **[Compaction Guide](guide-compaction.md)** — how tool outputs get compressed
- **[Memory Guide](guide-memory.md)** — extraction, conflict resolution, retrieval
- **[Tool Masking Guide](guide-tool-masking.md)** — dynamic tool visibility with state machines
- **[Observability](observability.md)** — Prometheus and OpenTelemetry setup
- **[Troubleshooting](troubleshooting.md)** — common errors and how to fix them
