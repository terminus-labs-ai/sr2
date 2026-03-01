"""Minimal SR2 pipeline — two layers, no external dependencies.

This is the simplest possible SR2 pipeline: a system prompt layer and a user
input layer. No database, no LLM API key, no runtime — just the library
compiling context.

Run:
    pip install sr2
    python examples/01_minimal_pipeline.py
"""

import asyncio

from sr2.cache.policies import create_default_cache_registry
from sr2.config.models import PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext


async def main():
    # 1. Register resolvers — they fetch content for each layer item.
    #    "config" resolver pulls from agent_config dict.
    #    "input" resolver pulls from trigger_input.
    registry = ContentResolverRegistry()
    registry.register("config", ConfigResolver())
    registry.register("input", InputResolver())

    # 2. Create the pipeline engine.
    engine = PipelineEngine(registry, create_default_cache_registry())

    # 3. Define your context layout.
    #    - "core" layer: immutable (system prompt never changes mid-session,
    #      so the LLM provider can cache it in the KV-cache prefix).
    #    - "conversation" layer: append_only (new input each turn, but the
    #      prefix from previous turns stays stable).
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

    # 4. Compile context for a single turn.
    result = await engine.compile(
        config,
        ResolverContext(
            agent_config={"system_prompt": "You are a helpful coding assistant."},
            trigger_input="How do I reverse a list in Python?",
        ),
    )

    # 5. Use the compiled context.
    print("=== Compiled Context ===")
    print(result.content)
    print()
    print(f"Total tokens: {result.tokens}")
    print(f"Pipeline status: {result.pipeline_result.overall_status}")
    print(f"Stages: {[(s.stage_name, s.status) for s in result.pipeline_result.stages]}")


if __name__ == "__main__":
    asyncio.run(main())
