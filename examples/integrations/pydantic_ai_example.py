# pip install sr2 pydantic-ai
"""Pydantic AI + SR2 context engineering.

SR2 compiles the optimal context (system prompt + memories + compacted history)
and provides it as the dynamic system prompt. Pydantic AI handles structured
output parsing, tool execution, and the agent loop.

Integration point: use Pydantic AI's dynamic system prompt hook to call
SR2.compile() before each LLM invocation.
"""

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent

from sr2.cache.policies import create_default_cache_registry
from sr2.config.models import PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext


def build_sr2_engine():
    """Create a minimal SR2 pipeline engine."""
    registry = ContentResolverRegistry()
    registry.register("config", ConfigResolver())
    registry.register("input", InputResolver())
    registry.register("session", SessionResolver())
    engine = PipelineEngine(registry, create_default_cache_registry())

    config = PipelineConfig(
        token_budget=16000,
        layers=[
            {
                "name": "core",
                "cache_policy": "immutable",
                "contents": [{"key": "system_prompt", "source": "config"}],
            },
            {
                "name": "conversation",
                "cache_policy": "append_only",
                "contents": [
                    {"key": "session_history", "source": "session", "optional": True},
                    {"key": "current_message", "source": "input"},
                ],
            },
        ],
    )
    return engine, config


# Shared state for the dynamic system prompt
@dataclass
class Deps:
    sr2_engine: PipelineEngine
    sr2_config: PipelineConfig
    session_history: list
    current_input: str = ""


# Create the Pydantic AI agent with a dynamic system prompt.
# The system prompt is recompiled by SR2 on every turn.
agent = Agent("openai:gpt-4o", deps_type=Deps)


@agent.system_prompt
async def sr2_system_prompt(ctx) -> str:
    """SR2 compiles the full context each turn -- compaction, token budgets,
    memory retrieval, and KV-cache optimization all happen here.
    Pydantic AI handles tool calling, structured outputs, and retries."""
    compiled = await ctx.deps.sr2_engine.compile(
        ctx.deps.sr2_config,
        ResolverContext(
            agent_config={
                "system_prompt": "You are a helpful data analysis assistant.",
                "session_history": ctx.deps.session_history,
            },
            trigger_input=ctx.deps.current_input,
        ),
    )
    return compiled.content


async def main():
    engine, config = build_sr2_engine()
    session_history = []

    for user_msg in ["Explain the Central Limit Theorem", "Give me a Python example"]:
        deps = Deps(
            sr2_engine=engine,
            sr2_config=config,
            session_history=session_history,
            current_input=user_msg,
        )
        result = await agent.run(user_msg, deps=deps)
        print(f"User: {user_msg}")
        print(f"Assistant: {result.output}\n")

        session_history.append({"role": "user", "content": user_msg})
        session_history.append({"role": "assistant", "content": str(result.output)})


if __name__ == "__main__":
    asyncio.run(main())
