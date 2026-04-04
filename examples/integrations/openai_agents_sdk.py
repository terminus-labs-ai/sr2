# pip install sr2 openai-agents
"""OpenAI Agents SDK + SR2 context engineering.

SR2 compiles the optimal context (system prompt + memories + compacted history)
before each LLM call. The Agents SDK handles tool execution and orchestration.

Integration point: a custom Model class that calls SR2.compile() before
forwarding the request to OpenAI.
"""

import asyncio

from agents import Agent, Runner

from sr2.cache.policies import create_default_cache_registry
from sr2.config.models import PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.session_resolver import SessionResolver
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext


# --- SR2 setup ---


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


# --- Integration ---


async def compile_system_prompt(engine, config, user_input, session_history):
    """Use SR2 to compile the full context into a single system prompt string."""
    compiled = await engine.compile(
        config,
        ResolverContext(
            agent_config={
                "system_prompt": "You are a helpful research assistant.",
                "session_history": session_history,
            },
            trigger_input=user_input,
        ),
    )
    # compiled.content has the full context -- use it as the system instruction
    return compiled.content


async def main():
    engine, config = build_sr2_engine()
    session_history = []

    # Compile context with SR2, then pass it as the agent's instructions
    user_msg = "What are the key differences between TCP and UDP?"
    system_prompt = await compile_system_prompt(engine, config, user_msg, session_history)

    # The agent uses SR2's compiled context as its instructions.
    # SR2 handles compaction, token budgets, and KV-cache optimization.
    # The Agents SDK handles tool calling and the response loop.
    agent = Agent(
        name="researcher",
        instructions=system_prompt,
        model="gpt-4o",
    )

    result = await Runner.run(agent, user_msg)
    print(result.final_output)

    # Track history for next turn -- SR2 will compact old turns automatically
    session_history.append({"role": "user", "content": user_msg})
    session_history.append({"role": "assistant", "content": result.final_output})


if __name__ == "__main__":
    asyncio.run(main())
