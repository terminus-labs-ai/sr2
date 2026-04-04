# pip install sr2 langchain langchain-openai
"""LangChain + SR2 context engineering.

SR2 compiles the optimal context (system prompt + memories + compacted history)
and injects it as the system message. LangChain handles the chain execution,
tool calling, and output parsing.

Integration point: compile context with SR2 before each chain invocation,
then pass the compiled content as SystemMessage.
"""

import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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


async def sr2_invoke(llm, engine, config, user_input, session_history):
    """Compile context with SR2, then invoke the LangChain LLM."""
    # SR2 compiles system prompt + memories + compacted conversation
    compiled = await engine.compile(
        config,
        ResolverContext(
            agent_config={
                "system_prompt": "You are a helpful coding assistant.",
                "session_history": session_history,
            },
            trigger_input=user_input,
        ),
    )

    # Pass SR2's compiled context as the system message.
    # SR2 handles token budgets, compaction, and KV-cache optimization.
    # LangChain handles the LLM call, tool routing, and output parsing.
    messages = [
        SystemMessage(content=compiled.content),
        HumanMessage(content=user_input),
    ]
    return await llm.ainvoke(messages)


async def main():
    engine, config = build_sr2_engine()
    llm = ChatOpenAI(model="gpt-4o")
    session_history = []

    # Multi-turn loop: SR2 manages context growth across turns
    for user_msg in ["Write a Python fibonacci function", "Now add memoization"]:
        response = await sr2_invoke(llm, engine, config, user_msg, session_history)
        print(f"User: {user_msg}")
        print(f"Assistant: {response.content}\n")

        session_history.append({"role": "user", "content": user_msg})
        session_history.append({"role": "assistant", "content": response.content})


if __name__ == "__main__":
    asyncio.run(main())
