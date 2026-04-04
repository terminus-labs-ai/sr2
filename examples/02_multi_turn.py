"""Multi-turn conversation with compaction.

Demonstrates:
- Three-layer pipeline (core, context, conversation)
- Session history across multiple turns
- Compaction of old tool outputs
- Token budget enforcement

Run:
    pip install sr2
    python examples/02_multi_turn.py
"""

import asyncio
import logging

# Suppress expected pipeline warnings when recompiling across turns
logging.getLogger("sr2.pipeline.engine").setLevel(logging.ERROR)

from sr2.cache.policies import create_default_cache_registry
from sr2.compaction.engine import CompactionEngine, ConversationTurn
from sr2.config.models import CompactionConfig, CompactionRuleConfig, PipelineConfig
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.config_resolver import ConfigResolver
from sr2.resolvers.input_resolver import InputResolver
from sr2.resolvers.registry import ContentResolverRegistry, ResolverContext
from sr2.resolvers.session_resolver import SessionResolver


async def main():
    # --- Setup ---
    registry = ContentResolverRegistry()
    registry.register("config", ConfigResolver())
    registry.register("input", InputResolver())
    registry.register("session", SessionResolver())

    engine = PipelineEngine(registry, create_default_cache_registry())

    config = PipelineConfig(
        token_budget=4000,
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
                    {"key": "session_history", "source": "session", "optional": True},
                    {"key": "current_message", "source": "input"},
                ],
            },
        ],
    )

    # --- Simulate a multi-turn conversation ---
    session_history = []
    turns = [
        "What's the capital of France?",
        "Tell me more about Paris.",
        "What about its population?",
        "Thanks! Now tell me about London.",
    ]

    print("=== Multi-Turn Conversation ===\n")

    for i, user_message in enumerate(turns):
        result = await engine.compile(
            config,
            ResolverContext(
                agent_config={
                    "system_prompt": "You are a geography expert.",
                    "session_history": session_history,
                },
                trigger_input=user_message,
            ),
        )

        # Simulate an assistant response
        assistant_reply = f"[Assistant response to turn {i + 1}]"

        # Add both to session history
        session_history.append({"role": "user", "content": user_message})
        session_history.append({"role": "assistant", "content": assistant_reply})

        print(f"Turn {i + 1}: {user_message}")
        print(f"  Tokens used: {result.tokens}")
        print(f"  Status: {result.pipeline_result.overall_status}")
        print()

    # --- Demonstrate compaction ---
    print("=== Compaction Demo ===\n")

    compaction_config = CompactionConfig(
        enabled=True,
        raw_window=2,  # keep last 2 turns verbatim
        min_content_size=10,
        rules=[
            CompactionRuleConfig(
                type="tool_output",
                strategy="schema_and_sample",
                max_compacted_tokens=80,
                recovery_hint=True,
            ),
        ],
    )
    compaction_engine = CompactionEngine(compaction_config)

    # Simulate turns with a large tool output
    turns = [
        ConversationTurn(
            turn_number=0,
            role="user",
            content="Search for files about authentication",
        ),
        ConversationTurn(
            turn_number=1,
            role="assistant",
            content="\n".join([f"src/auth/handler_{i}.py: line {i*10}" for i in range(30)]),
            content_type="tool_output",
        ),
        ConversationTurn(
            turn_number=2,
            role="user",
            content="Read the first file",
        ),
        ConversationTurn(
            turn_number=3,
            role="assistant",
            content="Here's the auth handler implementation...",
        ),
    ]

    result = compaction_engine.compact(turns)

    print(f"Original tokens: {result.original_tokens}")
    print(f"Compacted tokens: {result.compacted_tokens}")
    print(f"Turns compacted: {result.turns_compacted}")
    print()

    for turn in result.turns:
        label = "[COMPACTED] " if turn.compacted else ""
        preview = turn.content[:80].replace("\n", " ")
        print(f"  Turn {turn.turn_number} ({turn.role}): {label}{preview}...")


if __name__ == "__main__":
    asyncio.run(main())
