# pip install sr2 crewai
"""CrewAI + SR2 context engineering.

SR2 compiles the optimal context (system prompt + memories + compacted history)
for each agent's backstory. CrewAI handles task orchestration, delegation,
and the crew execution loop.

Integration point: compile SR2 context into the agent's backstory before
creating the crew. For long-running crews, recompile between tasks.
"""

import asyncio

from crewai import Agent, Task, Crew

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


async def compile_backstory(engine, config, role_prompt, task_context, history):
    """Use SR2 to compile rich context into a CrewAI agent's backstory."""
    compiled = await engine.compile(
        config,
        ResolverContext(
            agent_config={
                "system_prompt": role_prompt,
                "session_history": history,
            },
            trigger_input=task_context,
        ),
    )
    # SR2's compiled output becomes the agent's backstory.
    # This gives the agent compacted conversation history, retrieved memories,
    # and an optimized system prompt -- all within the token budget.
    return compiled.content


async def main():
    engine, config = build_sr2_engine()
    task_history = []

    # SR2 compiles context for each agent -- token budgets, compaction,
    # and memory retrieval are handled automatically.
    # CrewAI handles task delegation and the crew execution loop.
    researcher_backstory = await compile_backstory(
        engine,
        config,
        role_prompt="You are a senior research analyst specializing in AI trends.",
        task_context="Analyze recent developments in context engineering for LLMs",
        history=task_history,
    )

    writer_backstory = await compile_backstory(
        engine,
        config,
        role_prompt="You are a technical writer who creates clear, concise reports.",
        task_context="Write a summary report based on research findings",
        history=task_history,
    )

    researcher = Agent(
        role="AI Research Analyst",
        goal="Find key trends in context engineering",
        backstory=researcher_backstory,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Produce a clear summary report",
        backstory=writer_backstory,
    )

    research_task = Task(
        description="Research the latest context engineering techniques for LLMs.",
        expected_output="A list of 5 key trends with brief explanations.",
        agent=researcher,
    )

    report_task = Task(
        description="Write a concise report summarizing the research findings.",
        expected_output="A 200-word executive summary.",
        agent=writer,
    )

    crew = Crew(agents=[researcher, writer], tasks=[research_task, report_task])
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
