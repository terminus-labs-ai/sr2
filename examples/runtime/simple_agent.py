"""Example: Single agent execution using the SR2 runtime."""

import asyncio

from sr2_runtime.agent import Agent, AgentConfig


async def main():
    config = AgentConfig(
        name="researcher",
        config_dir="configs/agents/researcher",
    )
    agent = Agent(config)
    await agent.start()

    response = await agent.handle_user_message(
        "What are the key principles of SOLID design?"
    )
    print(f"Response: {response[:200]}...")

    await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
