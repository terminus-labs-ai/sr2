"""Example: Single SR2Runtime agent execution."""

import asyncio

from sr2.runtime import SR2Runtime


async def main():
    runtime = SR2Runtime.from_config("examples/runtime/researcher.yaml")
    result = await runtime.execute("What are the key principles of SOLID design?")

    print(f"Output: {result.output[:200]}...")
    print(f"Tokens: {result.metrics.total_tokens}")
    print(f"LLM calls: {result.metrics.llm_calls}")
    print(f"Success: {result.success}")


if __name__ == "__main__":
    asyncio.run(main())
