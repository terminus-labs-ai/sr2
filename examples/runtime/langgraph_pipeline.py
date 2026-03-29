"""Example: Two-agent LangGraph pipeline with SR2 nodes."""

import asyncio

from sr2.runtime.integrations.langgraph import SR2Node, SR2GraphState

# Requires: pip install langgraph
from langgraph.graph import StateGraph, END

# Agents
researcher = SR2Node("examples/runtime/researcher.yaml")
coder = SR2Node("examples/runtime/coder.yaml")

# Graph
graph = StateGraph(SR2GraphState)
graph.add_node("research", researcher)
graph.add_node("code", coder)
graph.set_entry_point("research")
graph.add_edge("research", "code")
graph.add_edge("code", END)
app = graph.compile()


async def main():
    result = await app.ainvoke({
        "current_task": "Research Python logging best practices, then write a logging utility.",
    })
    print("Research output:", result["outputs"].get("researcher", "")[:200])
    print("Code output:", result["outputs"].get("coder", "")[:200])


if __name__ == "__main__":
    asyncio.run(main())
