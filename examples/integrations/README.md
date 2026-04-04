# Framework Integration Examples

SR2 is a **library, not a framework**. It compiles the optimal context window for your LLM calls. These examples show how to plug SR2 into popular agent frameworks.

**The pattern is always the same:**

1. SR2 compiles context (system prompt + memories + conversation history)
2. The compiled context is passed to the framework as the system message
3. The framework handles tool calling, orchestration, and everything else
4. After the LLM responds, SR2's PostLLMProcessor handles compaction and memory extraction in the background

## Examples

| File | Framework | What it shows |
|------|-----------|---------------|
| `openai_agents_sdk.py` | OpenAI Agents SDK | Custom model class that injects SR2-compiled context before each LLM call |
| `langchain_example.py` | LangChain | Runnable wrapper that compiles context into the system message each turn |
| `pydantic_ai_example.py` | Pydantic AI | Dynamic system prompt via SR2's pipeline, updated every turn |
| `crewai_example.py` | CrewAI | SR2 managing context for CrewAI task execution |

## Prerequisites

All examples require `sr2` plus the framework's package. Install lines are at the top of each file.

These are illustrative -- they need API keys to actually run, but the integration patterns are real.
