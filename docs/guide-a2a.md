# Agent-to-Agent (A2A) Protocol Guide

SR2 includes support for Agent-to-Agent communication via the A2A protocol. Agents can discover each other, exchange messages, and compose workflows without requiring human intermediaries.

## Overview

The A2A protocol enables:
- **Service discovery** — Find other agents and their capabilities
- **Card exchange** — Share structured agent descriptions (name, tools, prompt)
- **Message passing** — Send requests and receive responses from other agents
- **Workflow composition** — Chain agent calls to solve complex problems

A typical workflow:

```
Agent A (user facing)
  ↓
  → Discover Agent B (tool provider)
  ↓
  → Send request to Agent B
  ← Receive response
  ↓
  → Respond to user with result
```

## Configuration

Enable A2A in your agent config:

```yaml
runtime:
  plugins:
    - a2a:
        service_name: "my_agent"
        service_url: "http://localhost:8008"
        discovery_url: "http://discovery.example.com"
```

Key settings:
- `service_name` — Unique identifier for your agent
- `service_url` — Public URL where this agent listens
- `discovery_url` — Service discovery endpoint (e.g., Consul, etcd, custom registry)

## Agent Card

Each agent publishes a card describing its capabilities:

```python
from sr2.a2a.card import AgentCardGenerator

card = AgentCardGenerator(
    service_name="research_agent",
    description="Conducts research and provides sourced answers",
    version="1.0.0",
    tools=[
        {
            "name": "search",
            "description": "Search the web for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
        },
        {
            "name": "summarize",
            "description": "Summarize a document",
            "parameters": {
                "document": {"type": "string", "description": "Document text"}
            },
        },
    ],
    system_prompt="You are a research assistant. Always cite sources.",
)
```

## Discovery

Find available agents:

```python
from sr2.a2a.client import A2AClientTool

client = A2AClientTool(discovery_url="http://discovery.example.com")

# List all agents
agents = await client.discover_agents()
for agent in agents:
    print(f"{agent.service_name}: {agent.description}")

# Find specific agent
research_agent = await client.discover_agent("research_agent")
print(research_agent.tools)
```

## Message Exchange

Send a request to another agent:

```python
# Prepare a message
message = {
    "role": "user",
    "content": "Find 3 recent articles about quantum computing"
}

# Call the agent
response = await client.send_message(
    target_service="research_agent",
    messages=[message],
)

print(response.content)
# Output: "I found these articles... [with sources]"
```

## Integration with Resolvers

Use A2A calls within your pipeline:

```python
from sr2.resolvers.registry import ContentResolver, ResolverContext, ResolvedContent

class ExternalAgentResolver(ContentResolver):
    """Calls another agent to enrich context."""

    def __init__(self, discovery_url: str) -> None:
        from sr2.a2a.client import A2AClientTool
        self.client = A2AClientTool(discovery_url)

    async def resolve(self, config, context):
        target = config.get("target_agent")
        prompt = config.get("prompt")

        if not target:
            return None

        # Call the external agent
        response = await self.client.send_message(
            target_service=target,
            messages=[{"role": "user", "content": prompt}],
        )

        return ResolvedContent(
            key=f"external_{target}",
            content=response.content,
            tokens=len(response.content) // 4,
            metadata={"agent": target},
        )
```

Use in config:

```yaml
layers:
  - name: memory
    contents:
      - key: research_context
        source: external_agent
        optional: true
        config:
          target_agent: research_agent
          prompt: "What are the latest advances in {{ topic }}?"
```

## Workflow Composition

Chain multiple agents:

```python
async def research_and_analyze():
    """Multi-agent workflow."""
    client = A2AClientTool(discovery_url="http://discovery.example.com")

    # Step 1: Research
    research_response = await client.send_message(
        target_service="research_agent",
        messages=[{"role": "user", "content": "Find recent AI papers"}],
    )

    # Step 2: Analyze
    analysis_response = await client.send_message(
        target_service="analysis_agent",
        messages=[
            {
                "role": "user",
                "content": f"Analyze these papers:\n{research_response.content}",
            }
        ],
    )

    # Step 3: Summarize
    summary_response = await client.send_message(
        target_service="summary_agent",
        messages=[
            {
                "role": "user",
                "content": f"Summarize for executive audience:\n{analysis_response.content}",
            }
        ],
    )

    return summary_response.content
```

## Request/Response Format

Messages follow OpenAI-compatible format:

```python
{
    "messages": [
        {
            "role": "user",
            "content": "Your request here"
        }
    ],
    "temperature": 0.7,
    "max_tokens": 2000,
}
```

Response:

```python
{
    "content": "Agent's response",
    "tokens_used": 342,
    "service_name": "target_agent",
}
```

## Context Isolation

A2A calls are intentionally stateless:
- No session history from caller is shared
- No memory access to caller's data
- Each call gets fresh context

Configure per-service token budgets in A2A interface:

```yaml
interfaces:
  a2a:
    token_budget: 8000  # Smaller budget for stateless calls
    compaction:
      enabled: false     # No history to compact
    summarization:
      enabled: false
    retrieval:
      enabled: false     # No memory access
```

## Error Handling

Gracefully handle agent failures:

```python
try:
    response = await client.send_message(
        target_service="research_agent",
        messages=[message],
        timeout=10,  # Seconds
    )
except asyncio.TimeoutError:
    logger.error("Agent didn't respond in time")
    response = None
except Exception as e:
    logger.error(f"Agent call failed: {e}")
    response = None

if response:
    content = response.content
else:
    content = "[Agent unavailable]"  # Graceful degradation
```

## Example: Research Agent

Here's a complete A2A-enabled research agent:

```python
from fastapi import FastAPI
from sr2.a2a.app import A2AApp
from sr2.a2a.card import AgentCardGenerator

# your agent instance (sr2-spectre or custom)
agent = ...

# Create A2A app
a2a_app = A2AApp(
    agent=agent,
    card=AgentCardGenerator(
        service_name="research_agent",
        description="Conducts research with web search and summarization",
        version="1.0.0",
        tools=[
            {
                "name": "search",
                "description": "Search the web for information",
            },
            {
                "name": "summarize",
                "description": "Summarize text",
            },
        ],
    ),
)

# Mount on FastAPI
app = FastAPI()
app.include_router(a2a_app.router, prefix="/a2a")

# Also mount regular HTTP API
app.include_router(agent.http.router)

# Run with: uvicorn main:app
```

Register with discovery service:

```bash
# Using consul
consul services register -name=research_agent \
  -address=192.168.1.100 \
  -port=8008 \
  -tags="a2a,sr2"
```

## Security Considerations

A2A communication runs over HTTP by default. For production:

1. **Use TLS** — Wrap with reverse proxy (nginx) for HTTPS
2. **Authenticate** — Add API keys or OAuth to discovery and message endpoints
3. **Rate limit** — Prevent abuse via tool calls to external agents
4. **Validate responses** — Don't blindly inject A2A responses into context
5. **Audit logging** — Track which agents called which agents

Example secured setup:

```yaml
runtime:
  plugins:
    - a2a:
        service_name: "my_agent"
        service_url: "https://my-agent.example.com"
        discovery_url: "https://discovery.example.com"
        api_key: "${A2A_API_KEY}"  # From environment
        timeout_seconds: 30
        max_retries: 2
```

## Troubleshooting

**Agent not discovered**
- Check discovery service is running
- Verify `service_url` is publicly accessible
- Look for registration errors in logs

**Messages timeout**
- Increase timeout setting
- Check network connectivity
- Verify target agent is running

**Infinite loops**
- Implement call depth tracking
- Set max_retries = 0 for A2A resolvers
- Avoid circular agent dependencies

**Context pollution**
- Validate A2A responses before injection
- Truncate large responses
- Use optional: true for A2A content items
