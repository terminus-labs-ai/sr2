# Custom Content Resolvers Guide

SR2's power comes from pluggable content resolvers. They fetch content for each layer item in your context window. This guide shows how to build custom resolvers for your domain.

## Overview

A resolver implements the `ContentResolver` protocol:

```python
from sr2.resolvers.registry import ContentResolver, ResolverContext, ResolvedContent

class MyCustomResolver(ContentResolver):
    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        """Fetch content and return it."""
        # Your implementation here
        return ResolvedContent(
            key="unique_key",
            content="The content string",
            tokens=42,  # Estimated token count
            metadata={"source": "my_resolver"},
        )
```

## Key Concepts

### ResolverContext
Contains everything you need:
- `agent_config` — The full agent configuration (YAML)
- `trigger_input` — User's message or API input
- `session_id` — Current conversation session
- `interface_type` — How the agent was triggered (http, telegram, etc.)

### ResolvedContent
Your return value:
- `key` — Unique identifier for this content item
- `content` — The actual string to include in context
- `tokens` — Estimated token count (used for budget enforcement)
- `metadata` (optional) — Extra data (not included in context, used for debugging/analytics)

### Return None on Failure
If your resolver can't fetch content, return `None`. If the content item is marked `optional: true` in config, the pipeline skips it gracefully. If required, the circuit breaker may mark the layer as degraded.

## Common Patterns

### 1. Static Template Resolver

Returns fixed content (system prompt, instructions):

```python
class MyInstructionsResolver(ContentResolver):
    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        instructions = "You are a helpful assistant focused on..."
        return ResolvedContent(
            key="instructions",
            content=instructions,
            tokens=len(instructions) // 4,  # Estimate
        )
```

### 2. Dynamic Data Resolver

Fetches from external sources (databases, APIs):

```python
import aiohttp

class DatabaseResolver(ContentResolver):
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url

    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        # Extract parameters from config
        table_name = config.get("table", "documents")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.db_url}/query",
                    params={"table": table_name, "session": context.session_id},
                ) as resp:
                    data = await resp.json()
                    content = json.dumps(data, indent=2)
                    return ResolvedContent(
                        key=f"db_{table_name}",
                        content=content,
                        tokens=len(content) // 4,
                    )
        except Exception as e:
            logger.error(f"Database resolver failed: {e}")
            return None
```

### 3. LLM-based Resolver

Uses a language model to synthesize content:

```python
from litellm import acompletion

class SynthesisResolver(ContentResolver):
    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        prompt = f"""
Given the user message: "{context.trigger_input}"

Synthesize key discussion points from the conversation history.
Keep it concise (under 500 tokens).
"""

        response = await acompletion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content
        return ResolvedContent(
            key="synthesis",
            content=content,
            tokens=len(content) // 4,
        )
```

### 4. Conditional Resolver

Returns different content based on context:

```python
class ContextAwareResolver(ContentResolver):
    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        # Different behavior based on interface type
        if context.interface_type == "telegram":
            content = "Short format for Telegram (max 200 tokens)"
        elif context.interface_type == "http":
            content = "Long format for HTTP (max 2000 tokens)"
        else:
            content = "Default format"

        return ResolvedContent(
            key="interface_specific",
            content=content,
            tokens=len(content) // 4,
            metadata={"interface": context.interface_type},
        )
```

## Integration

Register your resolver in the pipeline:

```python
from sr2.resolvers.registry import ContentResolverRegistry

registry = ContentResolverRegistry()

# Built-in resolvers
registry.register("config", ConfigResolver())
registry.register("input", InputResolver())

# Your custom resolvers
registry.register("my_db", DatabaseResolver("http://db.example.com"))
registry.register("synthesis", SynthesisResolver())
registry.register("instructions", MyInstructionsResolver())
```

Then reference them in config:

```yaml
layers:
  - name: core
    contents:
      - key: instructions
        source: instructions
      - key: tools
        source: config

  - name: memory
    contents:
      - key: retrieved
        source: retrieval
      - key: synthesis
        source: synthesis

  - name: conversation
    contents:
      - key: db_context
        source: my_db
        optional: true
      - key: user_input
        source: input
```

## Best Practices

1. **Handle errors gracefully** — Return `None` instead of raising exceptions. The pipeline will skip optional content or mark the layer as degraded.

2. **Estimate tokens accurately** — Use the character heuristic (len // 4) or tiktoken for better accuracy. Underestimating causes budget overruns; overestimating wastes context space.

3. **Cache when possible** — If your resolver makes external calls, implement caching to avoid redundant work across multiple calls.

4. **Use metadata for debugging** — Include `metadata` with info about cache hits, data source, version, etc. This helps troubleshooting without polluting the context.

5. **Make config flexible** — Allow configuration in the YAML layer definition. Users should be able to tune behavior without code changes.

6. **Log clearly** — Include debug logs showing what your resolver did, especially on failure paths.

7. **Test with mock context** — Write unit tests with `ResolverContext` fixtures to verify your resolver works correctly.

## Example: User Profile Resolver

Here's a complete resolver that fetches user preferences:

```python
import logging
from typing import Any

from sr2.resolvers.registry import ContentResolver, ResolverContext, ResolvedContent

logger = logging.getLogger(__name__)


class UserProfileResolver(ContentResolver):
    """Fetches user preferences and context from a backend service."""

    def __init__(self, profile_service_url: str) -> None:
        self.profile_service = profile_service_url

    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        """Fetch and format user profile data."""
        if not context.session_id:
            logger.debug("No session_id, skipping user profile")
            return None

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.profile_service}/profile/{context.session_id}",
                    timeout=2,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Profile service returned {resp.status}")
                        return None

                    profile = await resp.json()
                    content = self._format_profile(profile)
                    return ResolvedContent(
                        key="user_profile",
                        content=content,
                        tokens=len(content) // 4,
                        metadata={
                            "user_id": profile.get("id"),
                            "preferences_count": len(profile.get("preferences", [])),
                        },
                    )
        except asyncio.TimeoutError:
            logger.error("Profile service timeout")
            return None
        except Exception as e:
            logger.error(f"Failed to fetch profile: {e}")
            return None

    @staticmethod
    def _format_profile(profile: dict[str, Any]) -> str:
        """Format profile data into readable context."""
        lines = ["## User Profile"]
        if name := profile.get("name"):
            lines.append(f"Name: {name}")
        if role := profile.get("role"):
            lines.append(f"Role: {role}")

        if prefs := profile.get("preferences"):
            lines.append("\nPreferences:")
            for pref in prefs:
                lines.append(f"- {pref}")

        return "\n".join(lines)
```

Use it:

```python
registry.register("user_profile", UserProfileResolver("http://api.example.com"))
```

## Troubleshooting

**Resolver returns None unexpectedly**
- Check logs for exceptions
- Verify `ResolverContext` has the data you expect
- Add defensive null checks for config keys

**Tokens are way off**
- Switch to tiktoken for better accuracy
- Add logging to see actual content length
- Remember character heuristic is an estimate

**Pipeline skips your layer**
- Check if resolver returned None and layer is required (not optional)
- Look for circuit breaker activations in logs
- Verify resolver is registered with correct name

**Config validation fails**
- Make sure `source:` name matches registration
- Verify `key:` is unique within the layer
- Check extra config fields are supported by your resolver
