# Troubleshooting

## Enabling Debug Logging

Most problems become obvious with debug logs. Enable them with the `--log-level` flag:

```bash
sr2-agent configs/agents/edi --log-level DEBUG
```

Key loggers to watch:

| Logger | What It Shows |
|--------|--------------|
| `sr2.pipeline.engine` | Layer resolution, cache hits, circuit breaker state |
| `sr2.compaction.engine` | Which turns get compacted and why |
| `sr2.summarization.engine` | When summarization triggers |
| `runtime.llm.loop` | Full LLM requests and responses, tool calls |
| `runtime.mcp.client` | MCP server connections and tool discovery |
| `runtime.agent` | Agent startup, plugin initialization |

## Config Errors

### `ConfigValidationError: No layers defined`

Your pipeline config has an empty `layers` list. Every pipeline needs at least one layer.

```yaml
# Fix: add at least one layer
layers:
  - name: core
    cache_policy: immutable
    contents:
      - key: system_prompt
        source: config
```

### `ConfigValidationError: Sum of content max_tokens (...) exceeds token_budget (...)`

The combined `max_tokens` limits on your content items add up to more than `token_budget`. Either reduce individual `max_tokens` values or increase the budget.

```yaml
# Problem: 20000 + 20000 = 40000 > 32000
token_budget: 32000
layers:
  - name: core
    contents:
      - key: system_prompt
        source: config
        max_tokens: 20000
  - name: conversation
    contents:
      - key: history
        source: session
        max_tokens: 20000

# Fix: reduce max_tokens or increase budget
token_budget: 48000
```

### `ConfigValidationError: cache-killing layout`

A layer with `cache_policy: always_new` appears before a layer with `cache_policy: append_only`. This invalidates the KV-cache prefix every turn.

```yaml
# Problem: always_new before append_only
layers:
  - name: dynamic       # always_new — changes every turn
    cache_policy: always_new
  - name: conversation   # append_only — expects stable prefix above it
    cache_policy: append_only

# Fix: reorder so append_only comes first, or use immutable
layers:
  - name: core
    cache_policy: immutable
  - name: conversation
    cache_policy: append_only
  - name: dynamic
    cache_policy: always_new   # after append_only is fine
```

### `ValueError: Circular config inheritance detected`

Your YAML configs form a loop. Check the `extends:` fields:

```yaml
# a.yaml: extends: b.yaml
# b.yaml: extends: a.yaml  <- circular!

# Fix: break the cycle. Use a shared base config.
# base.yaml (no extends)
# a.yaml: extends: base.yaml
# b.yaml: extends: base.yaml
```

### `FileNotFoundError` on startup

An `extends:` field points to a file that doesn't exist. Check that the path is relative to the config file's directory.

```yaml
# In configs/agents/myagent/interfaces/user_message.yaml
extends: ../agent.yaml   # Must exist at configs/agents/myagent/agent.yaml
```

## Runtime Errors

### `KeyError: No pipeline config found for interface 'X'`

The HTTP endpoint can't find a pipeline config. This happens when there's no `interfaces/user_message.yaml` and no `pipeline:` section in `agent.yaml`.

```yaml
# Fix option 1: create interfaces/user_message.yaml
extends: ../agent.yaml
pipeline:
  token_budget: 48000
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config

# Fix option 2: add a pipeline section to agent.yaml (used as fallback)
pipeline:
  token_budget: 32000
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config
```

### `KeyError: No resolver registered for source: X`

A content item references a `source` that has no registered resolver. Built-in sources: `config`, `input`, `session`, `runtime`, `static_template`, `retrieval`, `mcp_resource`, `mcp_prompt`.

```yaml
# Problem
contents:
  - key: data
    source: custom_source   # no resolver registered for this

# Fix: use a built-in source, or register a custom resolver before compiling
registry.register("custom_source", MyCustomResolver())
```

### `KeyError: Key 'X' not found in agent_config`

The `config` resolver is looking for a key in `agent_config` that doesn't exist. Either add the key to your agent config dict, or mark the content item as optional.

```yaml
# Fix: mark as optional
contents:
  - key: agent_persona
    source: config
    optional: true    # won't crash if missing
```

### `ImportError: Database URL configured but asyncpg not installed`

You set a database URL but don't have the PostgreSQL driver. Install it:

```bash
pip install sr2[postgres]
```

### `ImportError: mcp package not installed`

MCP servers are configured but the MCP SDK isn't installed:

```bash
pip install sr2[mcp]
```

## Circuit Breakers

### `Circuit breaker open for layer 'X', skipping`

A layer has failed 3 consecutive times. SR2 opens the circuit breaker and skips that layer for a cooldown period (default: 5 minutes).

**What to do:**
1. Check the logs for the underlying failure (usually a resolver error)
2. The circuit breaker will automatically close after cooldown
3. On the next attempt after cooldown, the layer will be retried
4. A single success resets the failure counter

**Common causes:**
- Retrieval layer can't reach the database
- MCP resource resolver can't connect to the server
- External API timeout in a custom resolver

**Config:**
```yaml
degradation:
  circuit_breaker_threshold: 3    # failures before opening
  circuit_breaker_cooldown: 300   # seconds before retrying
```

## Degradation

SR2 has a 5-level degradation ladder. When the system is under extreme stress, it progressively disables features:

| Level | What's Disabled |
|-------|----------------|
| `full` | Nothing — all features active |
| `skip_summarization` | Summarization |
| `skip_intent` | Summarization + intent detection |
| `raw_context` | Summarization + intent + retrieval + compaction |
| `system_prompt_only` | Everything except core layer |

The degradation ladder is triggered by the runtime when error rates spike. At `system_prompt_only`, the agent runs with just the system prompt and the latest input — minimal but functional.

## MCP Issues

### Server fails to connect

The agent logs an error and continues without that server's tools. Check:
- Is the MCP server process running?
- For stdio transport: is the command path correct?
- For HTTP/SSE transport: is the URL reachable?

```bash
# Test an MCP server manually
npx @modelcontextprotocol/inspector stdio -- command args
```

### Tool not showing up

If an MCP tool isn't appearing in the agent's tool list:
1. Verify the exact tool name matches what the server exposes
2. Check if `curated_tools` is filtering it out
3. Enable debug logging to see what tools were discovered

### Resources/prompts not discovered

MCP servers must declare capabilities for resources and prompts. Check debug logs for messages about missing capabilities.

## Compaction Issues

### Content not being compacted

Check these in order:
1. **Is compaction enabled?** `compaction.enabled: true`
2. **Is the turn in the raw window?** Last `raw_window` turns are never compacted
3. **Is the content too small?** Below `min_content_size` tokens, compaction is skipped
4. **Is there a matching rule?** The turn's `content_type` must match a rule's `type` field
5. **Was it already compacted?** Compaction is idempotent — already-compacted turns are skipped

### Over-aggressive compaction

If too much context is being lost:
- Increase `raw_window` (keep more recent turns verbatim)
- Increase `min_content_size` (skip small content)
- Enable `recovery_hint: true` on rules so the agent can re-fetch

## Memory Issues

### Memories not being extracted

Memory extraction requires a `fast_complete` callable (a fast LLM for extraction). If not configured, extraction is silently skipped.

### Conflicting memories silently overwritten

By default, `identity` and `semi_stable` memories use `latest_wins_archive` — the old value is archived (soft-deleted). Check the memory store for archived entries if you need to see what was overwritten.

### Retrieval returning irrelevant results

- Try adjusting `top_k` (fewer results = higher relevance threshold)
- If using `hybrid` strategy without an embedding callable, only keyword search runs
- Check dimensional matching — `exact` strategy may be too strict, try `best_fit`

## Performance

### High token usage

1. Check `token_budget` — is it set appropriately?
2. Are compaction rules defined? Without rules, nothing gets compacted
3. Is summarization enabled? It reduces old context
4. Check layer ordering — `immutable` layers should be first for KV-cache reuse

### Slow pipeline compilation

1. Check resolver timeouts — external resolvers (MCP, retrieval) may be slow
2. Mark slow-to-resolve items as `optional: true` so they don't block the pipeline
3. Use `compaction_timing: post_llm_async` (default) so compaction doesn't block responses

## FAQ

**Q: Can I use SR2 without the agent runtime?**
Yes. SR2 is a library. Import `PipelineEngine`, register resolvers, call `engine.compile()`. See the [Getting Started](getting-started.md) guide.

**Q: What LLM providers does SR2 support?**
Any provider supported by LiteLLM — OpenAI, Anthropic, Google, Ollama, Azure, and many more. The runtime uses LiteLLM for all LLM calls.

**Q: Does SR2 work with LangChain / CrewAI / AutoGen?**
SR2 is framework-agnostic. It compiles context — your framework owns the LLM call. You'd call `engine.compile()` to get the context, then pass it to your framework's LLM call.

**Q: How do I measure if KV-cache optimization is working?**
Check the `sr2_cache_hit_rate` and `sr2_context_prefix_stable` metrics. A prefix stability of 1.0 means your cache prefix is fully reused between turns. See [Observability](observability.md).

**Q: What happens if the token budget is exceeded?**
The engine trims content from the last layers first. The first layer (typically the system prompt) is never trimmed. Content is removed item-by-item from the bottom, or truncated proportionally if needed.

**Q: Can I use PostgreSQL for memory storage?**
Yes. Install `sr2[postgres]`, then call `sr2.set_postgres_store(pool)`. Tables are created automatically.

**Q: Can I use Open WebUI or other OpenAI-compatible clients with SR2?**
Yes. The agent exposes `/v1/chat/completions` and `/v1/models` endpoints. Point your client at `http://localhost:8008/v1` with any API key (SR2 doesn't require auth). See the [Quick Reference](reference.md) for Open WebUI setup.
