# Bridge

SR2 Bridge is a reverse proxy that sits between an external LLM caller (Claude Code, LangChain, OpenCode, etc.) and the upstream API, applying SR2's context optimization — compaction, summarization, and memory — to requests in flight. The caller doesn't need any modification; the bridge intercepts full conversation history, optimizes it, and forwards the reduced version upstream.

## Why

Tools like Claude Code send the full message history on every request. As conversations grow, you're paying for (and waiting on) thousands of tokens of stale tool outputs, old file contents, and redundant confirmations. The bridge transparently compacts and summarizes this history, reducing token usage by 30-60% on long sessions with no behavioral change.

## Quick Start

**Terminal 1 — start the bridge:**
```bash
pip install -e ".[bridge]"
sr2-bridge
```

**Terminal 2 — point Claude Code at it:**
```bash
ANTHROPIC_BASE_URL=http://localhost:9200 claude
```

That's it. Zero config. The bridge uses sensible defaults (compaction enabled, raw window of 5 turns, no summarization until you configure a fast model).

## How It Works

```
External Caller (Claude Code, etc.)
    │
    ▼
SR2 Bridge Server (FastAPI on localhost:9200)
    │
    ├── AnthropicAdapter
    │     translates wire format ↔ SR2 internal
    │
    ├── BridgeEngine
    │     applies compaction, summarization, memory
    │     using SR2 core components directly
    │
    └── BridgeForwarder (httpx)
          forwards to upstream API with original auth
    │
    ▼
Upstream API (api.anthropic.com)
```

On each request:

1. The **adapter** extracts the system prompt and messages from the wire format
2. The **session tracker** identifies which session this request belongs to (by hashing the system prompt — Claude Code's system prompt is unique per session)
3. The **engine** compares the incoming message count to the last known count, ingests only new messages, runs compaction on older turns, and optionally triggers summarization
4. The **adapter** rebuilds the request body with optimized messages
5. The **forwarder** sends the optimized request upstream with the original auth headers
6. The response streams back through the bridge unmodified

## Session Identification

The bridge needs to know which requests belong to the same conversation. Three strategies are available:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `system_hash` (default) | Hash the system prompt text | Claude Code (system prompt contains CLAUDE.md, project context — unique per session) |
| `header` | Read `X-SR2-Session-ID` header | Custom callers that set explicit session IDs |
| `single` | All requests map to one session | Single-user setups, testing |

## Configuration

### Zero-Config

```bash
sr2-bridge
```

Uses defaults: port 9200, upstream `https://api.anthropic.com`, compaction enabled with raw window of 5, no summarization (requires fast model).

### With a Config File

```yaml
# bridge.yaml
bridge:
  port: 9200
  host: 127.0.0.1
  forwarding:
    upstream_url: https://api.anthropic.com
    timeout_seconds: 300
  session:
    strategy: system_hash
    idle_timeout_minutes: 120

pipeline:
  token_budget: 180000
  compaction:
    enabled: true
    raw_window: 5
    rules:
      - type: tool_output
        strategy: schema_and_sample
        max_compacted_tokens: 80
        recovery_hint: true
      - type: file_content
        strategy: reference
      - type: code_execution
        strategy: result_summary
      - type: confirmation
        strategy: collapse
  summarization:
    enabled: true
    trigger: token_threshold
    threshold: 0.75
```

```bash
sr2-bridge bridge.yaml
```

### CLI Overrides

```bash
sr2-bridge bridge.yaml --port 9300 --host 0.0.0.0 --upstream https://custom-api.example.com
```

### Bridge Config Reference

| Field | Default | Description |
|-------|---------|-------------|
| `bridge.port` | `9200` | Port to listen on |
| `bridge.host` | `127.0.0.1` | Host to bind to |
| `bridge.forwarding.upstream_url` | `https://api.anthropic.com` | Upstream API base URL |
| `bridge.forwarding.timeout_seconds` | `300` | Timeout for upstream requests |
| `bridge.session.strategy` | `system_hash` | Session identification strategy |
| `bridge.session.idle_timeout_minutes` | `120` | Idle session cleanup timeout |

The `pipeline:` section uses the same config as the SR2 core library — see [Configuration Reference](configuration.md) for all pipeline fields.

## Auth Handling

The bridge preserves all authentication headers from the incoming request and forwards them to the upstream API. For Claude Code with Pro/OAuth, this means the OAuth token passes through transparently — no API key configuration needed on the bridge.

For internal LLM calls (summarization, memory extraction), the bridge can reuse the captured auth token to make its own `fast_model` calls to the same upstream. This means zero API key config for Pro users.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Main proxy route — optimize context, forward, stream back |
| `/v1/messages/count_tokens` | POST | Passthrough to upstream (no optimization) |
| `/{path}` | ANY | Catchall passthrough for other API endpoints |
| `/health` | GET | Health check (uptime, active sessions, upstream URL) |
| `/metrics` | GET | Prometheus-format metrics |

## Streaming

The bridge fully supports SSE streaming. When the caller requests `"stream": true`:

1. The bridge optimizes the messages
2. Forwards the streaming request upstream
3. Passes each SSE chunk through to the caller in real-time
4. Accumulates the assistant's text from `content_block_delta` events
5. After the stream completes, fires a background task to post-process (update session state for future compaction)

The caller sees no difference from talking to the API directly — same SSE format, same events, same timing.

## Pipeline Integration

The bridge uses SR2's core components directly:

- **CompactionEngine** — applies compaction rules (schema_and_sample, reference, result_summary, collapse) to turns outside the raw window
- **ConversationManager** — manages the three-zone model (summarized → compacted → raw) per session
- **SummarizationEngine** — when configured with a fast model, summarizes the compacted zone when it exceeds the token threshold

See the [Compaction guide](guide-compaction.md) for strategy details and the [Architecture overview](architecture.md) for the three-zone model.

## Monitoring

### Health Check

```bash
curl http://localhost:9200/health
```

```json
{
  "status": "ok",
  "uptime_seconds": 3600.1,
  "active_sessions": 2,
  "upstream": "https://api.anthropic.com"
}
```

### Prometheus Metrics

```bash
curl http://localhost:9200/metrics
```

```
# HELP sr2_bridge_active_sessions Number of active bridge sessions
# TYPE sr2_bridge_active_sessions gauge
sr2_bridge_active_sessions 2

# HELP sr2_bridge_session_requests Total requests per session
# TYPE sr2_bridge_session_requests counter
sr2_bridge_session_requests{session="a1b2c3d4"} 47

# HELP sr2_bridge_session_tokens Estimated tokens per session zone
# TYPE sr2_bridge_session_tokens gauge
sr2_bridge_session_tokens{session="a1b2c3d4",zone="summarized"} 1
sr2_bridge_session_tokens{session="a1b2c3d4",zone="compacted"} 12
sr2_bridge_session_tokens{session="a1b2c3d4",zone="raw"} 5
```

## Logs

The bridge logs optimization events at INFO level:

```
2026-03-23 10:15:32 [runtime.bridge.session_tracker] INFO: New session: a1b2c3d4 (strategy=system_hash)
2026-03-23 10:15:33 [runtime.bridge.engine] INFO: Session a1b2c3d4: compacted 3 turns (1200 -> 180 tokens)
2026-03-23 10:15:33 [runtime.bridge.app] INFO: Session a1b2c3d4: optimized 8450 -> 5200 est. tokens (38% reduction)
```

Use `--log-level DEBUG` for per-request details including message counts and forwarding URLs.

## Custom Adapters

The bridge uses a Protocol-based adapter pattern. The built-in `AnthropicAdapter` handles the Anthropic Messages API. To support other wire formats, implement the `BridgeAdapter` protocol:

```python
class BridgeAdapter(Protocol):
    def extract_messages(self, body: dict) -> tuple[str | None, list[dict]]:
        """Extract (system_prompt, messages) from request body."""
        ...

    def rebuild_body(self, original_body: dict, optimized_messages: list[dict],
                     system_injection: str | None) -> dict:
        """Rebuild request body with optimized messages."""
        ...

    def parse_sse_text(self, chunk: bytes) -> str | None:
        """Extract assistant text from an SSE chunk."""
        ...
```

An OpenAI adapter stub exists at `src/runtime/bridge/adapters/openai.py` for future implementation.

## Troubleshooting

**Bridge starts but Claude Code gets connection errors:**
- Check that `ANTHROPIC_BASE_URL` is set to `http://localhost:9200` (not `https`)
- Verify the bridge is listening: `curl http://localhost:9200/health`

**Responses are slow:**
- The bridge adds minimal latency (message parsing + compaction). If responses are slow, check the upstream API
- Use `--log-level DEBUG` to see timing for each stage

**Session not being tracked (every request looks new):**
- With `system_hash` strategy, sessions are identified by system prompt content. If the system prompt changes between requests, the bridge sees them as different sessions
- Try `strategy: single` for testing

**Compaction not happening:**
- Compaction only applies to turns outside the raw window. With the default `raw_window: 5`, you need more than 5 messages before compaction kicks in
- Check that your compaction rules match the content types in your conversation (e.g., `tool_output` for tool results)
- Enable DEBUG logging to see compaction decisions

**Summarization not triggering:**
- Summarization requires a configured `llm_callable` (fast model). Without it, only compaction runs
- Check that `summarization.enabled: true` and the compacted zone exceeds `threshold * token_budget`
