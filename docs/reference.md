# SR2 — Developer Quick Reference

## Install

```bash
uv sync                          # Install all deps (core + dev)
```

## Run the Agent

```bash
# With HTTP API (interactive)
uv run sr2-agent agents/edi/configs --http --port 8008

# Headless (heartbeats only, no HTTP)
uv run sr2-agent agents/edi/configs

# Options
#   --name EDI               Override agent name
#   --defaults path.yaml     Custom defaults file
#   --log-level DEBUG        Verbose logging
```

Once running with `--http`, open the built-in chat UI at [http://localhost:8008](http://localhost:8008), or talk to it via curl:

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "session_id": "default"}'
```

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Built-in web chat UI |
| `POST /chat` | SR2 native chat (`{"message": "...", "session_id": "..."}`) |
| `POST /v1/chat/completions` | OpenAI-compatible chat (for Open WebUI, etc.) |
| `GET /v1/models` | OpenAI-compatible model list |
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `GET /.well-known/agent.json` | A2A Agent Card |

### Using with Open WebUI

SR2 exposes OpenAI-compatible endpoints, so you can connect [Open WebUI](https://github.com/open-webui/open-webui) (or any OpenAI-compatible client) directly to the agent:

```bash
docker run -d -p 3000:8080 \
  -e OLLAMA_API_BASE_URL="" \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8008/v1 \
  -e OPENAI_API_KEY=sr2 \
  -e WEBUI_AUTH=false \
  ghcr.io/open-webui/open-webui:main
```

Open [http://localhost:3000](http://localhost:3000) and select the `sr2-edi` model. Or use the `docker compose` setup — see the `open-webui` service in `docker-compose.yaml`.

## Generate Config Schema / Docs

```bash
# Markdown reference doc
uv run python -m schema_gen --format md > docs/configuration.md

# JSON Schema (for VS Code YAML validation)
uv run python -m schema_gen --format json > schema.json

# YAML-formatted schema
uv run python -m schema_gen --format yaml > schema.yaml
```

Or via the entry point:

```bash
sr2-config-docs --format md > CONFIG.md
```

### VS Code Integration

Add to `.vscode/settings.json` for YAML autocomplete + inline validation:

```json
{
  "yaml.schemas": {
    "./schema.json": [
      "configs/agents/*/agent.yaml",
      "configs/agents/*/interfaces/*.yaml",
      "configs/defaults.yaml"
    ]
  }
}
```

## Tests

```bash
# Unit tests (fast, no external deps)
uv run pytest tests/ --ignore=tests/integration/ -v

# Config tests only
uv run pytest tests/test_config/ -v

# With coverage
uv run pytest tests/ --ignore=tests/integration/ --cov=sr2 --cov-report=term-missing

# Integration tests (needs PostgreSQL)
docker compose -f docker-compose.test.yaml up -d
RUN_INTEGRATION=1 uv run pytest tests/integration/ -v
docker compose -f docker-compose.test.yaml down
```

## Lint

```bash
uv run ruff check src/
uv run ruff format src/
```

## Config Inheritance

```
configs/defaults.yaml                   ← Library defaults (all fields have defaults)
└── agents/edi/agent.yaml               ← Agent overrides (system prompt, tools, LLM, etc.)
            └── interfaces/x.yaml       ← Per-interface pipeline overrides
```

Deep merge, more specific wins. Use `extends:` to reference parent config.

## Key Directories

| Path | What |
|---|---|
| `configs/defaults.yaml` | Library-wide defaults |
| `configs/agents/edi/agent.yaml` | Agent config (system prompt, tools, LLM, plugins) |
| `configs/agents/edi/interfaces/` | Per-interface pipeline configs |
| `src/sr2/` | Core context engineering library |
| `src/runtime/` | Agent runtime (CLI, LLM loop, plugins, sessions) |
| `tests/` | Unit + integration tests |

## Creating a New Agent

1. Create `configs/agents/<name>/agent.yaml` with system prompt, tools, and LLM settings
2. Create pipeline configs in `configs/agents/<name>/interfaces/`
3. Run: `uv run sr2-agent configs/agents/<name> --http`

See `configs/agents/edi/` for a working example.

## Runtime Config (`runtime:`)

The `runtime` section configures the agent runtime — LLM connections, database, loop behavior, and session defaults.

```yaml
runtime:
  database:
    url: "${DATABASE_URL}"           # PostgreSQL connection string (env var substitution)
    pool_min: 2                      # Minimum connection pool size
    pool_max: 10                     # Maximum connection pool size

  llm:
    model:                           # Main LLM — chat, reasoning, tool use
      name: "claude-sonnet-4-20250514"
      api_base: null                 # API base URL (Ollama, vLLM, etc.)
      max_tokens: 4096
      model_params:
        temperature: 0.7
    fast_model:                      # Fast LLM — extraction, summarization, intent
      name: "claude-haiku-4-5-20251001"
      api_base: null
      max_tokens: 1000
      model_params:
        temperature: 0.3
    embedding:                       # Embedding model — memory retrieval
      name: "text-embedding-3-small"
      api_base: null

  loop:
    max_iterations: 25               # Max tool-call loop iterations

  session:
    max_turns: 200                   # Default max turns for unnamed sessions
    idle_timeout_minutes: 60         # Default idle timeout

  heartbeat:
    enabled: false                   # Enable schedule_heartbeat / cancel_heartbeat tools
    poll_interval_seconds: 30        # Scanner poll frequency (min: 5)
    max_context_turns: 10            # Turns carried from source session
    session_lifecycle: ephemeral     # Session lifecycle for heartbeat sessions
    pipeline: null                   # Custom pipeline config path (optional)
    max_pending_per_agent: 100       # Max queued heartbeats
```

All fields have defaults — you only need to specify what you want to override. See [Heartbeat Guide](guide-heartbeats.md) for details on the heartbeat system.

## Interfaces (`interfaces:`)

Interfaces define how the agent receives input. Each interface maps to a plugin type and a session.

```yaml
interfaces:
  telegram:
    plugin: telegram
    session:
      name: main_chat
      lifecycle: persistent
    pipeline: interfaces/user_message.yaml

  api:
    plugin: http
    port: 8008
    session:
      name: "{request.session_id}"   # Dynamic session name from request
      lifecycle: persistent
    pipeline: interfaces/user_message.yaml

  email_check:
    plugin: timer
    interval_seconds: 300
    session:
      name: heartbeat_email
      lifecycle: ephemeral
    pipeline: interfaces/heartbeat_email.yaml

  agent_calls:
    plugin: a2a
    session:
      name: "a2a_{task_id}"
      lifecycle: ephemeral
    pipeline: interfaces/a2a_inbound.yaml
```

Plugin-specific fields (`port`, `interval_seconds`, `enabled`, etc.) are passed through — the interface model allows extra fields.

### Session Lifecycles

| Lifecycle | Behavior | Persisted | Use Case |
|---|---|---|---|
| `persistent` | Survives across triggers. Compaction/summarization apply. | PostgreSQL | User conversations |
| `ephemeral` | Fresh per trigger. Destroyed after processing. | In-memory only | Heartbeats, A2A calls |
| `rolling` | Persistent but capped at `max_turns`. Oldest dropped. | PostgreSQL | Monitoring, log watchers |

## Sessions (`sessions:`)

Named session configurations override the runtime defaults.

```yaml
sessions:
  main_chat:
    max_turns: 200
    idle_timeout_minutes: 60
  _default:                          # Fallback for sessions not listed above
    max_turns: 100
    idle_timeout_minutes: 30
```

## MCP Servers (`mcp_servers:`)

MCP (Model Context Protocol) servers give the agent external tools — file access, web search, email, databases, etc. The agent connects to each server on startup, discovers its tools, and registers them so the LLM can call them like any other tool.

### How it works

```
agent.yaml                        Agent startup
┌─────────────┐                   ┌──────────────────────────────┐
│ mcp_servers: │   ──────────►   │ 1. Connect to each server    │
│   - name: fs │                  │ 2. Discover available tools  │
│     url: ... │                  │ 3. Filter by `tools:` list   │
│     tools:   │                  │ 4. Register in ToolExecutor  │
│       - read │                  │ 5. Expose schemas to LLM     │
└─────────────┘                   └──────────────────────────────┘
```

The LLM sees MCP tools identically to built-in tools — it calls them by name and gets results back. No special handling needed in prompts.

### Configuration

```yaml
mcp_servers:
  # stdio transport — runs a local command
  - name: filesystem
    url: npx -y @modelcontextprotocol/server-filesystem /tmp
    transport: stdio
    tools:                           # Optional — omit to register ALL tools
      - read_file
      - write_file

  # http transport — connects to a running server
  - name: web-search
    url: http://localhost:3001/mcp
    transport: http
    env:
      API_KEY: "${SEARCH_API_KEY}"   # ${VAR} substituted from env at startup

  # stdio with extra args and env
  - name: gmail
    url: npx @anthropic/mcp-server-gmail
    transport: stdio
    tools: [search_emails, read_email, send_email, create_draft]
    env:
      GMAIL_CREDENTIALS_PATH: "${GMAIL_CREDENTIALS_PATH}"
```

### Field reference

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Server name (used in logs and error messages) |
| `url` | Yes | — | **stdio**: shell command to run (e.g. `npx server-name`). **http/sse**: URL to connect to. |
| `transport` | No | `stdio` | `stdio` (local process), `http` (Streamable HTTP), or `sse` (Server-Sent Events) |
| `tools` | No | all | Curated tool list — only these tools are registered. Omit to register everything the server offers. |
| `env` | No | — | Environment variables passed to the server process (stdio only). Supports `${VAR}` substitution. |
| `args` | No | — | Extra command-line args appended to `url` (stdio only) |

### When to use `tools:` (curation)

Most MCP servers expose many tools. Curating keeps the LLM's tool list focused and reduces token usage:

```yaml
# BAD — gmail server exposes 20+ tools, most irrelevant for a heartbeat check
- name: gmail
  url: npx @anthropic/mcp-server-gmail
  transport: stdio

# GOOD — only the 4 tools this agent actually needs
- name: gmail
  url: npx @anthropic/mcp-server-gmail
  transport: stdio
  tools: [search_emails, read_email, send_email, create_draft]
```

Omit `tools:` when the server is small or you want everything (e.g., a filesystem server with 3 tools).

### Transports

| Transport | When to use | `url` value |
|---|---|---|
| `stdio` | Server ships as an npm/pip package you run locally | Shell command: `npx @org/server-name /path` |
| `http` | Server already running (self-hosted, cloud) | URL: `http://localhost:3001/mcp` |
| `sse` | Legacy MCP servers using SSE transport | URL: `http://localhost:3001/sse` |

### Roots

Tell MCP servers about your workspace directories so they can scope their behavior (e.g., a filesystem server only accesses declared paths):

```yaml
mcp_servers:
  - name: filesystem
    url: npx -y @modelcontextprotocol/server-filesystem
    transport: stdio
    roots:
      - "file://${HOME}/git/my-project"
      - "file://${HOME}/git/shared-libs"
```

Roots support `${VAR}` env var substitution.

### Resources

MCP Resources expose read-only data from servers (files, DB schemas, API endpoints). Agents can access them two ways:

**As tools** — the agent reads resources on demand during conversation:

```yaml
mcp_servers:
  - name: postgres
    url: npx @anthropic/mcp-server-postgres
    transport: stdio
    expose_resources_as_tools: true    # Registers mcp_list_resources + mcp_read_resource
    env:
      DATABASE_URL: "${DATABASE_URL}"
```

**In the pipeline** — auto-load resources into the context window:

```yaml
# In interfaces/user_message.yaml
layers:
  - name: core
    cache_policy: immutable
    contents:
      - key: system_prompt
        source: config
      - key: "postgres://localhost/mydb/schema"
        source: mcp_resource               # New resolver
        server: postgres
        optional: true
```

### Prompts

MCP Prompts expose reusable prompt templates with fill-in arguments. Like resources, they can be tools or pipeline content:

**As a tool:**

```yaml
mcp_servers:
  - name: prompt-library
    url: npx @my-org/mcp-prompt-server
    transport: stdio
    expose_prompts_as_tools: true      # Registers mcp_get_prompt
```

**In the pipeline:**

```yaml
layers:
  - name: core
    cache_policy: immutable
    contents:
      - key: system_prompt
        source: config
      - key: code_review                   # Prompt name on the server
        source: mcp_prompt                 # New resolver
        server: prompt-library
        arguments:
          language: python
          style: concise
        optional: true
```

### Sampling

MCP Sampling lets servers request the agent's LLM to generate completions — the reverse direction of tool calling. This enables MCP servers with their own agentic workflows.

```yaml
mcp_servers:
  - name: coding-agent
    url: npx @my-org/mcp-coding-server
    transport: stdio
    sampling:
      enabled: true                    # Default: false
      policy: auto_approve             # auto_approve | log_only | deny
      max_tokens: 2048                 # Cap per request (default: 1024)
      rate_limit_per_minute: 20        # Sliding window (default: 10)
```

Policies:
- `auto_approve` — run sampling requests silently
- `log_only` — log and run (default when enabled)
- `deny` — reject all sampling requests

### Full MCP field reference

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | Yes | — | Server name (used in logs and error messages) |
| `url` | Yes | — | **stdio**: shell command. **http/sse**: URL. |
| `transport` | No | `stdio` | `stdio`, `http`, or `sse` |
| `tools` | No | all | Curated tool list — only these tools are registered |
| `env` | No | — | Environment variables for the server process (stdio only) |
| `args` | No | — | Extra command-line args appended to `url` (stdio only) |
| `headers` | No | — | HTTP headers for http/sse transport (e.g., `Authorization`) |
| `roots` | No | — | Root URIs to advertise to the server |
| `resources` | No | — | List of `{uri, subscribe}` for resource discovery |
| `expose_resources_as_tools` | No | `false` | Register `mcp_list_resources` + `mcp_read_resource` tools |
| `prompts` | No | — | List of `{name, arguments}` for prompt auto-loading |
| `expose_prompts_as_tools` | No | `false` | Register `mcp_get_prompt` tool |
| `sampling.enabled` | No | `false` | Enable server-initiated LLM requests |
| `sampling.policy` | No | `log_only` | `auto_approve`, `log_only`, or `deny` |
| `sampling.max_tokens` | No | `1024` | Max tokens per sampling request |
| `sampling.rate_limit_per_minute` | No | `10` | Sliding-window rate limit |

### Troubleshooting

- **"mcp package not installed"** — Install with `pip install -e ".[mcp]"` or `uv sync --extra mcp`
- **Server fails to connect** — The agent logs the error and continues without that server's tools. Other servers are unaffected.
- **Tool not showing up** — Check that the tool name in `tools:` matches exactly what the server reports. Run the server standalone to see its tool list.
- **Resources/prompts not discovered** — The server must support these capabilities. Check logs for "does not support resources/prompts" debug messages.

## Validation

All config is validated on agent startup using Pydantic models. Typos, wrong types, and missing required fields produce clear error messages before anything runs.
