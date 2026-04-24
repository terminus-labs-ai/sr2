# SR2 — Developer Quick Reference

## Install

```bash
uv sync                          # Install all deps (core + dev)
```

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
uv run pytest tests/sr2/test_config/ -v

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
| `src/sr2/` | Core context engineering library |
| `tests/` | Unit + integration tests |

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
# In an interface pipeline config (e.g. interfaces/heartbeat_plan.yaml)
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
