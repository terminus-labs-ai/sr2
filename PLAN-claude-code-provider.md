# SR2 Contribution: Claude Code Provider + OpenAI-Compatible Proxy Support

## Context

SR2's runtime currently uses LiteLLM as its LLM backend. This contribution adds:

1. **Claude Code CLI provider** — alternative LLM backend that spawns `claude -p --bare`, giving SR2 agents access to Claude Code's full tool ecosystem (Bash, file editing, MCPs, search) while SR2 handles context engineering, sessions, and communication (Telegram/HTTP). Claude Code runs stateless (`--bare`) — **SR2 is the sole memory system** (no CLAUDE.md files, no auto-memory).

2. **OpenAI-compatible proxy documentation** (CLIProxyAPI, etc.) — documented config pattern for routing SR2's internal LLM tasks through any OpenAI-compatible proxy via `api_base`. Zero new code — just configs and docs.

3. **Docker support** — extended Dockerfile installs Claude Code CLI via official bash script, with docker-compose config for the full stack. Auth via mounted OAuth credentials from host.

## Architecture

```
SR2 Runtime (container or bare metal)
├── Plugins (existing, unchanged)
│   ├── telegram.py    → Telegram bot
│   └── http.py        → /v1/chat/completions (LibreChat connects here)
│
├── LLM Layer
│   ├── client.py      → LiteLLM (existing, unchanged)
│   │                     api_base → proxy (CLIProxyAPI, etc.)
│   ├── loop.py        → Agentic loop (existing, unchanged)
│   └── claude_code.py → NEW: Claude Code CLI provider
│
└── Agent
    └── agent.py       → Modified: Claude Code provider routing
```

**Execution paths:**
```
User message → Telegram/HTTP → Agent._handle_trigger()
                                  │
                   ┌──────────────┼──────────────┐
                   ▼              ▼              ▼
              Claude Code    LiteLLM         LiteLLM
              `claude -p`    (direct API)    (via proxy)
              full tools     standard        CLIProxyAPI
                   │              │              │
                   └──────────────┴──────────────┘
                                  │
                   SR2 context engineering applies to all paths
```

## Implementation Plan

### Step 1: `ClaudeCodeProvider` class

**New file:** `packages/sr2-runtime/src/sr2_runtime/llm/claude_code.py`

```python
class ClaudeCodeProvider:
    """Wraps the Claude Code CLI as an LLM provider.

    Spawns `claude -p` with stream-json output. Claude Code handles its own
    tool execution (Bash, Edit, etc.) internally — SR2's LLMLoop is bypassed.
    """

    def __init__(self, config: ClaudeCodeConfig):
        # Verify claude binary exists at init time
        if not shutil.which(config.path):
            raise FileNotFoundError(
                f"Claude Code CLI not found at '{config.path}'. "
                "Install with: curl -fsSL https://claude.ai/install.sh | bash"
            )
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._active_processes: list[asyncio.subprocess.Process] = []
        ...

    async def complete(self, prompt: str, system_prompt: str | None = None) -> LoopResult:
        """Run claude -p and return a fully populated LoopResult."""

    async def stream_complete(
        self, prompt: str, system_prompt: str | None = None,
        stream_callback: StreamCallback | None = None,
    ) -> LoopResult:
        """Run claude -p with stream-json, emit StreamEvents, return LoopResult.

        Parses stream-json events to:
        - Emit TextDeltaEvent for text deltas
        - Emit ToolStartEvent / ToolResultEvent for tool use (respects stream_content config)
        - Populate LoopResult with actual ToolCallRecords, token counts, iterations
        """

    async def shutdown(self) -> None:
        """Kill all active subprocesses."""
        for proc in self._active_processes:
            proc.kill()
```

**Returns actual `LoopResult`** (imported from `sr2_runtime.llm.loop`), not a compatible object:

| LoopResult field | Populated from stream-json |
|---|---|
| `response_text` | `result.result` from final event |
| `tool_calls` | `ToolCallRecord` entries parsed from assistant tool_use + tool_result events |
| `iterations` | Count of assistant turn events |
| `total_input_tokens` | `result.usage.input_tokens` |
| `total_output_tokens` | `result.usage.output_tokens` |
| `cached_tokens` | `result.usage.cache_read_input_tokens` |
| `stopped_reason` | Mapped from Claude Code's stop reason |

**CLI invocation:**
```python
# --bare: disables Claude Code's auto-memory, CLAUDE.md, hooks, skills, plugins.
# --system-prompt (not --append): SR2 provides the ENTIRE system prompt since
# --bare has no base prompt to append to.
cmd = [self._claude_path, "--bare", "-p", prompt,
       "--output-format", "stream-json", "--verbose",
       "--include-partial-messages"]

if system_prompt:
    cmd.extend(["--system-prompt", system_prompt])
if self._allowed_tools:
    cmd.extend(["--allowedTools", ",".join(self._allowed_tools)])
if self._permission_mode:
    cmd.extend(["--permission-mode", self._permission_mode])
if self._max_turns:
    cmd.extend(["--max-turns", str(self._max_turns)])
if self._max_budget_usd:
    cmd.extend(["--max-budget-usd", str(self._max_budget_usd)])
if self._working_directory:
    # Sets CWD for subprocess — affects where Bash, Edit, Read operate
    kwargs["cwd"] = self._working_directory
if self._env:
    kwargs["env"] = {**os.environ, **self._env}
```

**Key design decisions:**

1. **`--bare`**: SR2 is the sole memory system. No CLAUDE.md, no auto-memory. Claude Code is used purely for tool execution (Bash, Edit, MCPs).

2. **`--system-prompt` (not `--append-system-prompt`)**: With `--bare`, there's no base prompt. SR2 provides the entire system prompt containing compiled context (memories, summaries, system instructions).

3. **No `--resume`** (v1): SR2 already manages conversation history and passes it as context. Dual session tracking adds complexity with no clear benefit. Each invocation is fresh with full SR2-compiled context. Can be added later if performance demands it.

4. **Concurrency semaphore**: Each invocation spawns a Node.js subprocess. `max_concurrent` config (default 3) prevents resource exhaustion.

5. **Subprocess timeout**: Configurable `timeout_seconds` (default 300). On timeout, process is killed and error LoopResult returned.

**Stream-JSON parsing** (line-by-line from stdout):

| Event type | Action |
|---|---|
| `type: "system", subtype: "init"` | Log session info |
| `type: "stream_event"`, `delta.type == "text_delta"` | Emit `TextDeltaEvent`, accumulate text |
| `type: "assistant"` with `tool_use` blocks | Emit `ToolStartEvent`, record tool call |
| Tool result events | Emit `ToolResultEvent`, record result in `ToolCallRecord` |
| `type: "result"` | Capture final text, token counts, cost; emit `StreamEndEvent` |
| Malformed JSON line | Log warning, skip (don't crash) |

**Stderr** is captured via `asyncio.subprocess.PIPE` and logged. On non-zero exit, stderr content is included in the error message.

### Step 2: Configuration model

**Modify:** `packages/sr2-runtime/src/sr2_runtime/config.py`

```python
class ClaudeCodeConfig(BaseModel):
    """Configuration for Claude Code CLI as LLM provider."""
    enabled: bool = Field(default=False, description="Use Claude Code CLI instead of LiteLLM.")
    path: str = Field(default="claude", description="Path to claude CLI binary.")
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Glob", "Grep", "Agent", "WebSearch", "WebFetch"],
        description="Tools to pre-approve via --allowedTools.",
    )
    permission_mode: str | None = Field(
        default=None,
        description="Permission mode: default, acceptEdits, bypassPermissions.",
    )
    max_turns: int | None = Field(default=None, description="Max agentic turns per invocation.")
    max_budget_usd: float | None = Field(default=None, description="Max cost per invocation.")
    max_concurrent: int = Field(
        default=3, ge=1,
        description="Max concurrent Claude Code subprocesses. Prevents resource exhaustion.",
    )
    timeout_seconds: int = Field(
        default=300, ge=10,
        description="Subprocess timeout. Process killed after this duration.",
    )
    working_directory: str | None = Field(
        default=None,
        description="Working directory for Claude Code subprocess. "
        "Affects where Bash, Edit, Read tools operate.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for Claude Code subprocess. "
        "Supports ${VAR} substitution.",
    )
```

Add to `RuntimeLLMConfig`:
```python
claude_code: ClaudeCodeConfig = Field(
    default_factory=ClaudeCodeConfig,
    description="Claude Code CLI provider. When enabled, main agent loop uses Claude Code "
    "instead of LiteLLM. SR2 context engineering still applies.",
)
```

### Step 3: Agent integration

**Modify:** `packages/sr2-runtime/src/sr2_runtime/agent.py`

In `__init__()`:
```python
self._use_claude_code = runtime_conf.llm.claude_code.enabled
if self._use_claude_code:
    from sr2_runtime.llm.claude_code import ClaudeCodeProvider
    self._cc_provider = ClaudeCodeProvider(runtime_conf.llm.claude_code)
```

In `_handle_trigger()`, branch after context compilation:
```python
if self._use_claude_code:
    system_context = self._build_system_context(ctx)
    loop_result = await self._cc_provider.stream_complete(
        prompt=str(trigger.input_data),
        system_prompt=system_context,
        stream_callback=trigger.respond_callback if use_streaming else None,
    )
else:
    # Existing LiteLLM path (unchanged)
    ...
```

Returns actual `LoopResult` — session recording, metrics, post-processing all work unchanged downstream.

In `shutdown()`:
```python
if self._use_claude_code:
    await self._cc_provider.shutdown()  # Kills active subprocesses
```

**`fast_complete()`/`embed()` still use LiteLLM** — only the main agent loop switches. Summarization, memory extraction, and embeddings can independently use proxy or direct API.

**MCP limitation:** SR2's registered MCP tools are not available to Claude Code (which has its own MCP config). Document this as a known limitation. Users should configure MCP servers in Claude Code's own config if needed, or pass `--mcp-config` via the `env` config.

### Step 4: Docker support

**Modify:** `Dockerfile` — add Claude Code CLI installation via official bash script:

```dockerfile
FROM python:3.12-slim

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
  && mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
     | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
  && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
     > /etc/apt/sources.list.d/nodesource.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI via official bash script
# (npm install is no longer supported)
RUN curl -fsSL https://claude.ai/install.sh | bash

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock* ./
COPY src/ src/
COPY configs/ configs/
RUN uv sync --all-extras --frozen 2>/dev/null || uv sync --all-extras

EXPOSE 8008
CMD ["uv", "run", "sr2-agent", "configs/agents/edi", "--http"]
```

**New file:** `docker-compose.claude-code.yaml` — full stack with Claude Code + proxy:

```yaml
services:
  sr2-claude-code:
    build: .
    container_name: sr2-claude-code
    restart: unless-stopped
    networks:
      - sr2
    ports:
      - "8741:8008"
    volumes:
      - ./configs:/app/configs
      - ./.dockervols/workspace:/home/workspace       # Files Claude Code can access
      - ${HOME}/.claude:/root/.claude                  # Mount host Claude config (OAuth creds, sessions)
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}
      TELEGRAM_ALLOWED_USERS: ${TELEGRAM_ALLOWED_USERS}
      PROXY_URL: http://cliproxy:9090/v1              # CLIProxyAPI for fast model
    command: >
      uv run sr2-agent configs/agents/claude-code-proxy
      --http --port 8008

  # CLIProxyAPI (OpenAI-compatible proxy for fast model / embeddings)
  # Users bring their own proxy — this is an example.
  # Can also point PROXY_URL at an external host instead.
  cliproxy:
    image: ghcr.io/router-for-me/cliproxyapi:latest
    container_name: cliproxy
    restart: unless-stopped
    networks:
      - sr2
    ports:
      - "9090:9090"
    volumes:
      - ./.dockervols/cliproxy:/root/.cli-proxy-api

  # LibreChat connects to sr2-claude-code:8741/v1/chat/completions
  # Telegram connects via TELEGRAM_BOT_TOKEN

  postgres:
    image: pgvector/pgvector:pg16
    container_name: postgres
    environment:
      POSTGRES_USER: sr2
      POSTGRES_PASSWORD: sr2
      POSTGRES_DB: sr2
    networks:
      - sr2
    volumes:
      - ./.dockervols/postgres/data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sr2"]
      interval: 5s
      timeout: 5s
      retries: 5

networks:
  sr2:
    external: false
```

**Key Docker considerations:**
- **OAuth auth**: Host's `~/.claude/` mounted into container (writable — needed for token refresh). Run `claude` on the host first to authenticate, then the container reuses those credentials.
- **No ANTHROPIC_API_KEY needed**: Claude Code uses the subscription OAuth token. The entire toolset runs through the subscription.
- `/home/workspace` mount gives Claude Code access to project files
- `--bare` flag ensures Claude Code doesn't write CLAUDE.md or auto-memory files (SR2 handles all memory)
- CLIProxyAPI runs as a sibling container, accessible at `http://cliproxy:9090` via Docker network
- LibreChat (on remote server) connects to host port `8741`
- **Healthcheck**: Add `healthcheck: test: ["CMD", "claude", "--version"]` to verify Claude Code auth

### Step 5: Example configs

**New file:** `configs/agents/claude-code/agent.yaml`

```yaml
agent_name: Claude Code Agent
extends: ../../defaults.yaml

system_prompt: |
  You are a helpful AI assistant with full access to the local system.
  You can read files, execute commands, search the web, and manage code.
  Be concise. Use tools proactively to answer questions accurately.

pipeline:
  token_budget: 128000
  compaction:
    rules:
      - type: tool_output
        strategy: schema_and_sample
        max_compacted_tokens: 80
      - type: file_content
        strategy: reference
      - type: code_execution
        strategy: result_summary
        max_output_lines: 3

sessions:
  main_chat:
    max_turns: 200
    idle_timeout_minutes: 120
  _default:
    max_turns: 100
    idle_timeout_minutes: 60

runtime:
  llm:
    claude_code:
      enabled: true
      path: claude
      allowed_tools:
        - Read
        - Glob
        - Grep
        - Agent
        - WebSearch
        - WebFetch
      permission_mode: default
    fast_model:
      name: claude-haiku-4-5-20251001
      max_tokens: 1000
      model_params:
        temperature: 0.1
    model:
      name: claude-sonnet-4-20250514
      stream: true

interfaces:
  telegram_main:
    plugin: telegram
    session:
      name: telegram_{user_id}
      lifecycle: persistent
    token: ${TELEGRAM_BOT_TOKEN}
    allowed_users: ${TELEGRAM_ALLOWED_USERS}

  http_main:
    plugin: http
    session:
      name: "{request.session_id}"
      lifecycle: persistent
```

**New file:** `configs/agents/claude-code-proxy/agent.yaml`

```yaml
# Claude Code + OpenAI-compatible proxy (CLIProxyAPI, etc.)
#
# Main agent: Claude Code CLI (full tools)
# Fast model/embeddings: routed through proxy via api_base
#
# Setup:
#   docker compose -f docker-compose.claude-code.yaml up
#   OR: PROXY_URL=http://localhost:9090/v1 sr2-agent configs/agents/claude-code-proxy --http
agent_name: Claude Code Agent (Proxy)
extends: ../claude-code/agent.yaml

runtime:
  llm:
    claude_code:
      enabled: true
    fast_model:
      name: claude-haiku-4-5-20251001
      api_base: ${PROXY_URL}
      max_tokens: 1000
      model_params:
        temperature: 0.1
    model:
      name: claude-sonnet-4-20250514
      api_base: ${PROXY_URL}
      stream: true
    embedding:
      name: text-embedding-3-small
      api_base: ${PROXY_URL}
```

### Step 6: Exports

**Modify:** `packages/sr2-runtime/src/sr2_runtime/llm/__init__.py`

```python
from sr2_runtime.llm.claude_code import ClaudeCodeProvider, ClaudeCodeConfig, ClaudeCodeResponse
```

### Step 7: Tests

**New file:** `tests/runtime/test_runtime/test_claude_code.py`

**Core provider tests:**
- `test_builds_command` — CLI args from config (--bare, --system-prompt, --allowedTools, etc.)
- `test_builds_command_with_working_directory` — cwd and env passthrough
- `test_stream_parsing_text_deltas` — mock subprocess stdout, verify TextDeltaEvent emitted
- `test_stream_parsing_tool_events` — verify ToolStartEvent/ToolResultEvent from tool_use blocks
- `test_loop_result_populated` — LoopResult has real tool_calls, token counts, iterations
- `test_malformed_json_skipped` — invalid JSON lines logged and skipped, not crash
- `test_stderr_captured` — stderr content logged

**Error handling tests:**
- `test_cli_not_installed` — FileNotFoundError at init when binary missing
- `test_subprocess_timeout` — process killed after timeout_seconds, error LoopResult returned
- `test_nonzero_exit` — stderr included in error, graceful LoopResult
- `test_auth_failure` — recognizable error message for OAuth expiry

**Concurrency tests:**
- `test_semaphore_limits_concurrent` — requests queue when at max_concurrent
- `test_shutdown_kills_active` — shutdown() terminates running subprocesses

**Agent integration tests:**
- `test_agent_routes_to_claude_code` — _handle_trigger takes CC path when enabled
- `test_agent_litellm_unchanged` — existing path works when disabled
- `test_agent_shutdown_cleanup` — Agent.shutdown() calls provider.shutdown()

**Config tests:**
- `test_config_defaults` — default values correct
- `test_config_proxy_api_base` — LLMModelConfig with api_base works

### Step 8: Documentation

**Modify:** `CLAUDE.md` — add:

```markdown
## Claude Code Provider

SR2-runtime can use Claude Code CLI as the main LLM provider:

    sr2-agent configs/agents/claude-code --http --port 8741

## OpenAI-Compatible Proxy

Route SR2's internal LLM tasks through any OpenAI-compatible proxy by
setting `api_base` on model configs:

    runtime:
      llm:
        fast_model:
          name: claude-haiku-4-5-20251001
          api_base: http://localhost:9090/v1

## Docker Deployment (Claude Code)

    docker compose -f docker-compose.claude-code.yaml up
```

## Files Summary

### Create

| File | Purpose |
|------|---------|
| `packages/sr2-runtime/src/sr2_runtime/llm/claude_code.py` | Claude Code CLI provider |
| `configs/agents/claude-code/agent.yaml` | Agent config: Claude Code + direct API |
| `configs/agents/claude-code-proxy/agent.yaml` | Agent config: Claude Code + proxy for fast model |
| `docker-compose.claude-code.yaml` | Docker stack: SR2 + Claude Code + proxy + Postgres |
| `tests/runtime/test_runtime/test_claude_code.py` | Unit tests |

### Modify

| File | Change |
|------|--------|
| `Dockerfile` | Add Claude Code CLI install via `curl ... install.sh` |
| `packages/sr2-runtime/src/sr2_runtime/config.py` | Add `ClaudeCodeConfig`, extend `RuntimeLLMConfig` |
| `packages/sr2-runtime/src/sr2_runtime/agent.py` | Add CC execution path, `_claude_code_run()` |
| `packages/sr2-runtime/src/sr2_runtime/llm/__init__.py` | Export new classes |
| `CLAUDE.md` | Document CC provider, proxy pattern, Docker |

## Patterns Followed

- Protocol-based, not ABC
- Pydantic config with `Field(description=...)`
- Async-first: `asyncio.create_subprocess_exec`
- Structured logging: `logging.getLogger(__name__)`
- Graceful degradation on failure
- Type hints on all public signatures
- Config-driven via YAML
- `@dataclass` for value objects
- ruff: 100-char lines, Python 3.12 target

## Verification

1. `ruff check packages/ && ruff format --check packages/`
2. `pytest tests/runtime/test_runtime/test_claude_code.py -v`
3. `pytest tests/ --ignore=tests/integration/ -v`
4. Manual:
   ```bash
   # Bare metal
   sr2-agent configs/agents/claude-code --http --port 8741
   curl -N localhost:8741/v1/chat/completions \
     -d '{"model":"claude-code","messages":[{"role":"user","content":"ls /tmp"}],"stream":true}'

   # Docker
   docker compose -f docker-compose.claude-code.yaml up
   curl -N localhost:8741/v1/chat/completions \
     -d '{"model":"claude-code","messages":[{"role":"user","content":"ls /home/workspace"}],"stream":true}'
   ```

## Branch

`feat/claude-code-provider` from `main`
