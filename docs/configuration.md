# SR2 Configuration Reference

Auto-generated from Pydantic models. Single source of truth.

## Config Inheritance

```
configs/defaults.yaml          ← Library defaults (all fields have defaults)
  └── agent.yaml               ← Agent overrides (only specify what differs)
       └── interfaces/x.yaml   ← Interface overrides (only specify what differs)
```

Resolution: deep merge, more specific wins. `extends: defaults` or `extends: agent`.

## Pipeline Config

### Top-Level

Model: `PipelineConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `extends` | string \| null | `null` |  | Parent config to inherit from |
| `system_prompt` | string \| null | `null` |  | Override the agent-level system prompt for this interface/pipeline. |
| `token_budget` | integer | `32000` |  | Total token budget for the context window. The pipeline distributes this budget across layers. Higher values allow more context but increase cost and latency. |
| `pre_rot_threshold` | number | `0.25` |  | Fraction of token budget that triggers pre-emptive context rotation. When remaining budget drops below this fraction, summarization and compaction are triggered proactively to avoid hitting the hard limit. |
| `llm` | any | — |  | Per-interface LLM overrides |
| `kv_cache` | any | — |  | KV-cache optimization settings |
| `compaction` | any | — |  | Content compaction settings |
| `summarization` | any | — |  | Conversation summarization settings |
| `retrieval` | any | — |  | Retrieval-augmented context settings |
| `intent_detection` | any | — |  | Intent detection and topic shift settings |
| `tool_masking` | any | — |  | Tool visibility and masking settings |
| `tool_states` | list[any] | — |  | Tool visibility states. Each state defines allowed/denied tools. |
| `tool_transitions` | list[any] | — |  | Rules for transitioning between tool states. |
| `tool_schema_max_tokens` | integer \| null | `null` |  | Max tokens to allocate for tool schemas. If set, schemas are truncated to fit within this budget. None = no limit (all schemas sent as-is). Truncation strategy: drop descriptions, then drop parameters, then drop entire tools. |
| `degradation` | any | — |  | Circuit breaker and degradation settings |
| `memory` | any | — |  | Memory extraction settings |
| `layers` | list[any] | — |  | Context window layers, ordered from most stable to least stable |

**Example:**
```yaml
extends: null
system_prompt: null
token_budget: 32000
pre_rot_threshold: 0.25
llm: <llm>
kv_cache: <kv_cache>
compaction: <compaction>
summarization: <summarization>
retrieval: <retrieval>
intent_detection: <intent_detection>
tool_masking: <tool_masking>
tool_states: []
tool_transitions: []
tool_schema_max_tokens: null
degradation: <degradation>
memory: <memory>
layers: []
```

### KV-Cache Strategy

Model: `KVCacheConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `strategy` | enum | `"append_only"` |  | KV-cache optimization strategy. 'append_only' keeps a stable prefix for cache reuse. 'maximize_prefix_reuse' reorders layers for maximum prefix hit. 'no_cache_optimization' disables cache-aware ordering. |
| `compaction_timing` | enum | `"post_llm_async"` |  | When to run compaction. 'post_llm_async' runs after LLM response without blocking. 'immediate' runs synchronously before next turn. 'disabled' turns off compaction entirely. |
| `summarization_timing` | enum | `"natural_breakpoint"` |  | When to trigger summarization. 'natural_breakpoint' summarizes at topic shifts or pauses. 'token_threshold' summarizes when token usage exceeds the configured threshold. 'disabled' turns off summarization. |
| `memory_refresh` | enum | `"on_topic_shift"` |  | When to refresh retrieved memories. 'on_topic_shift' refreshes when intent detection flags a topic change. 'every_n_turns' refreshes at a fixed interval. 'session_start_only' loads memories once at session start. 'disabled' turns off memory refresh. |
| `memory_refresh_interval` | integer | `10` |  | Only used if memory_refresh=every_n_turns |

**`strategy` values:** `append_only`, `maximize_prefix_reuse`, `no_cache_optimization`

**`compaction_timing` values:** `post_llm_async`, `immediate`, `disabled`

**`summarization_timing` values:** `natural_breakpoint`, `token_threshold`, `disabled`

**`memory_refresh` values:** `on_topic_shift`, `every_n_turns`, `session_start_only`, `disabled`

**Example:**
```yaml
strategy: append_only
compaction_timing: post_llm_async
summarization_timing: natural_breakpoint
memory_refresh: on_topic_shift
memory_refresh_interval: 10
```

### Compaction

Model: `CompactionConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `enabled` | boolean | `true` |  | Enable compaction. When True, verbose tool outputs and file contents are replaced with compact references. Compacted content can be re-fetched by the agent using just-in-time retrieval tools. |
| `raw_window` | integer | `5` |  | Keep last N turns in full detail |
| `min_content_size` | integer | `100` |  | Don't compact below this token count |
| `cost_gate` | any | — |  | Cache-cost-aware compaction gating. When enabled, each turn is evaluated against prompt caching economics before compaction is applied. |
| `rules` | list[any] | — |  | Compaction rules. Each rule matches a content_type and applies a strategy. Available strategies: schema_and_sample, reference, result_summary, supersede, collapse. |

**Example:**
```yaml
enabled: true
raw_window: 5
min_content_size: 100
cost_gate: <cost_gate>
rules: []
```

#### Compaction Rule

Model: `CompactionRuleConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `type` | string | — | ✅ | Content type to match |
| `strategy` | string | — | ✅ | Compaction strategy name |
| `max_compacted_tokens` | integer | `80` |  | Maximum token count for compacted output. Content is truncated to fit within this budget. Only used by strategies that produce variable-length output (e.g., schema_and_sample). |
| `recovery_hint` | boolean | `false` |  | If True, append a hint explaining how to re-fetch the original content (e.g., 'Re-fetch with [tool_name]'). Helps the agent recover compacted data when needed. |

**Example:**
```yaml
type: <type>
strategy: <strategy>
max_compacted_tokens: 80
recovery_hint: false
```

### Summarization

Model: `SummarizationConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `enabled` | boolean | `true` |  | Enable automatic summarization. When True, older conversation history is summarized to reclaim token budget while preserving key information. |
| `trigger` | enum | `"token_threshold"` |  | What triggers summarization. 'token_threshold' triggers when token usage exceeds the threshold fraction. 'topic_shift' triggers on detected topic changes. 'manual' only triggers via explicit API call. |
| `threshold` | number | `0.75` |  |  |
| `model` | string | `"fast"` |  | Model identifier for summarization |
| `preserve_recent_turns` | integer | `3` |  | Number of most recent turns to exclude from summarization. These turns are always kept in full detail regardless of token pressure. |
| `output_format` | enum | `"structured"` |  | Format for generated summaries. 'structured' produces categorized bullet points (decisions, issues, preferences). 'prose' produces a narrative paragraph. |
| `injection` | enum | `"flat"` |  | How summaries are injected into context. 'flat' inserts the full summary as a single block. 'selective' inserts only the sections relevant to the current topic (requires intent detection). |
| `preserve` | list[string] | — |  | Categories of information to always preserve in summaries. These categories are extracted and retained even under heavy token pressure. |
| `discard` | list[string] | — |  | Categories of information to discard during summarization. Content matching these categories is dropped to save tokens. |
| `compacted_max_tokens` | integer | `6000` |  | Maximum token budget for the compacted zone before summarization triggers. Summarization fires when compacted_tokens > threshold * compacted_max_tokens. For heavy-traffic proxies (e.g. Claude Code), raise to 50000-100000. |

**`trigger` values:** `token_threshold`, `topic_shift`, `manual`

**`output_format` values:** `structured`, `prose`

**`injection` values:** `flat`, `selective`

**Example:**
```yaml
enabled: true
trigger: token_threshold
threshold: 0.75
model: fast
preserve_recent_turns: 3
output_format: structured
injection: flat
preserve: []
discard: []
compacted_max_tokens: 6000
```

### Retrieval

Model: `RetrievalConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `enabled` | boolean | `true` |  | Enable retrieval-augmented context. When True, relevant memories and documents are retrieved and injected into the context window. |
| `strategy` | enum | `"hybrid"` |  | Retrieval strategy. 'hybrid' combines semantic and keyword search. 'semantic' uses embedding similarity only. 'keyword' uses BM25/keyword matching. 'scoped' restricts retrieval to the current topic scope. |
| `top_k` | integer | `10` |  | Maximum number of retrieved items to include in context. |
| `max_tokens` | integer | `4000` |  | Maximum total tokens allocated for retrieved content. Results are truncated to fit within this budget. |

**`strategy` values:** `hybrid`, `semantic`, `keyword`, `scoped`

**Example:**
```yaml
enabled: true
strategy: hybrid
top_k: 10
max_tokens: 4000
```

### Intent Detection

Model: `IntentDetectionConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `enabled` | boolean | `true` |  | Enable intent detection. When True, user messages are classified to detect topic shifts, which can trigger memory refresh and selective summarization. |
| `model` | string | `"fast"` |  |  |

**Example:**
```yaml
enabled: true
model: fast
```

### Tool Masking

Model: `ToolMaskingConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `strategy` | enum | `"allowed_list"` |  | Tool masking strategy. 'allowed_list' only exposes allowed tools. 'prefill' uses assistant prefill to guide tool selection. 'logit_mask' applies logit bias to suppress disallowed tools. 'none' exposes all tools. |
| `initial_state` | string | `"default"` |  | Initial tool state name. References a named state in the tool masking configuration that defines which tools are available at conversation start. |

**`strategy` values:** `prefill`, `allowed_list`, `logit_mask`, `none`

**Example:**
```yaml
strategy: allowed_list
initial_state: default
```

### Per-Interface LLM Override

Model: `LLMConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `model` | any \| null | `null` |  | Main model override. |
| `fast_model` | any \| null | `null` |  | Fast model override. |
| `embedding` | any \| null | `null` |  | Embedding model override. |

**Example:**
```yaml
model: null
fast_model: null
embedding: null
```

#### Per-Interface Model Override

Model: `LLMModelOverride`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | string \| null | `null` |  | Model identifier override. |
| `api_base` | string \| null | `null` |  | API base URL override. |
| `max_tokens` | integer \| null | `null` |  | Max tokens per response override. |
| `model_params` | object \| null | `null` |  | Sampling parameter overrides (temperature, top_p, etc.). |

**Example:**
```yaml
name: null
api_base: null
max_tokens: null
model_params: null
```

### Degradation

Model: `DegradationConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `circuit_breaker_threshold` | integer | `3` |  | Number of consecutive failures before the circuit breaker opens. When open, requests are short-circuited to prevent cascading failures. |
| `circuit_breaker_cooldown_minutes` | integer | `5` |  | Minutes to wait before retrying after the circuit breaker opens. After this cooldown, the circuit enters half-open state and allows a trial request. |

**Example:**
```yaml
circuit_breaker_threshold: 3
circuit_breaker_cooldown_minutes: 5
```

### Layers

Model: `LayerConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | string | — | ✅ | Layer name |
| `cache_policy` | string | `"immutable"` |  | Cache policy name |
| `contents` | list[any] | — | ✅ | Content items in this layer |

**Example:**
```yaml
name: <name>
cache_policy: immutable
contents: []
```

#### Content Item

Model: `ContentItemConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `key` | string | — | ✅ | Unique key for this content item |
| `source` | string | — | ✅ | Content resolver name |
| `max_tokens` | integer \| null | `null` |  | Max tokens for this item |
| `optional` | boolean | `false` |  | If True, skip without error if resolver fails |

**Example:**
```yaml
key: <key>
source: <source>
max_tokens: null
optional: false
```

## Runtime Config

### Top-Level

Model: `RuntimeConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `database` | any | — |  |  |
| `llm` | any | — |  |  |
| `loop` | any | — |  |  |
| `session` | any | — |  | Default session settings. |
| `stream_content` | any | — |  |  |
| `heartbeat` | any | — |  |  |

**Example:**
```yaml
database: <database>
llm: <llm>
loop: <loop>
session: <session>
stream_content: <stream_content>
heartbeat: <heartbeat>
```

### Database

Model: `RuntimeDatabaseConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `url` | string \| null | `null` |  | PostgreSQL connection string. Supports ${VAR} env var substitution. |
| `pool_min` | integer | `2` |  | Minimum connection pool size. |
| `pool_max` | integer | `10` |  | Maximum connection pool size. |

**Example:**
```yaml
url: null
pool_min: 2
pool_max: 10
```

### LLM

Model: `RuntimeLLMConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `model` | any | — |  | Main LLM model configuration. |
| `fast_model` | any | — |  | Fast model for extraction, summarization, intent detection. |
| `embedding` | any | — |  | Embedding model for memory retrieval. |

**Example:**
```yaml
model: <model>
fast_model: <fast_model>
embedding: <embedding>
```

#### LLM Model

Model: `LLMModelConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | string | — | ✅ | Model identifier. |
| `api_base` | string \| null | `null` |  | API base URL. |
| `max_tokens` | integer | `4096` |  | Max tokens per response. |
| `stream` | boolean | `false` |  | Enable streaming for this model. |
| `model_params` | any | — |  | Sampling parameters. |

**Example:**
```yaml
name: <name>
api_base: null
max_tokens: 4096
stream: false
model_params: <model_params>
```

#### Model Parameters

Model: `ModelParams`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `temperature` | number \| null | `null` |  | Sampling temperature (0=deterministic, 2=max randomness). |
| `top_p` | number \| null | `null` |  | Nucleus sampling threshold. |
| `top_k` | integer \| null | `null` |  | Top-k sampling. Not supported by all providers. |
| `frequency_penalty` | number \| null | `null` |  | Penalize frequent tokens (-2 to 2). |
| `presence_penalty` | number \| null | `null` |  | Penalize already-used tokens (-2 to 2). |
| `stop` | array \| null | `null` |  | Stop sequences. |

**Example:**
```yaml
temperature: null
top_p: null
top_k: null
frequency_penalty: null
presence_penalty: null
stop: null
```

### Loop

Model: `RuntimeLoopConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `max_iterations` | integer | `25` |  | Max tool-call loop iterations before stopping. |

**Example:**
```yaml
max_iterations: 25
```

### Session Defaults

Model: `RuntimeSessionConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `max_turns` | integer | `200` |  | Default max turns for sessions not explicitly configured. |
| `idle_timeout_minutes` | integer | `60` |  | Default idle timeout in minutes before session cleanup. |

**Example:**
```yaml
max_turns: 200
idle_timeout_minutes: 60
```

### Stream Content

Model: `StreamContentConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `tool_status` | boolean | `true` |  | Stream tool invocation status. |
| `tool_results` | boolean | `false` |  | Stream tool result content. |

**Example:**
```yaml
tool_status: true
tool_results: false
```

## Interfaces & Plugins

### Interface

Model: `InterfaceConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `plugin` | string | — | ✅ | Plugin type: telegram | timer | http | a2a |
| `session` | any \| null | `null` |  | Session config for this interface. |
| `pipeline` | string \| null | `null` |  | Path to pipeline config (relative to config_dir). |

**Example:**
```yaml
plugin: <plugin>
session: null
pipeline: null
```

### MCP Server

Model: `MCPServerConfig`

| Field | Type | Default | Required | Description |
|---|---|---|---|---|
| `name` | string | — | ✅ | Server name for logging. |
| `url` | string | — | ✅ | Command (stdio) or URL (http). |
| `transport` | enum | `"stdio"` |  | Transport protocol. |
| `tools` | array \| null | `null` |  | Curated tool list. None = all tools. |
| `headers` | object \| null | `null` |  | HTTP headers for http/sse transport (e.g. Authorization). Supports ${VAR} env var substitution. |
| `env` | object \| null | `null` |  | Environment variables for the server process. Supports ${VAR} env var substitution. |
| `args` | array \| null | `null` |  | Additional args for stdio transport. |
| `roots` | array \| null | `null` |  | Root URIs to advertise to the server (e.g. 'file:///home/user/project'). Supports ${VAR} env var substitution. |
| `resources` | array \| null | `null` |  | Resources to auto-discover from this server. |
| `expose_resources_as_tools` | boolean | `false` |  | Register mcp_list_resources and mcp_read_resource as agent tools. |
| `prompts` | array \| null | `null` |  | Prompts to auto-load from this server. |
| `expose_prompts_as_tools` | boolean | `false` |  | Register mcp_get_prompt as an agent tool. |
| `sampling` | any | — |  | Sampling policy for server-initiated LLM requests. |

**`transport` values:** `stdio`, `http`, `sse`

**Example:**
```yaml
name: <name>
url: <url>
transport: stdio
tools: null
headers: null
env: null
args: null
roots: null
resources: null
expose_resources_as_tools: false
prompts: null
expose_prompts_as_tools: false
sampling: <sampling>
```

## Reference

### Compaction Rules Reference

Each rule matches a `content_type` on conversation turns and replaces verbose content with a compact reference.

| Strategy | Content Type | What It Does | Recoverable |
|---|---|---|---|
| `schema_and_sample` | `tool_output` | Replaces with line count + first 3 lines as sample | Yes — re-call the tool |
| `reference` | `file_content` | Replaces with file path + metadata (lines, language, size) | Yes — `read_file(path)` |
| `result_summary` | `code_execution` | Replaces with exit code (✓/✗) + first N lines of output | Yes — read result file |
| `supersede` | `redundant_fetch` | Marks as `(superseded by turn N)` | Already in context from later fetch |
| `collapse` | `confirmation` | Collapses to `→ ✓ tool_name(args)` one-liner | Original action already completed |

**Options per rule:**

```yaml
compaction:
  rules:
    - type: tool_output
      strategy: schema_and_sample
      max_compacted_tokens: 80    # Max tokens for the compacted output
      recovery_hint: true          # Adds 'Re-fetch with [tool]' hint

    - type: file_content
      strategy: reference
      include_metadata:            # Which metadata to include in the reference
        - line_count
        - language
        - size

    - type: code_execution
      strategy: result_summary
      max_output_lines: 3          # How many stdout lines to keep

    - type: redundant_fetch
      strategy: supersede
      # No options — metadata.superseded_by_turn provides the turn number

    - type: confirmation
      strategy: collapse
      # No options — metadata.tool_name and metadata.args_summary used
```

**What does NOT get compacted:**
- User messages (always preserved in full)
- Agent reasoning / decisions (can't be re-derived)
- Error messages (critical for debugging)
- Content below `min_content_size` tokens
- Turns in the `raw_window` (last N turns kept in full detail)

### Cache Policies Reference

Each layer declares a cache policy that controls when its content is recomputed.

| Policy | Behavior | Best For |
|---|---|---|
| `immutable` | Never recompute after first call | System prompt, tool definitions, persona |
| `refresh_on_topic_shift` | Recompute when intent detection flags a topic change | Retrieved memories, contextual data |
| `refresh_on_state_change` | Recompute when state hash changes | Last known state (heartbeat polling) |
| `append_only` | Always recompute (content grows each turn) | Conversation history |
| `always_new` | Always recompute (changes every invocation) | Timestamps, dynamic instructions |
| `per_invocation` | Always recompute, no caching between calls | A2A task input, event payloads |
| `template_reuse` | Same as immutable — reuse across calls of this type | Sub-agent system prompts |

**KV-Cache impact:** Layers earlier in the config (top) form the cached prefix.
If an early layer changes, everything after it is re-computed by the LLM.
Order layers from most stable to least stable.

### Session Lifecycles

| Lifecycle | Behavior | Persisted | Use Case |
|---|---|---|---|
| `persistent` | Survives across triggers. Compaction/summarization apply. | ✅ PostgreSQL | User conversations |
| `ephemeral` | Fresh per trigger. Destroyed after processing. | ❌ In-memory | Heartbeats, A2A calls |
| `rolling` | Persistent but capped at `max_turns`. Oldest dropped. | ✅ PostgreSQL | Monitoring, log watchers |

