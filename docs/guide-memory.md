# Memory System

SR2's memory system extracts structured facts from conversations, detects conflicts with existing knowledge, resolves them automatically, and retrieves relevant memories for future context. This gives your agent long-term memory that survives across sessions.

## How It Works

```
  LLM response arrives
        │
        ▼
  ┌──────────────┐
  │  Extraction   │  LLM parses the turn for structured facts
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Conflict     │  Checks new facts against existing memories
  │  Detection    │  (same key, different value?)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Resolution   │  Applies per-type strategy (archive old, keep both, etc.)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Store        │  Saves to InMemoryMemoryStore or PostgresMemoryStore
  └──────────────┘

  Later, on a new turn:

  ┌──────────────┐
  │  Retrieval    │  Hybrid search (keyword + semantic) with dimensional matching
  └──────┬───────┘
         │
         ▼
  Injected into context as a layer
```

The whole pipeline runs as post-LLM processing — after each assistant response, SR2 extracts memories asynchronously. On the next turn, the retrieval resolver pulls relevant memories into the context window.

## Memory Schema

Every memory has these fields:

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Dotted namespace (e.g., `user.employer`, `project.language`) |
| `value` | string | The fact itself |
| `memory_type` | string | Stability classification (see below) |
| `confidence_source` | string | How the fact was learned |
| `stability_score` | float | 0.0–1.0, derived from memory_type |
| `dimensions` | dict | Scoping (e.g., `{"channel": "slack", "project": "sr2"}`) |
| `access_count` | int | How often this memory has been retrieved |
| `last_accessed` | datetime | When it was last retrieved |
| `archived` | bool | Soft-deleted by conflict resolution |
| `conflicts_with` | string | ID of a conflicting memory (for keep-both resolution) |

## Memory Types

Memory types determine how stable a fact is and how conflicts are resolved:

| Type | Stability | Examples | Conflict Strategy |
|------|-----------|----------|-------------------|
| `identity` | 1.0 | Name, birthday, native language | Archive old, keep new |
| `semi_stable` | 0.7 | Employer, city, tech stack | Archive old, keep new |
| `dynamic` | 0.3 | Current mood, today's focus, active task | Delete old, keep new |
| `ephemeral` | 0.0 | Temporary context, this-session-only | Delete old, keep new |

Higher stability = longer retention. The conflict resolver uses these types to decide what happens when a new fact contradicts an existing one.

## Confidence Sources

The extractor tags each memory with how it was learned:

| Source | Score | Example |
|--------|-------|---------|
| `explicit_statement` | 1.0 | "I'm 30 years old" |
| `direct_answer` | 0.9 | Q: "Where do you work?" A: "Anthropic" |
| `contextual_mention` | 0.7 | "...when I was at Google last year..." |
| `inferred` | 0.5 | Deduced from multiple signals |
| `offhand` | 0.3 | "I think I prefer Python" |

## Extraction

The `MemoryExtractor` sends each conversation turn to a fast LLM with a structured prompt. The LLM returns JSON:

```json
[
  {"key": "user.employer", "value": "Anthropic", "memory_type": "semi_stable", "confidence_source": "direct_answer"},
  {"key": "user.language", "value": "Python", "memory_type": "semi_stable", "confidence_source": "contextual_mention"}
]
```

Configuration options:

- **`max_memories_per_turn`** — caps extraction per turn (default: no limit). Prevents a single verbose turn from flooding the store.
- **`key_schema`** — optional list of key prefixes and examples to guide the LLM toward consistent naming:

```python
key_schema = [
    {"prefix": "user.identity", "examples": ["user.identity.name", "user.identity.employer"]},
    {"prefix": "project", "examples": ["project.language", "project.framework"]},
]
```

Items missing `key` or `value` are silently dropped. Invalid `memory_type` values default to `semi_stable` (0.7 stability). Malformed JSON returns an empty result — extraction never crashes the pipeline.

## Conflict Detection

When a new memory is extracted, the `ConflictDetector` checks for existing memories with the same key but a different value:

- Same key, different value (case-insensitive comparison) = **conflict detected**
- Same key, same value = **no conflict** (duplicate, skipped)
- Archived memories are excluded from conflict detection
- A memory never conflicts with itself

Each conflict includes:
- `conflict_type`: currently `key_match` (future: semantic similarity)
- `confidence`: 1.0 for exact key match
- References to both the existing and new memory

## Conflict Resolution

The `ConflictResolver` applies a strategy based on the memory type:

| Memory Type | Default Strategy | What Happens |
|-------------|-----------------|--------------|
| `identity` | `latest_wins_archive` | New memory wins. Old memory is archived (soft-deleted, still queryable). |
| `semi_stable` | `latest_wins_archive` | Same as identity. |
| `dynamic` | `latest_wins_discard` | New memory wins. Old memory is permanently deleted. |

You can override per-type:

```python
resolver = ConflictResolver(
    store=store,
    strategies={"identity": "keep_both"},
)
```

The `keep_both` strategy tags both memories with `conflicts_with` pointing to each other. Useful when you want the agent to acknowledge contradictions rather than silently overwrite.

## Retrieval

The `HybridRetriever` finds relevant memories for the current conversation context.

### Strategies

| Strategy | How It Works |
|----------|-------------|
| `hybrid` | Combines keyword + semantic results, deduplicates, merges scores |
| `keyword` | Full-text search across memory keys and values |
| `semantic` | Embedding-based cosine similarity (requires an embedding callable) |

### Scoring Boosts

Retrieved memories are scored and ranked. Two optional boosts adjust the ranking:

- **Recency boost** (`recency_weight`): Recently accessed memories score higher. A memory accessed today scores higher than one from 60 days ago.
- **Frequency boost** (`frequency_weight`): Frequently accessed memories score higher. A memory retrieved 100 times outranks one retrieved once.

Both weights default to 0. Set them to 0.3–0.5 for noticeable effect.

### Caps

- **`top_k`** — max number of memories returned (default: 10)
- **`max_tokens`** — token budget for retrieved content. Results are trimmed to fit.

### Result Format

Retrieved memories are injected into context as:

```xml
<retrieved_memories>
  [user.employer] Anthropic
  [user.language] Python
  [project.framework] FastAPI
</retrieved_memories>
```

## Dimensional Matching

Memories can be scoped to dimensions like `channel`, `project`, or `team`. When retrieving, the `DimensionalMatcher` adjusts scores based on whether the memory's dimensions match the current context.

### Strategies

| Strategy | Behavior |
|----------|----------|
| `best_fit` | Matching dimensions get a 1.2x boost. Mismatching get a 0.3x penalty. Unscoped memories get a 0.9x penalty (small). |
| `exact` | Only returns memories that match ALL current dimensions. Strict filtering. |
| `fallback_to_generic` | Tries exact match first. If no matches, falls back to unscoped memories. |

Example: if the current context has `{"channel": "slack"}`:

- A memory with `{"channel": "slack"}` gets boosted (best_fit) or included (exact)
- A memory with `{"channel": "email"}` gets heavily penalized (best_fit) or excluded (exact)
- An unscoped memory (no dimensions) gets a small penalty (best_fit) or excluded (exact) / included (fallback_to_generic)

## Storage Backends

### InMemoryMemoryStore

Default. All memories live in a Python dict. Fast, no dependencies, but lost on restart.

### PostgresMemoryStore

Persistent storage using PostgreSQL. Requires `asyncpg`:

```bash
pip install -e ".[postgres]"
```

Set up via the SR2 facade:

```python
sr2 = SR2(config)
await sr2.set_postgres_store(asyncpg_pool)  # creates tables automatically
```

## Pipeline Integration

Memory retrieval fits into the pipeline as a layer:

```yaml
layers:
  - name: core
    cache_policy: immutable
    contents:
      - key: system_prompt
        source: config

  - name: memory
    cache_policy: append_only
    contents:
      - key: ltm_memories
        source: retrieval    # <- HybridRetriever resolves this
        optional: true

  - name: conversation
    cache_policy: append_only
    contents:
      - key: session_history
        source: session
```

The retrieval resolver uses the current trigger input as the search query, calls the retriever, optionally applies dimensional matching, and formats the results.

### Retrieval Config

```yaml
retrieval:
  enabled: true
  strategy: hybrid     # hybrid | keyword | semantic
  top_k: 10
  max_tokens: 4000
```

### Refresh Timing

Controls when memories are re-retrieved:

| Option | Behavior |
|--------|----------|
| `on_topic_shift` | Re-retrieve when intent detection signals a topic change (default) |
| `every_n_turns` | Re-retrieve every N turns |
| `session_start_only` | Retrieve once at session start |
| `disabled` | No retrieval |
