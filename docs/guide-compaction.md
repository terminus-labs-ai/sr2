# Compaction

As conversations grow, tool outputs and file contents eat your token budget. Compaction replaces verbose content with compact summaries while preserving enough context for the agent to re-fetch anything it needs.

## The Three-Zone Model

SR2 manages conversation history in three zones:

```
  ┌─────────────────────────────┐
  │     Summarized Zone         │  Oldest context, LLM-generated summary
  ├─────────────────────────────┤
  │     Compacted Zone          │  Tool outputs → references, file content → paths
  ├─────────────────────────────┤
  │     Raw Zone (last N turns) │  Full verbatim content, nothing removed
  └─────────────────────────────┘
```

- **Raw zone**: The last `raw_window` turns (default: 5). Never touched by compaction.
- **Compacted zone**: Turns older than the raw window. Compaction rules replace verbose content with compact references.
- **Summarized zone**: When the compacted zone grows too large, summarization produces a structured summary and clears the compacted zone.

This means your agent always has full detail for recent turns, compact references for medium-age turns, and a summary for ancient context.

## Configuration

```yaml
compaction:
  enabled: true
  raw_window: 5          # keep last 5 turns verbatim
  min_content_size: 100  # don't compact content under 100 tokens
  rules:
    - type: tool_output
      strategy: schema_and_sample
      max_compacted_tokens: 80
      recovery_hint: true
    - type: file_content
      strategy: reference
      recovery_hint: true
    - type: code_execution
      strategy: result_summary
```

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Master switch for the compaction system |
| `raw_window` | `5` | Number of most recent turns kept in full detail |
| `min_content_size` | `100` | Minimum token count before compaction kicks in (estimated as `len(content) // 4`) |
| `rules` | `[]` | List of compaction rules mapping content types to strategies |

### Rule Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `type` | yes | — | Content type to match (e.g., `tool_output`, `file_content`, `code_execution`) |
| `strategy` | yes | — | Which compaction strategy to apply |
| `max_compacted_tokens` | no | `80` | Maximum tokens in the compacted output |
| `recovery_hint` | no | `false` | Append a hint showing how the agent can re-fetch the original content |

## Strategies

### `schema_and_sample`

Best for: tool outputs with many lines (API responses, search results, directory listings).

Keeps the first few lines as a sample plus a total line count.

**Before (350 tokens):**
```
[Tool: search_files]
src/sr2/pipeline/engine.py:14: class PipelineEngine:
src/sr2/pipeline/engine.py:28:     async def compile(self, config, ctx):
src/sr2/pipeline/engine.py:45:         for layer in config.layers:
src/sr2/pipeline/engine.py:52:             resolved = await self._resolve_layer(layer, ctx)
src/sr2/pipeline/engine.py:67:     async def _resolve_layer(self, layer, ctx):
... (20 more lines)
```

**After (~30 tokens):**
```
-> 25 lines. Sample: src/sr2/pipeline/engine.py:14: class PipelineEngine:, src/sr2/pipeline/engine.py:28:     async def compile...
```

### `reference`

Best for: file contents that were read into the context.

Replaces the full file with a path reference and metadata.

**Before (800 tokens):**
```
[Tool: read_file]
"""SR2 facade — single entry point for the runtime..."""
from __future__ import annotations
import logging
import os
... (200 lines of Python)
```

**After (~15 tokens):**
```
-> Saved to /src/sr2/sr2.py (200 lines, python, 8.5KB)
```

With `recovery_hint: true`:
```
-> Saved to /src/sr2/sr2.py (200 lines, python, 8.5KB)
Recovery: read_file("/src/sr2/sr2.py")
```

### `result_summary`

Best for: code execution results, shell command outputs.

Keeps the exit code and first N output lines.

**Before (200 tokens):**
```
[Tool: run_command]
Exit code: 0
===== test session starts =====
collected 688 items
tests/test_config/test_validation.py::test_valid_config PASSED
tests/test_config/test_validation.py::test_empty_layers PASSED
tests/test_config/test_validation.py::test_budget PASSED
... (50 more lines)
===== 688 passed in 12.34s =====
```

**After (~20 tokens):**
```
-> ✓ Exit 0. ===== test session starts =====... (50 more lines)
```

A non-zero exit code shows as `✗ Exit 1`.

### `supersede`

Best for: duplicate or redundant fetches of the same data.

When the same content is fetched multiple times, earlier fetches are marked as superseded.

**Before:** Full content from turn 2
**After:** `-> (superseded by turn 5)`

### `collapse`

Best for: simple tool confirmations (file written, command acknowledged).

Reduces to a one-liner with tool name and arguments.

**Before (40 tokens):**
```
[Tool: write_file]
Successfully wrote 150 lines to /src/main.py
File saved.
```

**After (~10 tokens):**
```
-> ✓ write_file(main.py)
```

## Recovery Hints

When `recovery_hint: true`, the compacted output includes a line showing how the agent can get the original content back:

```
-> Saved to /src/sr2/sr2.py (200 lines, python, 8.5KB)
Recovery: read_file("/src/sr2/sr2.py")
```

The agent sees this and can call the tool to re-fetch the full content if needed. This is the key insight: compaction doesn't lose information, it just moves it from "always in context" to "available on demand."

## How Compaction Runs

### Decision Rules (in order)

1. **Never compact user messages** — user turns are always preserved verbatim
2. **Never compact turns in the raw window** — last N turns stay full
3. **Never re-compact** — already-compacted turns are skipped (idempotent)
4. **Skip small content** — below `min_content_size` tokens, not worth compacting
5. **Skip unmatched types** — if a turn's `content_type` has no matching rule, leave it alone
6. **Apply the matching rule** — find the first rule where `type` matches and apply its strategy

### Timing

Compaction runs as post-LLM processing, controlled by the `compaction_timing` setting:

| Timing | Behavior |
|--------|----------|
| `post_llm_async` | Non-blocking, runs after the LLM response is returned (default) |
| `immediate` | Synchronous, blocks the next turn until compaction finishes |
| `disabled` | No compaction |

The default `post_llm_async` is almost always what you want — the agent responds immediately and compaction happens in the background before the next turn.

### Pipeline Integration

Compaction integrates through the `ConversationManager`:

1. New turn added to the raw zone
2. After LLM response, `PostLLMProcessor` triggers compaction
3. `ConversationManager.run_compaction()` processes turns outside the raw window
4. When the compacted zone exceeds a token threshold, summarization kicks in
5. Summarized content moves to the summarized zone, compacted zone clears

## Tuning Tips

**`raw_window`** — Start with 5. Increase if your agent frequently references recent tool outputs. Decrease if you're hitting token budget limits.

**`min_content_size`** — Start with 100. This prevents the engine from wasting effort compacting small messages that are already efficient. Lower it if you want more aggressive compaction.

**Rule ordering** — Rules are matched by `type` field. If a turn matches multiple types, the first matching rule wins. Put more specific types first.

**When to enable recovery hints** — Always enable for `reference` and `schema_and_sample` strategies. The agent can re-fetch files and search results easily. Less useful for `collapse` (the agent already knows what it wrote).
