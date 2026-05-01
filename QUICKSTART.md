# SR2 Quick Start

Get SR2 running in 5 minutes.

## Prerequisites

- Python 3.12+
- pip or uv

## Install

```bash
# From source (development)
git clone <repo>
cd sr2
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Or when published:
pip install sr2
```

## Minimal pipeline

The simplest SR2 config — system prompt + conversation history, no memory or reduction:

```yaml
# config.yaml
layers:
  - name: system_prompt
    cache: static

  - name: conversation
    window: 20
    max_tokens: 10000
    session_history:
      max_tokens: 10000
```

```python
from sr2 import SR2, TurnResult, TokenUsage

# Initialize with config file or dict
sr2 = SR2(config="config.yaml")

# Pre-LLM: compile context
context = await sr2.process(inputs={"session_id": "abc123"})
print(context.to_text())      # assembled context
print(context.total_tokens)   # token count
print(context.metrics)        # what SR2 did

# Make the LLM call (harness responsibility)
# ...

# Post-LLM: process turn result
turn = TurnResult(
    role="assistant",
    content="The answer is 42.",
    token_usage=TokenUsage(prompt_tokens=500, completion_tokens=50),
)
result = await sr2.post_process(turn)
print(result.metrics)  # extraction results, maintenance actions
```

## Full-featured agent

With memory, compaction, summarization, and circuit breakers:

```yaml
# full.yaml
layers:
  - name: system_prompt
    cache: static

  - name: project_context
    cache: static
    max_tokens: 2000
    memory:
      read:
        scope: [project, shared]
      write: false

  - name: tools
    max_tokens: 3000
    priority: 50

  - name: summary
    max_tokens: 1000
    cache: ephemeral
    summarization:
      scope: [conversation]
      preserve: [decisions, errors, preferences]

  - name: conversation
    window: 10
    max_tokens: 8000
    priority: 90
    session_history:
      max_tokens: 8000
    memory:
      read:
        scope: [private]
        max_tokens: 2000
        max_per_turn: 3
      write:
        scope: [private]
    compaction:
      rules: [schema_and_sample, result_summary]
```

```python
sr2 = SR2(config="full.yaml")
context = await sr2.process(inputs={"session_id": "agent-session-1"})
```

## Config as dict

No YAML file needed — pass a dict directly:

```python
config = {
    "layers": [
        {"name": "system_prompt", "cache": "static"},
        {"name": "conversation", "window": 15, "max_tokens": 8000,
         "session_history": {"max_tokens": 8000}},
    ]
}
sr2 = SR2(config=config)
```

## Config inheritance

Share base configs across agents with `extends`:

```yaml
# base.yaml
layers:
  - name: system_prompt
    cache: static
  - name: conversation
    window: 10
    max_tokens: 8000
```

```yaml
# agent.yaml
extends: base.yaml
layers:
  - name: conversation
    window: 20          # override just what's different
    max_tokens: 10000
```

```python
sr2 = SR2(config="agent.yaml")  # merges base + agent
```

## Harness integration pattern

The standard flow for any harness:

```
Harness                        SR2                        LLM
  |                               |                          |
  |-- process(config, inputs) -->|                          |
  |                               |                          |
  |<-- CompiledContext ----------|                          |
  |                               |                          |
  |-------------------------------|--> LLM call ------------>|
  |                               |<-- response -------------|
  |                               |                          |
  |-- post_process(turn_result) ->|                          |
  |<-- PostProcessResult ---------|                          |
  |                               |                          |
```

Key points:
- SR2 does NOT make the LLM call
- Harness translates LLM response into SR2's `TurnResult` schema
- Provider translation is a harness concern — SR2 never sees Anthropic/OpenAI objects

## Next steps

- **README.md** — Architecture overview and design principles
- **CONTRIBUTING.md** — Development workflow and code standards
- **PLAN-sr2-v2-redesign.md** — Full redesign specification (in `/data/obsidian/projects/sr2/`)
