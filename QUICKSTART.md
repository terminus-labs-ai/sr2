# SR2 Quick Start

Get SR2 running in a few minutes. For the conceptual model (why SR2 owns the
turn loop), see [README.md](README.md).

## Prerequisites

- Python 3.12+
- `uv` (or pip)
- An LLM provider key (e.g. `ANTHROPIC_API_KEY`)

## Install

```bash
git clone <repo>
cd sr2
uv sync            # installs sr2 + dev deps into .venv
source .venv/bin/activate
```

## Run a config from the CLI

The fastest path. `configs/minimal.yaml` ships with the repo:

```yaml
# configs/minimal.yaml
models:
  default:
    model: anthropic/claude-haiku-4-5-20251001   # falls back to ANTHROPIC_API_KEY

pipeline:
  token_budget: 200000
  layers:
    - name: system
      target: system
      resolvers:
        - type: static
          config:
            text: "You are a helpful assistant."

    - name: conversation
      target: messages
      resolvers:
        - type: session   # prior turn history (default events: user_input + assistant_response)
        - type: input     # current user message
```

```bash
sr2 configs/minimal.yaml      # starts an interactive chat loop
```

**Config shape:** the YAML has two top-level keys. `models` becomes one
`LiteLLMCallable` per entry (the `default` key is the driver). `pipeline` is
passed straight to `PipelineConfig(**pipeline)`.

> `target` is **required** on every layer — `system`, `messages`, or `tools`.

## Use SR2 as a library

SR2 owns the turn loop; you inject the LLM client and a token counter.

```python
import asyncio
from sr2 import SR2
from sr2.config.models import PipelineConfig
from sr2.integrations.litellm import LiteLLMCallable
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.models import TextBlock

config = PipelineConfig(
    layers=[
        {"name": "system", "target": "system",
         "resolvers": [{"type": "static",
                        "config": {"text": "You are a helpful assistant."}}]},
        {"name": "conversation", "target": "messages",
         "resolvers": [{"type": "session"}, {"type": "input"}]},
    ]
)

sr2 = SR2(
    config,
    llm=LiteLLMCallable("anthropic/claude-haiku-4-5-20251001"),
    token_counter=CharacterTokenCounter(),
)

async def main():
    async for event in sr2.turn([TextBlock(text="What is 2 + 2?")]):
        if event.type == "text":
            print(event.text, end="")
    print()

asyncio.run(main())
```

`turn()` takes a list of content blocks (the user input) and returns an async
stream of `StreamEvent`s.

## Handling the stream

`turn()` yields several event types. Most callers only need `text` and `end`:

| `event.type` | Meaning |
|---|---|
| `text` | A chunk of assistant text (`event.text`) |
| `usage` | Token usage for the turn (`event.usage`) |
| `tool_use_emitted` | The model requested tools this iteration (`event.tool_uses`) |
| `tool_result_received` | Tool results came back (`event.tool_results`) |
| `iteration_complete` | One tool-loop iteration finished (`event.iteration`) |
| `end` | Turn complete — emitted exactly once |

```python
async for event in sr2.turn(user_input):
    if event.type == "text":
        buffer.append(event.text)
    elif event.type == "end":
        break
```

## Multi-turn conversations

Reuse the same `SR2` instance. The `session` resolver accumulates history across
turns automatically — it captures both your input and the assistant's response:

```python
async for _ in sr2.turn([TextBlock(text="My name is Diego.")]):
    ...
async for event in sr2.turn([TextBlock(text="What's my name?")]):
    if event.type == "text":
        print(event.text, end="")   # → "Your name is Diego."
```

### Seeding prior history

For stateless callers (or the gateway/relay path) that already hold a
conversation, pre-populate history before the first `turn()`:

```python
from sr2.models import Message, TextBlock

sr2.seed_session([
    Message(role="user", content=[TextBlock(text="Earlier question")]),
    Message(role="assistant", content=[TextBlock(text="Earlier answer")]),
])
```

`seed_session` overwrites the history in every `SessionResolver`. No LLM calls.

## Adding tools

SR2 drives the tool loop but **never executes tools** — you inject an executor:

```python
from sr2.models import ToolResultBlock

async def tool_executor(block):          # block: ToolUseBlock (id, name, input)
    if block.name == "get_weather":
        result = await fetch_weather(**block.input)
        return ToolResultBlock(tool_use_id=block.id, content=result)
    return ToolResultBlock(tool_use_id=block.id, content="unknown tool",
                           is_error=True)

sr2 = SR2(config, llm=..., token_counter=..., tool_executor=tool_executor)
```

When the model requests tools, SR2 calls `tool_executor` for each (concurrently,
order-preserving), feeds results back, and loops — up to
`max_tool_iterations` (default 25). If tools are requested with no executor
configured, SR2 raises `ConfigError`.

**Getting tool *definitions* to the model** is the other half: tools reach the
model from layers with `target: tools` via a `ToolProvider`. There is no
built-in tool provider yet — register a custom one via the `sr2.tool_providers`
entry point (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Config as a dict

`PipelineConfig` is a plain Pydantic model — build it however you like:

```python
config = PipelineConfig(**{
    "token_budget": 100_000,
    "layers": [
        {"name": "system", "target": "system",
         "resolvers": [{"type": "static", "config": {"text": "..."}}]},
        {"name": "conversation", "target": "messages",
         "resolvers": [{"type": "session"}, {"type": "input"}]},
    ],
})
```

## Token counters

- `CharacterTokenCounter` (`sr2.pipeline.token_counting`) — `chars // 4`,
  zero-dep, fine for development and the CLI default.
- `TiktokenTokenCounter` (`sr2.tokenization.counting`) — accurate
  (`cl100k_base`), requires `tiktoken`.

## Next steps

- [README.md](README.md) — architecture and the three-tier model
- [CLAUDE.md](CLAUDE.md) — full module map and subsystem reference
- [CONTRIBUTING.md](CONTRIBUTING.md) — writing resolvers, transformers, providers
