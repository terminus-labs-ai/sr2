# Examples

Runnable examples demonstrating SR2's core features. No LLM API key or database required.

## Setup

```bash
git clone https://github.com/terminus-labs-ai/sr2.git
cd sr2
pip install -e .
```

## Examples

| # | File | What It Shows |
|---|------|--------------|
| 1 | [01_minimal_pipeline.py](01_minimal_pipeline.py) | Two-layer pipeline, resolver registration, context compilation |
| 2 | [02_multi_turn.py](02_multi_turn.py) | Session history, compaction rules, token budget enforcement |
| 3 | [03_memory.py](03_memory.py) | Memory extraction, conflict detection/resolution, hybrid retrieval |
| 4 | [04_tool_masking.py](04_tool_masking.py) | Tool state machines, transitions, masking strategies |

## Running

```bash
python examples/01_minimal_pipeline.py
python examples/02_multi_turn.py
python examples/03_memory.py
python examples/04_tool_masking.py
```

## Next Steps

These examples use the SR2 library directly. For a complete working agent with an LLM loop, HTTP API, and plugin system, see `configs/agents/edi/` and the [Quick Reference](../docs/reference.md).
