# SR2 Benchmarks

Three benchmarks that measure the value of managing the context window as a structured resource versus naive concatenation.

## Quick Start

```bash
# Token savings — no API key needed
uv run python benchmarks/token_savings.py

# Coherence — requires an LLM API key
OPENAI_API_KEY=sk-... uv run python benchmarks/coherence.py --model gpt-4o-mini

# Cost — requires an LLM API key
OPENAI_API_KEY=sk-... uv run python benchmarks/cost.py --models gpt-4o-mini
```

All three benchmarks use real SR2 library components (compaction engine, conversation manager, pipeline engine, resolver registry). Nothing is stubbed or simulated.

### Running with Ollama

The coherence and cost benchmarks use the OpenAI SDK, which is compatible with Ollama's API. Point it at your local instance:

```bash
# Make sure Ollama is running and has a model pulled
ollama pull llama3.1

# Set environment variables to route the OpenAI SDK to Ollama
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1

# Run benchmarks against a local model
uv run python benchmarks/token_savings.py

uv run python benchmarks/coherence.py --model llama3.1

uv run python benchmarks/cost.py --models llama3.1
```

Any model available via `ollama list` works. Smaller models (e.g., `llama3.1`, `mistral`, `qwen2.5`) run faster; larger ones (e.g., `llama3.1:70b`) give better coherence results.

**Note:** The cost benchmark reports dollar amounts using hardcoded pricing for OpenAI/Anthropic models. With Ollama (free, local inference), the dollar figures won't be meaningful — focus on the token count comparison and savings percentage instead.

---

## Benchmark 1: Token Savings

```bash
uv run python benchmarks/token_savings.py [--turns N] [--raw-window N] [--budget N]
```

**What it does:** Runs a multi-turn coding agent conversation through two tracks in parallel:

- **Naive track** — concatenates system prompt + tool definitions + full history every turn, the way most agents work out of the box.
- **Managed track** — feeds each turn through `ConversationManager` (three-zone compaction) and `PipelineEngine` (layered context compilation).

No LLM calls. Runs in under a second.

**What the results tell you:**

| Metric | What it means |
|---|---|
| Cumulative token savings | Total input tokens saved across the entire session. Directly proportional to API cost. |
| Budget compliance | How many turns exceed the token budget. Naive has no budget concept — it just grows. Managed respects the configured limit. Try `--budget 4000` to see naive blow past at 2.4x while managed stays under. |
| KV-cache prefix hit rate | Percentage of turns where the core + memory layers (the prefix) were byte-identical to the previous turn. 100% means the provider can reuse its KV cache for that prefix on every call. |
| Pipeline overhead | Compile time per turn in microseconds, framed as a percentage of a typical LLM round-trip. Usually <0.02% — negligible. |
| Per-turn table | Shows how naive context grows linearly while managed context stays bounded. Early turns may show managed > naive because managed includes retrieved memories. |
| Compaction events | When tool outputs got compacted (schema-and-sample), showing original vs compacted token counts. |
| Layer distribution | Final turn's context broken down by layer: core (system prompt + tools), memory (retrieved context), conversation (three-zone history). |
| Three-zone breakdown | How the conversation layer splits across raw (recent verbatim turns), compacted (tool outputs reduced to references), and summarized (LLM-digested older history). Summarization shows 0 in this benchmark because there's no LLM — see the coherence benchmark for summarization in action. |
| Growth chart | ASCII visualization of naive vs managed context size over time. Naive grows linearly; managed stays flat. |

---

## Benchmark 2: Coherence

```bash
uv run python benchmarks/coherence.py [--model MODEL] [--turns N] [--budget N]
```

**What it does:** Generates a 50-turn conversation with "anchor decisions" planted at known turns (e.g., "Let's use PostgreSQL" at turn 5, "We'll go with JWT auth" at turn 15). Then builds the final context two ways:

- **Naive** — system prompt + as many recent messages as fit in the budget, dropping oldest first.
- **Managed** — full SR2 pipeline with compaction and LLM-powered summarization.

Sends 5 recall questions to the LLM (e.g., "What database did the team decide to use?") with each context and scores keyword matches.

**What the results tell you:**

| Metric | What it means |
|---|---|
| Three-zone breakdown | Shows the final managed context split across raw, compacted, and summarized zones. Unlike the token savings benchmark, this one actually uses LLM-powered summarization, so you'll see the summarized zone doing real work. |
| HIT / MISS per question | Whether the LLM could recall each anchor decision from the provided context. Naive truncation drops old messages, so early decisions are lost. Managed context preserves them through summarization. |
| Score (e.g., 1/5 vs 5/5) | Total recall accuracy. The gap between naive and managed is the coherence advantage. |
| Information density | Recalls per 1,000 tokens. Managed context packs more relevant information into fewer tokens. A higher density means better use of the context window. |
| KV-cache prefix hit rate | Same as token savings — proves the stable prefix held up even with summarization running. |

**Requires:** `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Without a key, prints a skip message and exits 0. Works with Ollama via `OPENAI_BASE_URL` (see Quick Start above).

**Supported models:** Any OpenAI, Anthropic, or Ollama model. Defaults to `gpt-4o-mini` (OpenAI) or `claude-sonnet` (Anthropic).

---

## Benchmark 3: Cost

```bash
uv run python benchmarks/cost.py [--models MODEL1,MODEL2] [--turns N]
```

**What it does:** Runs the same per-turn comparison as token savings, but instead of estimating tokens locally, sends both naive and managed contexts to a real API with `max_tokens=1` and records `usage.prompt_tokens` (or `usage.input_tokens` for Anthropic) — the actual billed count from the provider's tokenizer.

Sums across all turns to get total session cost.

**What the results tell you:**

| Metric | What it means |
|---|---|
| Naive Input / Managed | Actual billed prompt tokens (cumulative across all turns), not estimates. |
| Saved % | Real cost reduction as measured by the provider's billing. |
| Cost Naive / Cost Managed | Dollar amounts using published per-token pricing. |
| KV-cache prefix hit rate | Prefix stability measured alongside real API calls. |

**Requires:** `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Without a key, prints a skip message and exits 0. Works with Ollama via `OPENAI_BASE_URL` (see Quick Start above).

**Note:** This benchmark makes `2 * num_turns` API calls per model. With the default 30 turns and `max_tokens=1`, cost is negligible (fractions of a cent) but it does take a minute or two to run.

---

## Understanding KV-Cache Prefix Hit Rate

This metric appears in all three benchmarks and deserves explanation.

LLM providers cache the internal computation (key-value pairs) for token sequences they've already processed. When consecutive API calls share the same prefix, the provider can skip recomputing those tokens — the "cache hit" saves both latency and (with some providers) cost.

SR2's layered architecture puts stable content first:

```
[ core: system prompt + tools ][ memory: retrieved context ][ conversation: 3-zone history ]
```

Core and memory rarely change between turns. The conversation layer (the suffix) changes every turn, but that's fine — the prefix is already cached. A 100% prefix hit rate means every single turn reused the cached prefix computation.

Naive concatenation doesn't guarantee any particular ordering, so even small changes to early content invalidate the entire cache.

---

## File Structure

```
benchmarks/
    README.md                    # This file
    token_savings.py             # Benchmark 1: no LLM needed
    coherence.py                 # Benchmark 2: requires LLM
    cost.py                      # Benchmark 3: requires LLM
    context_benchmark.py         # Legacy standalone simulation (deprecated)
    _shared/
        __init__.py              # sys.path setup
        conversation_data.py     # Conversation generation and anchor decisions
        pipeline_factory.py      # Factory to wire up real SR2 components
        reporting.py             # Terminal formatting utilities
```
