# How I stopped my AI agents from getting dumber after 10 turns

TL;DR: Built an open-source context engineering library. 5/5 decision recall vs 3/5 naive on Claude Opus, 52% token reduction, 3.6x information density, 100% cache hit rate. Apache 2.0, `pip install sr2`. Link at the bottom if you want to skip the story.

## The Journey (skip if you don't like storytelling)

Let me preface this — I am in no way an authority on AI agent design (this is literally my first serious project in this space), just a guy with a deep hyperfocus that found an issue and went after trying to sort it out.

I've been building AI agents locally for the past few months — personal assistant, task runner, coding helper, the whole shebang. This started when the whole open-source agent hype exploded. I got excited, tried one of the popular ones, and before I even sent a single message the context was already sitting at 15K+ tokens. Fresh install, no conversation, 15K gone. Then I tried another that looked promising but needed serious elbow grease to set up and was way more complex than what I needed. Stayed with another one for a couple weeks, but they hard-capped context at 30K and if you loaded too many tools into an agent — you guessed it — bye bye context.

Ollama on a 7900 XTX, models running on my CPU when I need the GPU free, nothing cloud, nothing fancy. And at some point I realized that the longer a conversation went, the worse my agents got. Not like gradually worse. Like, noticeably dumber, forgetting things I told them a few turns ago, repeating tool calls they've already made, getting stuck in endless loops where the only solution was to burn the memory and start from scratch — not to mention the added frustration, of course. I've tried several different models, params, all that I could get my hands on, and while it did work to some degree, the core issue was still there.

Every framework I looked at handles tool calling, orchestration, memory — cool. But what actually goes into the LLM's mouth every turn? Everything. Concatenated. System prompt, full conversation history, every tool definition in JSON (which, btw, brackets and quotes and all that JSON formatting consumes context space too) and every result in full, whatever memories got retrieved. Growing and growing until you hit the token limit and start chopping from the top. And that's when things go sideways.

So I spent a lot of time trying to figure out why this was happening, and the answer turned out to be embarrassingly simple: unmanaged context window. Not just that context grows wildly during a conversation, but also a lot of bloat that gets thrown into it from the start and how the context gets chopped off once it hits its limit.

I built a library to fix it. This post is about the problem, how I approached it, and what the numbers look like. If you're building agents and you've been frustrated by this, maybe it helps. If you have a better approach, I genuinely want to hear it.

And just a disclaimer for those thinking "why go through all of this for context management?" — each of us builds differently. I like having data that tells me something is actually working, not just vibes. This process has been a blast and I have tangible results to show for it.

&nbsp;

## The problem (or: why your agent forgets everything)

Here's what's happening under the hood in most agent setups. Every turn in the loop:

1. Your agent gets a user message (or a heartbeat, or whatever trigger)
2. The framework builds a context: system prompt + tools + conversation history + retrieved memories
3. That context gets sent to the LLM
4. LLM responds, maybe calls a tool
5. Tool result gets appended to the conversation
6. Go to step 1

The problem is step 2. That context just keeps growing. And it's not just the messages — tool results are huge. A single `read_file` call can dump 200 lines into your history. A test run? 50+ lines of pytest output. Search results? Pages of JSON. All of that stays in your conversation history forever, taking up space, even though you probably don't need the raw content of a file you read 20 turns ago.

Eventually one of two things happens: you hit your token budget and start truncating from the top (bye bye system prompt, bye bye early context that might actually matter), or you're running locally and your inference just gets slower and slower because you're processing thousands of tokens of stale tool output every single turn.

And here's the part that really bugged me once I understood it: most LLM providers (and even local setups with vLLM or similar) can cache the beginning of your context between turns. If your system prompt and tool definitions are always the same bytes at the front, the provider doesn't need to reprocess them. But the moment you change anything in that prefix — move a tool definition, update a timestamp, modify an old message — that cache gets nuked and you're paying full price again.

So the problem is actually threefold:

1. Context grows unbounded and eventually gets truncated destructively
2. Stale tool outputs* waste tokens every turn
3. Naive context assembly destroys your cache efficiency

_*Stale tool outputs include those things the agent attempted that failed_

&nbsp;

## How I approached it

I spent a while looking for a library that handled this. Couldn't find one that did what I wanted, so I started building. The core idea ended up being pretty simple once I figured it out:

**Treat the context window like zones, not a single blob.**

Instead of one big list of messages, I split conversation history into three zones:

1. **Raw** — the last N turns, kept completely verbatim. Recent context needs to be exact.
2. **Compacted** — older turns where the big stuff (tool outputs, file reads, search results) gets compressed down to a reference. "→ 200 lines. Sample: [first 3 lines]... Recovery: Re-fetch with read_file." The agent can always get the original back if it needs it. Nothing is lost permanently.
3. **Summarized** — the oldest context, where an LLM digests everything down to what actually matters: decisions that were made, things that are still unresolved, preferences the user expressed. Routine "ok done" confirmations and dead-end explorations get dropped.

As turns age, they move through zones automatically. New stuff stays exact. Old stuff gets progressively compressed. Nothing ever just gets chopped off the top.

On top of that, the full context assembly is done through a layered pipeline:

```
Layer 1: core      (system prompt + tools)     → never changes
Layer 2: memory    (retrieved memories)         → rarely changes  
Layer 3: conversation (three-zone history)      → changes every turn
```

Ordered most-stable to least-stable. The stuff that never changes is always first, so it's always the cache prefix. The stuff that changes every turn is last. This is the layout that makes the cache actually work.

Everything declarative, defined in YAML config, no context management code in the agent logic itself.

**Two-model strategy.** The main agent loop uses your big model (whatever you're running for reasoning and tool use), but summarization and other background tasks use a small fast model. On my setup that means the 7900 XTX runs the main model and the CPU handles the small one for summarization. You're not burning expensive inference on compressing old context — that's grunt work for a smaller model. Both are configurable per-interface in the YAML.

**Flexible layers.** What if you don't want conversation history? What if you want it stateless? The layers are fully customizable — pick and choose what goes where. A cron/heartbeat agent can run with just the core layer, no conversation at all. Same library, different config.

&nbsp;

## The numbers

_Yeah, this is nice and all, but it's all talk._ I'm the type that needs to see data to believe something is working, so I built benchmarks that run directly against your LLM provider (LiteLLM SDK powered, so it fits whatever you use). Real multi-turn sessions, real tool calls, real token counts. Side-by-side: same conversation through the SR2 pipeline vs naive concatenation (what most frameworks do). All in the repo for you to run yourself.

### Coherence benchmark (Claude Opus, 50 turns)

This one asks the LLM to recall decisions made earlier in the conversation. The naive approach truncates from the top when it runs out of space. SR2 compacts and summarizes instead.

| Question | Naive | Managed |
|----------|-------|---------|
| What database did the team decide to use? | ❌ MISS | ✅ HIT |
| What authentication method was chosen? | ❌ MISS | ✅ HIT |
| What message queue system was selected? | ✅ HIT | ✅ HIT |
| What frontend framework did the team pick? | ✅ HIT | ✅ HIT |
| What container orchestration tool was decided on? | ✅ HIT | ✅ HIT |
| **Score** | **3/5** | **5/5** |

Naive used 7,122 tokens. Managed used 3,329 tokens. That's **3.6x more information per token**.

The naive approach lost the two earliest decisions because they got truncated when the context grew past the budget. SR2 kept all five because compaction compressed the old tool outputs instead of throwing them away.

### Cost benchmark (Claude Opus, 30 turns)

| | Naive | Managed | Saved |
|---|-------|---------|-------|
| Input tokens | 184,456 | 88,638 | 51.9% |
| Cost | $0.184 | $0.089 | $0.096 |

**52% token reduction, $0.096 saved per session.** Multiply that by however many agent sessions you run per day.

### Token savings benchmark (simulated, 30 turns)

With default configs, nothing tweaked:

- **74% fewer total tokens** processed over 30 turns
- **80% smaller context window** by the final turn (1,938 vs 9,545 tokens)
- **100% KV-cache prefix stability** — core and memory layers had identical bytes every turn
- **25% average compaction** on tool outputs
- **139μs avg compile overhead** — 0.028% of a 500ms LLM call

The growth curve is the thing that convinced me this was worth releasing. Naive context grows linearly — every turn adds more, nothing ever shrinks. Managed context plateaus because compaction and summarization keep it bounded. The longer the conversation goes, the bigger the gap gets.

&nbsp;

## What it is and what it isn't

SR2 is a Python library. Apache 2.0, open source. `pip install sr2` and you get the core: pydantic, pyyaml, litellm, nothing else.

It is NOT a framework. It doesn't own your agent loop, your LLM calls, or your tool execution. You give it a config and some context, it gives you back a compiled string and a token count. That's it. What you do with that is your business. It works with anything through LiteLLM — OpenAI, Anthropic, Ollama, whatever you're running.

The repo does include an optional agent runtime with Telegram, HTTP, and CLI interfaces if you want something batteries-included, but the core library is just the context pipeline.

&nbsp;

## Some things I'm proud of

**Observability out of the box.** I love data. There's no arguing against data. So every pipeline run produces metrics — cache hit rates, compaction ratios, token counts per layer, circuit breaker events, the whole thing. The repo ships with a `docker-compose.yaml` that spins up Prometheus and Grafana pre-wired to the metrics exporter. `docker compose up` and you have dashboards. Not "here's how to set up monitoring" — actual dashboards, ready to go. Because if you can't see what your context pipeline is doing, how do you know it's working?

**Graceful degradation.** Each layer has a circuit breaker. If your retrieval service fails 3 times in a row, the breaker opens and that layer gets skipped. Agent keeps running with reduced context instead of crashing. Core layer is never skipped.

**Per-interface configs.** Same agent, different context strategies per trigger. Telegram chat gets 48K tokens with full compaction. A cron heartbeat gets 3K, stripped down. An API call gets 8K, stateless. All YAML, no code changes. They can even have different memories altogether — your very own multi-persona agent just by creating a few YAML config files.

**Tool state machine.** Instead of adding/removing tools dynamically (which destroys your cache), tool availability is controlled through states and masking. The tool definitions stay stable in the prefix, you just control which ones the model can actually select.

**Config inheritance.** `defaults.yaml` → `agent.yaml` → `interfaces/telegram.yaml`. Deep merge, more specific wins. If you've used Helm, same energy.

&nbsp;

## Long ways to go still

This is only the very first iteration. There's still a lot of fixes and improvements to come. I haven't tested against a bunch of different models yet — starting small with llama and expanding from there (probably glm-4.7-flash next), since model output varies. Community feedback will shape where this goes.

&nbsp;

## Try it

```bash
pip install sr2
```

GitHub: [github.com/terminus-labs-ai/sr2](https://github.com/terminus-labs-ai/sr2)

688 tests, docs for everything, an example agent you can run with Ollama, and the benchmark scripts so you can run them yourself.

This is v0.1.0. I'm one person, it's early. But the pipeline works, the architecture is solid, and I've been running my own agents on it daily.

If you're building agents and have dealt with this problem, I'd love to hear how you approached it. Are you just concatenating and hoping for the best? Built your own solution? Found a library I missed? Let me know.

&nbsp;

*If you know Mass Effect and are wondering — yes, the name is a Mass Effect reference. I name all my agents after characters from the series (EDI, Liara, Tali), judge me, I don't care. It was either this or naming them after Dragon Ball characters and I think I made the right call.*