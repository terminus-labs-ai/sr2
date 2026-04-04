# SR2 Pro

SR2 Pro extends the core library with production-grade backends, observability, and operational tooling.

## Install

```bash
pip install sr2-pro
```

SR2 Pro is a drop-in extension. Import and register the backends — everything else works the same.

## Features

### PostgreSQL + pgvector Memory Backend

Production memory storage with vector similarity search. Replaces the default SQLite backend for multi-instance deployments where agents need shared, persistent memory with semantic retrieval.

### OpenTelemetry Export

Push pipeline traces and spans to any OpenTelemetry-compatible collector. Every pipeline stage (resolve, compact, summarize, cache lookup) emits structured spans with token counts, timing, and cache metadata.

### Prometheus Metrics

Expose pipeline metrics on a `/metrics` endpoint. Track token usage, cache hit rates, compaction ratios, circuit breaker state, and pipeline latency across all your agents.

### Alert Rules Engine

Define threshold-based alerts on pipeline metrics. Get notified when cache hit rates drop below target, circuit breakers trip, or token budgets are consistently exceeded.

### SLA Degradation Policies

*Coming soon.* Define service-level targets for pipeline latency and context quality. SR2 Pro will automatically adjust compaction aggressiveness and summarization triggers to meet SLA constraints under load.

## Pricing

See [sr2.dev/pricing](https://sr2.dev/pricing) for plans and details.
