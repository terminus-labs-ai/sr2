# Observability Guide

SR2 exposes 43 metrics covering cache performance, compaction, retrieval, token usage, pipeline latency, degradation events, and tool masking. The raw `MetricCollector` is included in sr2 core. To export metrics to external systems, install **sr2-pro**:

```bash
pip install sr2-pro
```

sr2-pro provides:

1. **Prometheus scraping** — `/metrics` endpoint in Prometheus text exposition format
2. **OpenTelemetry export** — push metrics to any OTLP-compatible backend
3. **Alert rules** — threshold-based alerting with suppression

Both exporters can run simultaneously.

## Architecture

```
                                   ┌──────────────────┐
                                   │  Grafana (:3000)  │
                                   └────────┬─────────┘
                                            │ query
                        ┌───────────────────┴──────────────────┐
                        │                                      │
               ┌────────▼────────┐                    ┌────────▼────────┐
               │   Prometheus    │                    │ OTLP Backend    │
               │   (:9090)       │                    │ (Jaeger, etc.)  │
               └────────┬────────┘                    └────────▲────────┘
                        │ scrape                               │ push
                        │                                      │
               ┌────────▼────────────────────────────────────  │
               │            SR2 Agent (:8008)              │
               │                                               │
               │  GET /metrics ──► PrometheusExporter           │
               │  MetricCollector ──► OTelExporter ─────────────┘
               └───────────────────────────────────────────────┘
```

## Option A: Prometheus Scraping (simplest)

The `/metrics` endpoint is always available when running with `--http`. No extra dependencies needed.

### 1. Start the agent with HTTP

```bash
uv run sr2-agent configs/agents/edi --http --port 8008
```

Verify metrics are exposed:

```bash
curl http://localhost:8008/metrics
```

> **Tip:** The agent also serves a built-in chat UI at `GET /` and OpenAI-compatible endpoints at `/v1/chat/completions` and `/v1/models`. See the [Quick Reference](reference.md) for details.

You should see output like:

```
# HELP sr2_cache_hit_rate Pipeline metric
# TYPE sr2_cache_hit_rate gauge
sr2_cache_hit_rate{agent="EDI",interface="telegram"} 0.85
# HELP sr2_pipeline_total_tokens Pipeline metric
# TYPE sr2_pipeline_total_tokens gauge
sr2_pipeline_total_tokens{agent="EDI",interface="telegram"} 4200
```

### 2. Configure Prometheus to scrape the agent

Add a scrape job to `o11y/prometheus.yaml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "ollama_metrics"
    static_configs:
      - targets: ["ollama-exporter:8080"]

  - job_name: "sr2"
    static_configs:
      - targets: ["sr2-edi:8008"]  # container name + port
    metrics_path: /metrics
    scrape_interval: 10s
```

If running outside Docker, use `localhost:8008` instead.

### 3. Start the stack

```bash
docker compose up -d prometheus grafana
```

Prometheus is at `http://localhost:9501`, Grafana at `http://localhost:9502`.

### 4. Add Prometheus as a Grafana data source

1. Open Grafana (`http://localhost:9502`, default login `admin`/`admin`)
2. Go to **Connections > Data sources > Add data source**
3. Select **Prometheus**
4. Set URL to `http://prometheus:9090` (Docker internal) or `http://localhost:9501` (host)
5. Click **Save & test**

### 5. Query metrics in Grafana

Example PromQL queries for dashboards:

| Panel | Query |
|---|---|
| Cache hit rate | `sr2_cache_hit_rate` |
| Token usage per request | `sr2_pipeline_total_tokens` |
| Pipeline latency | `sr2_pipeline_total_duration_ms` |
| Compaction ratio | `sr2_compaction_ratio` |
| Budget utilization | `sr2_budget_utilization` |
| Cache hit rate (avg last 100) | `sr2_cache_hit_rate_avg100` |
| Circuit breaker fires | `sr2_circuit_breaker_activations` |
| Degradation events | `sr2_full_degradation_events` |

---

## Option B: OpenTelemetry Export

OTel pushes metrics to any OTLP-compatible backend (Grafana Cloud, Datadog, Jaeger, a local collector, etc.).

### 1. Install sr2-pro (includes OTel dependencies)

```bash
pip install sr2-pro
```

sr2-pro includes `opentelemetry-api` and `opentelemetry-sdk`. You also need an exporter package for your backend:

| Backend | Package |
|---|---|
| OTLP (gRPC) | `opentelemetry-exporter-otlp-proto-grpc` |
| OTLP (HTTP) | `opentelemetry-exporter-otlp-proto-http` |
| Prometheus (pull) | `opentelemetry-exporter-prometheus` |
| Console (debug) | included in SDK |

```bash
# Example: OTLP over gRPC
uv pip install opentelemetry-exporter-otlp-proto-grpc
```

### 2. Configure the OTel SDK via environment variables

The OTel SDK reads standard `OTEL_*` environment variables. Set these before starting the agent:

```bash
# Required: where to send metrics
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Optional: protocol (grpc or http/protobuf)
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"

# Optional: resource attributes for identifying the agent
export OTEL_RESOURCE_ATTRIBUTES="service.name=sr2-edi,deployment.environment=production"

# Optional: export interval (default 60s, lower for dev)
export OTEL_METRIC_EXPORT_INTERVAL="10000"
```

### 3. Initialize the OTel SDK at startup

The SR2 `OTelExporter` creates OTel instruments via `opentelemetry.metrics.get_meter()`, but the SDK's `MeterProvider` must be configured before the agent starts. Add this to your entrypoint or a startup script:

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

# Pick your exporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

resource = Resource.create({"service.name": "sr2-edi"})

reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(),       # reads OTEL_EXPORTER_OTLP_ENDPOINT from env
    export_interval_millis=10_000,
)

provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)
```

This must run **before** the agent is instantiated (before `OTelExporter.__init__` calls `get_meter()`).

For quick debugging, use the console exporter instead:

```python
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

reader = PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=5_000)
provider = MeterProvider(resource=resource, metric_readers=[reader])
metrics.set_meter_provider(provider)
```

### 4. Start the agent

```bash
uv run sr2-agent configs/agents/edi --http --port 8008
```

On startup you should see:

```
OTelExporter registered on MetricCollector (meter=sr2)
```

Metrics are now pushed to your OTLP endpoint on every pipeline invocation.

### 5. Example: local OTLP collector + Grafana

Add an OTel Collector to `docker-compose.yaml`:

```yaml
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: otel-collector
    command: ["--config=/etc/otel-config.yaml"]
    volumes:
      - ./o11y/otel-collector.yaml:/etc/otel-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    networks:
      - sr2
```

Create `o11y/otel-collector.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [prometheus]
```

Then add the collector as a Prometheus scrape target:

```yaml
# o11y/prometheus.yaml
scrape_configs:
  - job_name: "otel-collector"
    static_configs:
      - targets: ["otel-collector:8889"]
```

---

## Metric Reference

### Histograms (distributions)

| Metric | Unit | Description |
|---|---|---|
| `sr2_pipeline_total_tokens` | tokens | Total tokens used per pipeline run |
| `sr2_pipeline_total_duration_ms` | ms | End-to-end pipeline latency |
| `sr2_stage_duration_ms` | ms | Per-stage duration |
| `sr2_stage_tokens` | tokens | Per-stage token count |
| `sr2_retrieval_latency_ms` | ms | Memory retrieval latency |

### Counters (monotonically increasing)

| Metric | Description |
|---|---|
| `sr2_cache_invalidation_events` | Cache prefix invalidations |
| `sr2_fallback_rate` | Pipeline fallback events |
| `sr2_circuit_breaker_activations` | Circuit breaker trips |
| `sr2_full_degradation_events` | Complete pipeline failures |
| `sr2_denied_tool_attempts` | Tool calls blocked by masking |

### Gauges (point-in-time values)

| Metric | Description |
|---|---|
| `sr2_cache_hit_rate` | KV-cache prefix reuse ratio |
| `sr2_cache_efficiency` | Overall cache efficiency score |
| `sr2_cost_savings_ratio` | Token savings from caching |
| `sr2_context_prefix_stable` | Whether prefix was stable (0/1) |
| `sr2_compaction_ratio` | Compaction token reduction ratio |
| `sr2_compaction_recovery_rate` | How often compacted data is re-fetched |
| `sr2_compaction_coverage` | Fraction of turns eligible for compaction |
| `sr2_summarization_fidelity` | Summarization quality score |
| `sr2_summarization_ratio` | Token reduction from summarization |
| `sr2_summarization_frequency` | How often summarization triggers |
| `sr2_retrieval_precision` | Retrieval relevance score |
| `sr2_retrieval_empty_rate` | Fraction of retrievals returning nothing |
| `sr2_budget_utilization` | Token budget usage (0.0-1.0) |
| `sr2_token_efficiency` | Useful tokens / total tokens |
| `sr2_response_quality` | Response quality estimate |
| `sr2_task_completion_rate` | Task completion ratio |
| `sr2_state_transition_rate` | State machine transition frequency |
| `sr2_zone_raw_tokens` | Tokens in raw (uncompacted) zone |
| `sr2_zone_compacted_tokens` | Tokens in compacted zone |
| `sr2_zone_summarized_tokens` | Tokens in summarized zone |

All metrics include `agent` and `interface` labels.

---

## Built-in Alerts

The `AlertRuleEngine` monitors metrics in real-time with these default thresholds:

| Metric | Condition | Severity |
|---|---|---|
| `sr2_cache_hit_rate` | < 0.50 | warning |
| `sr2_full_degradation_events` | > 0 | critical |
| `sr2_circuit_breaker_activations` | > 0 | warning |
| `sr2_retrieval_latency_ms` | > 500 | warning |
| `sr2_fallback_rate` | > 0.10 | warning |

Alerts are suppressed for 5 minutes after firing to prevent storms.

---

## Troubleshooting

**No metrics on `/metrics`:** The endpoint only returns data after at least one pipeline run. Send a message to the agent first.

**`OTelExporter` not registering:** Check that sr2-pro is installed (`pip install sr2-pro`). If it's missing, the agent starts normally but without OTel export — this is by design.

**OTel metrics not appearing in backend:** Verify the `MeterProvider` is configured *before* the agent starts. Check `OTEL_EXPORTER_OTLP_ENDPOINT` is reachable. Use the `ConsoleMetricExporter` first to confirm metrics are flowing.

**Prometheus not scraping:** Check the scrape target matches the agent's host:port. Run `curl http://<target>/metrics` from the Prometheus container to verify connectivity.
