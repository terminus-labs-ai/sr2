# Plugin Development Guide

SR2 uses a plugin system for extensible backends — memory stores, metric exporters, alert engines, and degradation policies. Plugins are discovered via Python entry points and activated through YAML config.

## How It Works

1. You implement a protocol (e.g., `MemoryStore`, `PushExporter`)
2. You register it via a Python entry point in your `pyproject.toml`
3. SR2 discovers it lazily on first access via `PluginRegistry`
4. Users activate it in their agent config YAML

No imports from your package appear in SR2 core — it's fully decoupled.

## Entry-Point Groups

| Group | Protocol | What it does |
|-------|----------|--------------|
| `sr2.stores` | `MemoryStore` | Memory persistence backends |
| `sr2.push_exporters` | `PushExporter` | Real-time metric export (e.g., OTel) |
| `sr2.pull_exporters` | `PullExporter` | On-demand metric export (e.g., Prometheus) |
| `sr2.alerts` | `AlertEngine` | Metric threshold alerting |
| `sr2.degradation_policies` | `DegradationPolicy` | Custom degradation strategies |

## Writing a Custom Memory Store

### 1. Implement the Protocol

Your store must implement all methods from `MemoryStore`:

```python
# my_store/store.py
from sr2.memory.schema import Memory, MemorySearchResult


class RedisMemoryStore:
    """Redis-backed memory store."""

    def __init__(self, redis_url: str):
        self._url = redis_url

    async def save(self, memory: Memory, embedding: list[float] | None = None) -> None:
        ...

    async def get(self, memory_id: str) -> Memory | None:
        ...

    async def get_by_key(
        self, key: str, include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[Memory]:
        ...

    async def search_by_key_prefix(
        self, prefix: str, include_archived: bool = False,
    ) -> list[Memory]:
        ...

    async def delete(self, memory_id: str) -> bool:
        ...

    async def archive(self, memory_id: str) -> bool:
        ...

    async def search_vector(
        self, embedding: list[float], top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        ...

    async def search_keyword(
        self, query: str, top_k: int = 10,
        include_archived: bool = False,
        scope_filter: list[str] | None = None,
        scope_refs: list[str] | None = None,
    ) -> list[MemorySearchResult]:
        ...

    async def count(self, include_archived: bool = False) -> int:
        ...

    async def list_scope_refs(
        self, scope_filter: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[tuple[str, str | None]]:
        ...
```

If your store needs setup (table creation, connection pooling), also implement `LifecycleStore`:

```python
from sr2.protocols import LifecycleStore

class RedisMemoryStore:  # also satisfies LifecycleStore
    async def create_tables(self) -> None:
        """Initialize Redis data structures."""
        ...
```

### 2. Create the Registration Function

```python
# my_store/__init__.py
from my_store.store import RedisMemoryStore

def register():
    """Called by SR2 via entry point discovery."""
    from sr2.memory.registry import register_store
    register_store("redis", RedisMemoryStore)
```

### 3. Register the Entry Point

```toml
# pyproject.toml
[project.entry-points."sr2.stores"]
redis = "my_store:register"
```

### 4. Use It

```yaml
# agent.yaml
pipeline:
  memory:
    store: redis
    store_kwargs:
      redis_url: "redis://localhost:6379"
```

That's it. `pip install my-store` makes it available to any SR2 agent.

## Writing a Metric Exporter

SR2 has two exporter types:

- **PushExporter** — registers a callback on `MetricCollector`, fires on every `collect()` call. Use for real-time telemetry (OTel, Datadog, StatsD).
- **PullExporter** — returns a string on demand via `export()`. Use for scrape endpoints (Prometheus, JSON).

### Push Exporter Example

```python
# my_exporter/datadog.py
from sr2.metrics.collector import MetricCollector
from sr2.metrics.definitions import MetricSnapshot


class DatadogExporter:
    """Pushes metrics to Datadog on every collect cycle."""

    def __init__(self, collector: MetricCollector):
        self._client = ...  # your Datadog client
        collector.on_collect(self._on_snapshot)

    def _on_snapshot(self, snapshot: MetricSnapshot) -> None:
        for mv in snapshot.metrics:
            self._client.gauge(mv.name, mv.value, tags=mv.labels)
```

Registration:

```python
# my_exporter/__init__.py
def register():
    from sr2.metrics.registry import register_push_exporter
    from my_exporter.datadog import DatadogExporter
    register_push_exporter("datadog", DatadogExporter)
```

```toml
[project.entry-points."sr2.push_exporters"]
datadog = "my_exporter:register"
```

Activate in config:

```yaml
observability:
  push_exporters: [datadog]
```

### Pull Exporter Example

```python
# my_exporter/json_exporter.py
import json
from sr2.metrics.collector import MetricCollector


class JSONExporter:
    """Exports metrics as JSON for a /metrics endpoint."""

    def __init__(self, collector: MetricCollector):
        self._collector = collector

    def export(self) -> str:
        latest = self._collector.get_latest(1)
        if not latest:
            return "{}"
        snapshot = latest[0]
        return json.dumps({
            m.name: {"value": m.value, "labels": m.labels}
            for m in snapshot.metrics
        })
```

Registration:

```python
def register():
    from sr2.metrics.registry import register_pull_exporter
    from my_exporter.json_exporter import JSONExporter
    register_pull_exporter("json", JSONExporter)
```

```toml
[project.entry-points."sr2.pull_exporters"]
json = "my_exporter:register"
```

Activate in config:

```yaml
observability:
  pull_exporter: json
```

The pull exporter is used by `SR2.export_metrics()`, which powers the `/metrics` HTTP endpoint.

## Writing an Alert Engine

Alert engines evaluate metric snapshots against rules and return triggered alerts.

```python
# my_alerts/engine.py
from sr2.protocols.alerts import Alert, AlertEngine
from sr2.metrics.definitions import MetricSnapshot


class SlackAlertEngine:
    """Evaluates metrics and posts alerts to Slack."""

    def __init__(self, webhook_url: str = ""):
        self._webhook = webhook_url
        self._rules: list[dict] = []

    def configure(self, rules: list[dict]) -> None:
        """Configure alert rules from dicts.

        Each dict: {metric_name, condition, value, severity}
        """
        self._rules = rules

    async def evaluate(self, snapshot: MetricSnapshot) -> list[Alert]:
        alerts = []
        for rule in self._rules:
            metric = snapshot.get(rule["metric_name"])
            if metric and self._check(metric.value, rule["condition"], rule["value"]):
                alert = Alert(
                    metric_name=rule["metric_name"],
                    actual_value=metric.value,
                    threshold_value=rule["value"],
                    condition=rule["condition"],
                    severity=rule.get("severity", "warning"),
                    timestamp=metric.timestamp.timestamp(),
                    labels=metric.labels,
                    message=f"{rule['metric_name']} breached threshold",
                )
                alerts.append(alert)
                await self._post_to_slack(alert)
        return alerts

    def _check(self, actual: float, condition: str, threshold: float) -> bool:
        ops = {"<": lambda a, b: a < b, ">": lambda a, b: a > b}
        return ops.get(condition, lambda a, b: False)(actual, threshold)

    async def _post_to_slack(self, alert: Alert) -> None:
        ...  # POST to self._webhook
```

Registration:

```python
def register():
    from sr2.metrics.registry import register_push_exporter  # if needed
    # For alerts, register directly on the PluginRegistry
    from sr2.plugins import PluginRegistry
    alert_reg = PluginRegistry("sr2.alerts")
    alert_reg.register("slack", SlackAlertEngine)
```

```toml
[project.entry-points."sr2.alerts"]
slack = "my_alerts:register"
```

```yaml
observability:
  alert_engine: slack
```

## Adding License Gating

If your plugin is commercial, gate registration behind a license check:

```python
def register():
    from my_package.license import require_license
    require_license()  # raises PluginLicenseError if invalid

    from sr2.memory.registry import register_store
    from my_package.store import MyStore
    register_store("my_store", MyStore)
```

SR2's `PluginRegistry` catches `PluginLicenseError` during discovery and stores it separately. When a user tries to `get()` a license-blocked plugin, they get a clear error message instead of a generic `ImportError`.

## Testing Plugins

Use `_reset_registry()` to isolate tests:

```python
from sr2.memory.registry import register_store, get_store, _reset_registry


def test_my_store_registers():
    _reset_registry()
    from my_store import register
    register()
    assert get_store("redis") is not None
```

To verify your class satisfies a protocol:

```python
from sr2.protocols import LifecycleStore, EmbeddingStore
from my_store.store import RedisMemoryStore


def test_satisfies_protocols():
    store = RedisMemoryStore("redis://localhost")
    assert isinstance(store, LifecycleStore)  # if you implement create_tables
```

## Plugin Discovery Flow

```
1. Agent config says: store: redis
2. SR2 calls get_store("redis")
3. PluginRegistry checks internal dict → miss
4. Triggers lazy discovery: importlib.metadata.entry_points(group="sr2.stores")
5. Finds entry point: redis = "my_store:register"
6. Calls my_store.register() → registers RedisMemoryStore
7. Returns RedisMemoryStore class
8. SR2 instantiates it with store_kwargs
```

Discovery happens once per registry per process. Subsequent calls hit the in-memory dict directly.

## Summary

| Step | What to do |
|------|-----------|
| 1 | Implement the protocol (`MemoryStore`, `PushExporter`, `PullExporter`, `AlertEngine`) |
| 2 | Write a `register()` function that calls the appropriate `register_*()` |
| 3 | Add an entry point in `pyproject.toml` pointing to your `register` function |
| 4 | Users `pip install` your package and add the plugin name to their YAML config |
