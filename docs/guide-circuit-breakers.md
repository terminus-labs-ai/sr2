# Circuit Breaker Degradation Guide

SR2's circuit breaker pattern prevents cascading failures. When a layer (retrieval, summarization, memory) fails repeatedly, the breaker opens automatically and that layer is skipped. This allows your agent to keep running with reduced context rather than crashing.

## How It Works

Each layer (except core/system prompt) has its own circuit breaker:

1. **Closed** — Normal operation. Failures are tracked.
2. **Open** — Too many consecutive failures (default: 3). Layer is skipped.
3. **Half-Open** — After cooldown (default: 5 minutes), one trial request is allowed to test recovery.

If the trial succeeds, the breaker closes and normal operation resumes.

## Configuration

In your pipeline config:

```yaml
degradation:
  circuit_breaker_threshold: 3         # Failures before opening
  circuit_breaker_cooldown_minutes: 5  # Wait before half-open
```

Default thresholds are conservative. Adjust based on your reliability needs:

- **Threshold = 1**: Very aggressive. Opens on first failure. Use for critical resolvers.
- **Threshold = 3**: Balanced (default). Tolerates occasional glitches.
- **Threshold = 5+**: Lenient. Better for resolvers with intermittent issues.

- **Cooldown = 1**: Fast recovery attempts. Good for transient failures.
- **Cooldown = 5+**: Longer grace period. Useful if backend services recover slowly.

## Example Scenarios

### Scenario 1: Flaky Memory Retrieval

Your retrieval layer fails 2/10 times:

```
Request 1: Retrieval fails     → Counter = 1
Request 2: Retrieval succeeds  → Counter = 0 (reset)
Request 3: Retrieval fails     → Counter = 1
Request 4: Retrieval fails     → Counter = 2
Request 5: Retrieval fails     → Counter = 3 → Circuit opens
Request 6: Retrieval skipped   (open)
...5 minutes pass...
Request 11: Trial retrieval    (half-open) → succeeds → Circuit closes
Request 12: Retrieval normal   (closed)
```

### Scenario 2: Critical Summarization

Summarization is important — use a higher threshold:

```yaml
degradation:
  circuit_breaker_threshold: 5
```

This allows up to 4 consecutive failures before giving up.

## Per-Layer Behavior

**Core Layer** (system prompt)
- Never skipped, even with circuit breaker open
- Pipeline fails entirely if core can't be resolved

**Memory Layer** (retrieved context)
- Skipped if circuit opens
- Agent continues without historical context

**Conversation Layer** (session history, user input)
- Skipped if circuit opens
- Critical — usually always succeeds unless resolver has bugs

## Monitoring

Check circuit breaker status in `PipelineResult`:

```python
result = await engine.compile(config, context)

for stage in result.stages:
    if stage.status == "degraded":
        print(f"Layer '{stage.stage_name}' is degraded")
        print(f"Reason: {stage.error}")
```

Or export metrics to Prometheus:

```python
from sr2_pro.metrics.prometheus import PrometheusExporter  # requires sr2-pro

exporter = PrometheusExporter()
exporter.export(result)
# Metrics include: sr2_circuit_breaker_status, sr2_degradation_events
```

> **PrometheusExporter requires sr2-pro.** Install with `pip install sr2-pro`. See [sr2.dev/pricing](https://sr2.dev/pricing).

## Best Practices

### 1. Monitor Breaker Openings
Set up alerts:

```yaml
# Prometheus alert
groups:
  - name: sr2_alerts
    rules:
      - alert: CircuitBreakerOpen
        expr: sr2_circuit_breaker_status{state="open"} > 0
        for: 1m
        annotations:
          summary: "Circuit breaker open for {{ $labels.layer }}"
```

### 2. Distinguish Required from Optional
Mark non-critical layers as `optional: true`:

```yaml
layers:
  - name: memory
    contents:
      - key: retrieved_memories
        source: retrieval
        optional: true  # Missing memories don't crash the agent
      - key: user_input
        source: input
        # No optional=true — agent can't work without input
```

### 3. Design Resolvers for Resilience
Resolvers should handle transient errors:

```python
async def resolve(self, config, context):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return await self._fetch_data()
        except TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                continue
            logger.error("Max retries reached")
            return None  # Circuit breaker takes over
```

### 4. Use Graceful Degradation Ladder
If a layer fails, SR2 can degrade to a reduced-context mode. Configure the degradation ladder:

```python
from sr2.degradation.ladder import DegradationLadder

ladder = DegradationLadder()
# Modes: full -> skip_summarization -> skip_intent -> raw_context -> system_prompt_only
```

### 5. Test Circuit Behavior
Write tests for breaker behavior:

```python
import pytest
from sr2.pipeline.engine import PipelineEngine

@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failure():
    """Test that circuit opens after N failures."""
    # Create a resolver that always fails
    failing_resolver = FailingTestResolver()
    registry.register("failing", failing_resolver)

    engine = PipelineEngine(registry)

    # Simulate 3 consecutive failures
    for i in range(3):
        result = await engine.compile(config, context)
        assert result.stages[1].status == "failed"

    # On 4th attempt, breaker should be open
    result = await engine.compile(config, context)
    assert result.stages[1].status == "degraded"
    assert "circuit_breaker_open" in result.stages[1].error
```

## Troubleshooting

**Circuit breaker opens too often**
- Increase `circuit_breaker_threshold`
- Fix the underlying resolver (timeouts, connection errors, etc.)
- Check external service health

**Circuit opens and never recovers**
- Verify `circuit_breaker_cooldown_minutes` is reasonable
- Check if trial requests are actually succeeding
- Look for permanent issues in resolver logs

**Agent degrades unexpectedly during operation**
- Monitor individual layer success rates
- Add defensive error handling in resolvers
- Consider splitting failure-prone layers into separate endpoints

## Advanced: Custom Degradation

For fine-grained control, implement custom degradation in your resolver:

```python
class ResilientMemoryResolver(ContentResolver):
    async def resolve(self, config, context):
        # Try primary backend
        try:
            return await self._fetch_from_primary()
        except Exception as e:
            logger.warning(f"Primary failed: {e}")

        # Fall back to cache
        try:
            return await self._fetch_from_cache()
        except Exception:
            logger.error("Fallback also failed")

        # Return degraded but non-null response
        return ResolvedContent(
            key="memories",
            content="[Memories unavailable]",
            tokens=3,
            metadata={"degraded": True},
        )
```

This pattern lets resolvers handle their own recovery before the circuit breaker activates.
