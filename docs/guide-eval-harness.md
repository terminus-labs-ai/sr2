# Evaluation Harness Guide

The SR2 evaluation harness measures context engineering quality quantitatively. Compare compaction strategies, test summarization approaches, and detect regressions automatically.

## Overview

Evaluate how well your pipeline preserves information while staying within token budgets:

```python
from sr2.eval import (
    EvalRunner,
    ABTestRunner,
    RegressionDetector,
    create_coherence_suite,
)
from sr2.pipeline.engine import PipelineEngine
from sr2.resolvers.registry import ContentResolverRegistry

# Setup
engine = PipelineEngine(registry, cache_registry)
runner = EvalRunner(engine, registry)

# Run a suite
cases = create_coherence_suite()
results = await runner.run_suite(cases, config)

# Check results
for result in results:
    print(f"{result.case_name}: {result.metrics.coherence_score:.1%}")
```

## Core Concepts

### Eval Case
A single evaluation scenario with realistic context:

```python
from sr2.eval import EvalCase

case = EvalCase(
    id="test_001",
    name="Long conversation coherence",
    description="Agent should remember facts across 50 turns",
    system_prompt="You are a project planning assistant.",
    conversation_turns=[
        ("We chose Stripe for payments", "Good choice for PCI compliance."),
        ("What payment system did we pick?", "You chose Stripe."),
    ],
    expected_key_facts=["Stripe", "payments", "PCI"],
    expected_decisions=["Stripe"],
    expected_tokens=2000,
    tags=["coherence", "memory"],
)
```

### Eval Metrics
Quantitative measures from a single run:

- **Coherence Score** (0-1): Do expected facts appear in context?
- **Decision Preservation** (0-1): Are decisions retained?
- **Token Efficiency** (0-1): Ratio of actual to expected tokens
- **Cache Hit Rates**: KV-cache prefix reuse percentage
- **Degradation Events**: How many layers failed?
- **Compilation Time**: Pipeline performance

### Eval Result
Output from running a case:

```python
result = await runner.run_case(case, config)

# Check if it passed
if result.passed():
    print("✅ Test passed")
else:
    print("❌ Test failed")
    print(f"Coherence: {result.metrics.coherence_score:.1%}")
```

## Usage Patterns

### 1. Basic Evaluation

Run a single config against a test suite:

```python
from sr2.config.loader import load_config
from sr2.eval import create_coherence_suite, EvalRunner

# Load configuration
config = load_config("configs/defaults.yaml")

# Run eval suite
cases = create_coherence_suite()
results = await runner.run_suite(cases, config, concurrency=3)

# Print results
EvalRunner.print_results(results)
```

### 2. A/B Testing

Compare two pipeline configurations:

```python
from sr2.eval import ABTestRunner

# Create runner
ab_runner = ABTestRunner(runner)

# Compare configs
config_a = load_config("configs/defaults.yaml")
config_b = load_config("configs/optimized.yaml")
cases = create_coherence_suite()

# Run test
result = await ab_runner.compare(
    config_a=config_a,
    config_b=config_b,
    cases=cases,
    config_a_name="Baseline",
    config_b_name="Optimized",
)

print(result.summary())
```

### 3. Regression Detection

Track performance against a baseline:

```python
from sr2.eval import RegressionDetector

# Create detector with custom thresholds
detector = RegressionDetector(
    thresholds={
        "coherence_score": 0.05,
        "token_efficiency": 0.10,
    }
)

# Set baseline
detector.set_baseline(baseline_results)

# Check for regressions
current_results = await runner.run_suite(cases, new_config)
alerts = detector.check(current_results)
```

## Sample Eval Suites

Three built-in suites covering critical scenarios:

- **Coherence**: Memory retention across long conversations
- **Compaction**: Token efficiency through compression
- **Summarization**: Information preservation during summarization

```python
from sr2.eval import (
    create_coherence_suite,
    create_compaction_suite,
    create_summarization_suite,
)

suites = {
    "coherence": create_coherence_suite(),
    "compaction": create_compaction_suite(),
    "summarization": create_summarization_suite(),
}
```

## Performance Optimization

```python
# Run cases in parallel
results = await runner.run_suite(cases, config, concurrency=10)

# Sample subset of cases
sample = cases[:5]
results = await runner.run_suite(sample, config)
```

## Troubleshooting

**Low coherence**: Enable memory retrieval, check compaction rules
**High token usage**: Adjust compaction, enable summarization earlier
**Regression false positives**: Increase baseline sample size, loosen thresholds
