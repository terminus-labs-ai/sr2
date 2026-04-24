# SR2 Code Review Checklist

Checklist for reviewing SR2 contributions — by human or AI agent. Every PR should pass all applicable checks.

For context on why these matter, see [DEVELOPMENT.md](DEVELOPMENT.md) and [TESTING.md](TESTING.md).

---

## Architecture

- [ ] **Correct package**: Code belongs in sr2 core (context management) or in sr2-relay/sr2-spectre (separate repos).
- [ ] **Dependency direction**: No upward imports — sr2 core has no knowledge of relay or spectre.
- [ ] **No internal imports**: Consumers import only from public API (`sr2.*`), never `sr2.pipeline.engine._serialize_layer`.
- [ ] **Entry points for extensions**: Out-of-tree functionality uses entry-point registration, not direct imports into core.

## Design Patterns

- [ ] **Protocol-first**: New extensibility points use `Protocol` classes, not abstract base classes.
- [ ] **Registry pattern**: New implementations are registered via registries, not hardcoded in engine/factory code.
- [ ] **Stateless resolvers**: Resolvers receive all state via `ResolverContext`. No internal caching, no mutable instance state.
- [ ] **Config-driven**: Behavior is controlled by YAML config fields, not hardcoded values. New config fields have defaults and descriptions.
- [ ] **Async I/O**: All I/O operations are `async`. No blocking calls.

## Config Changes

- [ ] **Pydantic model updated**: New config fields added to appropriate model in `config/models.py`.
- [ ] **Default provided**: Every new field has a sensible default value.
- [ ] **Description provided**: Every new field has a `Field(description="...")`.
- [ ] **Validation added**: Constraints added in `config/validation.py` if applicable.
- [ ] **Docs regenerated**: Config docs updated via `sr2-config-docs`.

## KV-Cache Safety

- [ ] **Prefix stability**: Changes to Layer 1 (core) content don't introduce non-determinism (no timestamps, random IDs).
- [ ] **Correct cache policy**: New layers use the appropriate cache policy (immutable for static, append_only for conversation).
- [ ] **No eager side effects**: Resolvers don't modify shared state that affects other resolvers in the same compilation pass.
- [ ] **Deferred touch**: Memory access counts are updated post-LLM, not during retrieval.

## Memory Safety

- [ ] **Scope isolation**: Memory reads filter by `allowed_read` scopes. Writes respect `allowed_write`.
- [ ] **No scope leaks**: Private memories are never returned to agents outside their scope.
- [ ] **Conflict detection**: Changes to memory extraction or storage consider conflict detection.
- [ ] **Embedding handling**: If memories are saved, embeddings are generated when `embed_callable` is available.

## Testing

- [ ] **Tests exist**: New features have corresponding tests. Bug fixes have regression tests.
- [ ] **Behavioral tests**: Tests verify outcomes through public APIs, not internal method calls.
- [ ] **Mocks at boundaries only**: Only LLM calls, databases, HTTP, and filesystem are mocked. No mocking of internal classes.
- [ ] **Async patterns**: Async tests use `async def`, fixtures use `yield` for cleanup.
- [ ] **One behavior per test**: Each test function tests exactly one scenario.
- [ ] **Tests pass**: `pytest tests/ --ignore=tests/integration/ -v` passes clean.

## Quality

- [ ] **Type hints**: Public functions and methods have type annotations.
- [ ] **No hardcoded values**: Magic numbers and strings are in config or constants.
- [ ] **Error handling**: Errors are raised (config/validation) or gracefully degraded (pipeline/runtime), never silently swallowed.
- [ ] **Naming conventions**: Classes, functions, config keys, and metric names follow project conventions (see [DEVELOPMENT.md § Naming](DEVELOPMENT.md#naming-conventions)).
- [ ] **Lint clean**: `ruff check src/` and `ruff format --check src/` pass.

## Metrics & Observability

- [ ] **New features instrumented**: Significant new functionality has corresponding metrics in `MetricCollector`.
- [ ] **Metric naming**: New metrics use `sr2_` prefix with `snake_case`.
- [ ] **Thresholds**: Critical metrics have alert thresholds defined in `metrics/definitions.py`.

## Documentation

- [ ] **Docstrings**: Public classes and functions have docstrings explaining the contract (what, not how).
- [ ] **Config docs**: If config models changed, docs are regenerated.
- [ ] **No docs bloat**: No unnecessary README updates, no changelog entries for in-progress work.

---

## Quick Decision Guide

| Question | Answer |
|----------|--------|
| Should this be a resolver or engine logic? | If it fetches/computes content for a layer, it's a resolver. If it orchestrates resolvers, it's engine logic. |
| Should this be config or code? | If it could reasonably vary per agent or per interface, it's config. If it's structural, it's code. |
| Should this test use a mock? | Only if it crosses a system boundary (LLM, DB, HTTP). Otherwise use real objects. |
| Should I add a new metric? | If the behavior is operationally meaningful and could trigger an alert or dashboard query, yes. |
| Does this need an integration test? | If it wires together 3+ real components or requires a database, yes. Otherwise unit tests suffice. |
