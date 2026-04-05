# SR2 Testing Guide

How to write tests for SR2. This covers test philosophy, patterns, and anti-patterns — not setup instructions (see [CONTRIBUTING.md](CONTRIBUTING.md) for that).

---

## Philosophy

### Test Behavior, Not Implementation

Tests verify **what** a component does through its public API, not **how** it does it internally. If you refactor internals without changing behavior, zero tests should break.

```python
# Good: tests the public contract
async def test_compile_produces_context_within_budget():
    engine = PipelineEngine(resolvers, cache_registry)
    result = await engine.compile(config, context)
    assert result.tokens <= config.token_budget

# Bad: tests internal implementation
async def test_compile_calls_serialize_layer_for_each_layer():
    with patch.object(engine, '_serialize_layer') as mock:  # NO
        await engine.compile(config, context)
        assert mock.call_count == 3  # Brittle — breaks on refactor
```

### One Behavior Per Test

Each test verifies one thing. If a test name needs "and", split it.

```python
# Good: focused
async def test_compaction_skips_user_messages(): ...
async def test_compaction_skips_raw_window(): ...

# Bad: testing multiple behaviors
async def test_compaction_skips_user_messages_and_raw_window_and_small_content(): ...
```

### Test Names Describe Scenarios

Use `test_<action>_<condition>` or `test_<component>_<behavior>`:

```python
test_resolve_existing_key()           # Happy path
test_resolve_missing_key_raises()     # Error case
test_compaction_disabled_returns_none()  # Config variant
test_get_by_key_sorted()              # Order behavior
test_malformed_json()                 # Edge case
test_extraction_failure_doesnt_block_compaction()  # Degradation
```

---

## What to Mock

### Mock at System Boundaries

These are the **only** things you should mock:

| Boundary | How to Mock |
|----------|-------------|
| LLM calls | Inline async callable: `async def mock_llm(prompt: str) -> str:` |
| LiteLLM | `patch("litellm.acompletion", new_callable=AsyncMock)` |
| Database | Use `InMemoryMemoryStore` or `SQLiteMemoryStore(":memory:")` |
| HTTP | `httpx.AsyncClient` mock or `patch` |
| MCP servers | Inject async callables into resolver constructors |

### Never Mock Internal Classes

```python
# Wrong
with patch.object(ConversationManager, 'run_compaction'):  # NO
with patch('sr2.memory.extraction.MemoryExtractor._filter'):  # NO
mock_registry = MagicMock(spec=ContentResolverRegistry)  # NO

# Right — use real objects, mock only the boundary
store = InMemoryMemoryStore()  # Real store, in-memory
extractor = MemoryExtractor(llm_callable=mock_llm, store=store)  # Real extractor, mock LLM
```

---

## Mock Patterns

### Pattern 1: Inline Async Callable (LLM mocking)

The most common pattern. Used for memory extraction, summarization, compaction LLM calls.

```python
async def mock_llm(prompt: str) -> str:
    return json.dumps([
        {"key": "user.name", "value": "Alice", "memory_type": "identity"}
    ])

extractor = MemoryExtractor(llm_callable=mock_llm, store=store)
result = await extractor.extract("I'm Alice", conversation_id="conv_1", turn_number=1)
```

### Pattern 2: Mock Resolver Class

For pipeline engine tests where you need configurable, trackable resolvers.

```python
class MockResolver:
    def __init__(self, content="test content", tokens=100):
        self._content = content
        self._tokens = tokens
        self.call_count = 0

    async def resolve(self, key, config, context):
        self.call_count += 1
        return ResolvedContent(key=key, content=self._content, tokens=self._tokens)

class FailingResolver:
    async def resolve(self, key, config, context):
        raise RuntimeError("resolver failed")
```

### Pattern 3: unittest.mock for External Libraries

When patching module-level imports (LiteLLM, httpx).

```python
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

def _make_mock_response(content="Hello", prompt_tokens=100, completion_tokens=50):
    msg = SimpleNamespace(content=content, tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice], usage=usage)

async def test_complete_returns_content():
    mock_resp = _make_mock_response(content="Test response")
    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
        client = LLMClient(config)
        resp = await client.complete(model_key="model", messages=[...])
        assert resp.content == "Test response"
```

### Pattern 4: Async Callable Injection

For resolvers that accept callables (MCP prompts, resources).

```python
async def mock_get_prompt(name, arguments=None, server_name=None):
    return f"[user] Review {arguments.get('language', 'code')}"

resolver = MCPPromptResolver(mock_get_prompt)
result = await resolver.resolve("code_review", {"server": "prompts", "arguments": {"language": "python"}}, context)
```

---

## Test Structure

### File Layout

```python
"""Tests for [module/component]."""

import json
import pytest
from sr2.module import Component
from sr2.resolvers.registry import ResolverContext, ResolvedContent

# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def store():
    return InMemoryMemoryStore()

@pytest.fixture
def context():
    return ResolverContext(agent_config={}, trigger_input="test")

# ── Helpers ───────────────────────────────────────────

def _make_turn(num: int, role: str = "assistant", content: str = "content") -> ConversationTurn:
    return ConversationTurn(turn_number=num, role=role, content=content)

def _make_config(layers, token_budget=32000) -> PipelineConfig:
    return PipelineConfig(token_budget=token_budget, layers=layers)

# ── Tests ─────────────────────────────────────────────

class TestComponent:
    """Tests for Component behavior."""

    async def test_happy_path(self, store):
        ...

    async def test_error_case(self, store):
        ...

    async def test_edge_case(self, store):
        ...
```

### When to Use Classes vs Standalone Functions

| Use | When |
|-----|------|
| **Standalone async functions** | Simple resolver tests, few test cases, no shared state |
| **Test classes** | Store/engine tests with shared fixtures, many related test cases |
| **`setup_method`** | Integration tests with complex multi-component initialization |

### Fixture Patterns

**Simple instance (no cleanup):**
```python
@pytest.fixture
def store():
    return InMemoryMemoryStore()
```

**Async with cleanup:**
```python
@pytest.fixture
async def sqlite_store():
    store = SQLiteMemoryStore(":memory:")
    await store.initialize()
    yield store
    await store.close()
```

**Composed fixtures:**
```python
@pytest.fixture
def retriever(store):
    return HybridRetriever(store=store, strategy="keyword")
```

**Config fixtures (return dicts, not objects):**
```python
@pytest.fixture
def sample_layer_config():
    return {
        "name": "core",
        "cache_policy": "immutable",
        "contents": [{"key": "system_prompt", "source": "config"}],
    }
```

---

## Async Testing

pytest-asyncio is configured in `auto` mode (`asyncio_mode = "auto"` in pyproject.toml). This means:

- Async test functions are automatically detected — `@pytest.mark.asyncio` is optional but harmless
- Async fixtures work without special decorators
- All async tests share the same event loop policy

```python
# Both work — the decorator is optional in auto mode
async def test_resolve():
    result = await resolver.resolve("key", {}, context)
    assert result.content == "expected"

@pytest.mark.asyncio
async def test_resolve_explicit():
    result = await resolver.resolve("key", {}, context)
    assert result.content == "expected"
```

---

## Integration Tests

Integration tests live in `tests/integration/` and wire together real components without mocking. They're skipped by default and enabled via environment variables.

### When to Write Integration Tests

- Testing full pipeline compilation with real resolvers
- Testing memory stores with real databases (PostgreSQL)
- Testing end-to-end flows that span multiple components

### Markers

```python
import os
import pytest

requires_postgres = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL") and not os.environ.get("RUN_INTEGRATION"),
    reason="Set TEST_POSTGRES_URL or RUN_INTEGRATION=1",
)

requires_llm = pytest.mark.skipif(
    not os.environ.get("TEST_LLM_API_KEY"),
    reason="TEST_LLM_API_KEY not set",
)
```

### Structure

```python
class TestEndToEndPipeline:
    def setup_method(self):
        self.resolver_reg = build_resolver_registry()  # Real resolvers
        self.cache_reg = create_default_cache_registry()
        self.engine = PipelineEngine(self.resolver_reg, self.cache_reg)

    async def test_minimal_config_compiles(self):
        config = PipelineConfig(token_budget=8000, layers=[...])
        ctx = ResolverContext(agent_config={"system_prompt": "You are helpful."}, trigger_input="Hello")
        result = await self.engine.compile(config, ctx)

        assert "helpful" in result.content
        assert result.tokens > 0
        assert result.pipeline_result.overall_status == "success"
```

### Running

```bash
# Unit tests only (default)
pytest tests/ --ignore=tests/integration/ -v

# Integration tests (requires Docker PostgreSQL)
docker compose -f docker-compose.test.yaml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v
docker compose -f docker-compose.test.yaml down
```

---

## Assertion Patterns

### Assert Outcomes, Not Call Counts

```python
# Good: verifies the outcome
result = await engine.compile(config, context)
assert result.tokens <= config.token_budget
assert "system prompt" in result.content

# Bad: verifies implementation details
assert resolver.call_count == 3  # Brittle — what if caching changes?
```

### Common Assertions

```python
# Type and structure
assert isinstance(result, CompiledContext)
assert isinstance(result, ResolvedContent)

# Content
assert result.content == "expected content"
assert "substring" in result.content
assert result.key == "system_prompt"

# Numeric
assert result.tokens == estimate_tokens("expected content")
assert result.tokens <= config.token_budget
assert result.tokens > 0

# Collections
assert len(result.memories) == 2
assert list(result.layers.keys()) == ["core", "memory", "conversation"]

# Metadata
assert result.metadata["source"] == "mcp_prompt"
assert result.metadata.get("server") == "prompts"

# Errors
with pytest.raises(KeyError, match="Key 'missing' not found"):
    await resolver.resolve("missing", {}, ctx)

# Order
assert results[0].value == "newest"  # Verify sort order
assert result.content.index("first") < result.content.index("second")

# Boolean state
assert manager.zones().raw == []
assert len(manager.zones().compacted) > 0
```

---

## Test Anti-Patterns

### Implementation-Coupled Tests

```python
# Bad: tests internal method
def test_filter_removes_tool_artifacts():
    extractor = MemoryExtractor(...)
    result = extractor._filter_memories(memories)  # NO — private method
```

Fix: test through `extract()` and verify the output doesn't contain tool artifacts.

### Over-Mocking

```python
# Bad: mocking everything
engine = MagicMock(spec=PipelineEngine)
engine.compile = AsyncMock(return_value=CompiledContext(...))
result = await engine.compile(config, context)
assert result.tokens == 50  # You're testing your mock, not the engine
```

Fix: use real `PipelineEngine` with `MockResolver` injected.

### Brittle String Assertions

```python
# Bad: exact match on generated text
assert result.content == "System: You are a helpful assistant.\n\nUser: Hello\n\nAssistant: "

# Good: check semantic content
assert "helpful assistant" in result.content
assert result.content.startswith("System:")
```

### Testing Framework Internals

```python
# Bad: testing Pydantic validation directly
def test_config_model_validates_fields():
    with pytest.raises(ValidationError):
        PipelineConfig(token_budget=-1)  # This tests Pydantic, not SR2
```

Unless you added custom validators, Pydantic's validation is Pydantic's problem.

### Shared Mutable State Between Tests

```python
# Bad: module-level mutable state
_shared_store = InMemoryMemoryStore()

def test_save(self):
    await _shared_store.save(memory)  # Leaks into other tests

# Good: fixture per test
@pytest.fixture
def store():
    return InMemoryMemoryStore()  # Fresh for each test
```
