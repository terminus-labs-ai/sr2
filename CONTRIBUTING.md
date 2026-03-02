# Contributing to SR2

## Setup

```bash
# Clone
git clone https://github.com/terminus-labs-ai/sr2.git
cd sr2

# Install with dev dependencies
pip install -e ".[dev]"
```

Requires Python 3.12+.

## Running Tests

```bash
# Unit tests
pytest tests/ --ignore=tests/integration/ -v

# With coverage
pytest tests/ --ignore=tests/integration/ --cov=sr2 --cov-report=term-missing

# Integration tests (requires PostgreSQL)
docker compose -f docker-compose.test.yml up -d
RUN_INTEGRATION=1 pytest tests/integration/ -v
docker compose -f docker-compose.test.yml down

# Single test file
pytest tests/test_memory/test_extraction.py -v
```

## Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix
ruff check src/ --fix

# Format
ruff format src/
```

Config is in `pyproject.toml`: Python 3.12 target, 100-char line length.

## Code Style

- Type hints on all public function signatures
- Async by default for I/O operations
- Pydantic models for configuration with `Field(description=...)`
- Tests use pytest with `pytest-asyncio` (auto mode — no `@pytest.mark.asyncio` needed on fixtures)

## Project Structure

```
src/
├── sr2/           # The library (pip install sr2)
│   ├── config/        Config models, loader, validation
│   ├── pipeline/      Engine, router, conversation manager
│   ├── resolvers/     Content resolvers
│   ├── cache/         Cache policies
│   ├── compaction/    Rule-based content compaction
│   ├── summarization/ LLM-powered summarization
│   ├── memory/        Extraction, retrieval, conflicts
│   ├── degradation/   Circuit breaker, degradation ladder
│   ├── tools/         Tool state machine, masking
│   ├── metrics/       Prometheus/OTel exporters
│   └── a2a/           Agent-to-Agent protocol
│
├── runtime/       # Agent runtime (optional)
│   ├── agent.py       Main Agent class
│   ├── cli.py         CLI entry point
│   ├── llm/           LiteLLM wrapper, loop
│   ├── plugins/       Telegram, HTTP, timer, A2A
│   └── session.py     Session management
│
configs/           # Example configs
tests/             # 688 tests
examples/          # Runnable examples
```

## Making Changes

1. Create a branch from `main`
2. Write or update tests for your changes
3. Run `pytest` and `ruff check` before pushing
4. Open a pull request with a clear description of what and why

## Config Changes

If you modify Pydantic models in `src/sr2/config/models.py`, regenerate the config docs:

```bash
python -m schema_gen --format md > docs/configuration.md
```

## Adding a New Resolver

1. Create `src/sr2/resolvers/your_resolver.py` implementing the `ContentResolver` protocol
2. Register it in `SR2._build_resolver_registry()` or let users register it manually
3. Add tests in `tests/test_resolvers/`

The resolver protocol:

```python
async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
    ...
```

## Questions?

Open an issue on GitHub. We're happy to help.
