# Contributing to SR2

## Setup

```bash
# Clone
git clone https://github.com/terminus-labs-ai/sr2.git
cd sr2

# Install all packages + dev dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install
```

Requires Python 3.12+.

The pre-commit hooks run `ruff check` and `ruff format --check` automatically on
every commit so lint issues are caught before they reach CI.

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
pytest tests/sr2/test_memory/test_extraction.py -v
```

## Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check packages/

# Auto-fix
ruff check packages/ --fix

# Format
ruff format packages/
```

Config is in `pyproject.toml`: Python 3.12 target, 100-char line length.

## Code Style

- Type hints on all public function signatures
- Async by default for I/O operations
- Pydantic models for configuration with `Field(description=...)`
- Tests use pytest with `pytest-asyncio` (auto mode — no `@pytest.mark.asyncio` needed on fixtures)

## Project Structure

```
packages/
├── sr2/src/sr2/           # Core library (PyPI: sr2)
│   ├── config/            Config models, loader, validation
│   ├── pipeline/          Engine, router, conversation manager
│   ├── resolvers/         Content resolvers
│   ├── cache/             Cache policies
│   ├── compaction/        Rule-based content compaction
│   ├── summarization/     LLM-powered summarization
│   ├── memory/            Extraction, retrieval, conflicts
│   ├── degradation/       Circuit breaker, degradation ladder
│   ├── tools/             Tool state machine, masking
│   ├── metrics/           Prometheus/OTel exporters
│   └── a2a/               Agent-to-Agent protocol
│
configs/           # Example configs
tests/             # Core library tests
examples/          # Runnable examples
```

## Making Changes

1. Create a branch from `main`
2. Write or update tests for your changes
3. Run `pytest` and `ruff check` before pushing
4. Open a pull request with a clear description of what and why

## Config Changes

If you modify Pydantic models in `packages/sr2/src/sr2/config/models.py`, regenerate the config docs:

```bash
python -m schema_gen --format md > docs/configuration.md
```

## Adding a New Resolver

1. Create `packages/sr2/src/sr2/resolvers/your_resolver.py` implementing the `ContentResolver` protocol
2. Register it in `SR2._build_resolver_registry()` or let users register it manually
3. Add tests in `tests/sr2/test_resolvers/`

The resolver protocol:

```python
async def resolve(self, key: str, config: dict, context: ResolverContext) -> ResolvedContent:
    ...
```

## Questions?

Open an issue on GitHub. We're happy to help.
