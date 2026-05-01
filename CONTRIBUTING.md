# Contributing to SR2 v2

## Branching

- **revamp** — v2 development branch. All v2 work happens here.
- **main** — stable v1.x release. Not touched until v2 is ready.

Create feature branches from `revamp`:
```bash
git checkout revamp
git checkout -b feat/memory-system
```

## Code standards

### SOLID (mandatory)

- **SRP** — One class, one responsibility. If a module does two things, split it.
- **OCP** — New capabilities via plugins and protocols. Never modify core to add a feature.
- **LSP** — Protocol implementations must be substitutable for the base protocol.
- **ISP** — Protocols must be narrow. If an implementer doesn't need half the methods, the protocol is too wide.
- **DIP** — Depend on protocols, not concrete implementations.

### OCP in practice

When adding a new capability:

```python
# GOOD — extend via protocol
class MyStore(MemoryStore):
    async def save(self, memory): ...
    # register via entry point: sr2.stores = {my_store = "..."}

# BAD — modify core to add special case
# in pipeline/engine.py:
# if store_type == "my_store":  # hardcoded branch
```

### DRY in practice

```python
# GOOD — shared type defined once
@dataclass(frozen=True)
class ResolvedContent:
    content: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

# Used by: ContentProvider.resolve(), PipelineEngine._resolve_provider(), tests

# BAD — duplicate dataclasses in different modules
```

### Error handling

All errors inherit from `SR2Error`. Use the specific error types:

```python
from sr2.core.errors import ConfigError, PluginNotFoundError, PipelineError

def load_config(path):
    if not Path(path).exists():
        raise ConfigError(f"Config file not found: {path}")
```

Never raise bare `Exception` or `ValueError` from SR2 code.

## Testing

### Run tests

```bash
cd /home/shepard/git/sr2
source .venv/bin/activate
pytest tests/ -v
```

### Test requirements

- Every new file gets a corresponding test file
- Test behavior through public APIs, not internals
- Mock only at system boundaries (DB, HTTP, external services)
- Internal refactoring must not break tests

### Test structure

```
tests/
  core/           # core models, errors
  config/         # config models, loader
  pipeline/       # engine, results
  plugins/        # registry, discovery
  degradation/    # circuit breaker
  memory/         # memory store, retrieval, extraction
  compaction/     # compaction engine, rules
  summarization/  # summarization engine
  cache/          # cache policies
  metrics/        # metric collection
  protocols/      # protocol contracts
  tokenization/   # token counting
  tools/          # tool management
  resolvers/      # content resolvers
```

### TDD cycle

1. Write failing test
2. Run to verify failure
3. Write minimal implementation
4. Run to verify pass
5. Refactor if needed
6. Commit with test + implementation together

## Commit conventions

```
type: short description

- feat: new feature or capability
- fix: bug fix
- test: new or modified tests
- docs: documentation changes
- refactor: code restructure with no behavior change
- chore: maintenance, deps, config
```

Examples:
```
feat: add memory store protocol with save/search/get/delete
test: add circuit breaker state transition tests
refactor: extract token counting into shared utility
```

## Review checklist

Before requesting review:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] No `# TODO` without a linked issue
- [ ] Protocols are `runtime_checkable`
- [ ] Public API is in `__init__.py` exports
- [ ] Module docstrings explain *why*, not *what*
- [ ] Error messages are actionable (tell user how to fix)

## Architecture decisions

When making a design choice, ask:

1. **Does this belong in SR2 core or a plugin?** If it's agent-specific behavior, it's a plugin.
2. **Is this a protocol or a concrete class?** If multiple implementations are possible, it's a protocol.
3. **Does this violate the facade boundary?** The harness should never need to know about internal subsystems.
4. **Is this SRP-compliant?** If the module name has "and" in its responsibility, split it.

## Project layout

```
/home/shepard/git/sr2/           # Server codebase
  src/sr2/                        # Source code
  tests/                          # Tests (mirrors src structure)
  pyproject.toml                  # Package config + entry points
  README.md                       # Architecture overview
  QUICKSTART.md                   # Getting started guide
  CONTRIBUTING.md                 # This file

/data/obsidian/projects/sr2/      # Project hub (plans, status)
  PLAN-sr2-v2-redesign.md         # Full redesign specification
  CLAUDE.md                       # Project instructions

/home/shepard/git/sr2-pro/        # Premium plugins (PostgreSQL, OTel, Prometheus)
```
