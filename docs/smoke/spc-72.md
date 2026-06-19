# Smoke Runbook — spc-72 (lazy/optional psycopg backend)

**Proves:** `import sr2.memory` (and the `PostgresMemoryStore` symbol) succeeds when psycopg is NOT installed, `InMemoryMemoryStore` stays usable, and selecting the Postgres backend without psycopg fails with a clear, psycopg-naming error only at instantiation — never a raw `ModuleNotFoundError` at package import.
**Does NOT cover:** the live Postgres round-trip path (save/search against a real DB) — that is covered by `tests/test_pg_memory_store.py`, which needs a reachable test Postgres.

> Every command is on a single line. Copy one line at a time. No line continuations.

---

## 0. One-time setup

```bash
bash /tmp/spc72-smoke/setup.sh
```

```bash
cat /tmp/spc72-smoke/block_psycopg.py
```

**Why the block helper:** psycopg IS installed in the sr2 venv, so "absence" is simulated. `block_psycopg.py` installs a `sys.meta_path` finder that raises `ModuleNotFoundError` for `psycopg`, then runs whatever script path is passed as `argv[1]` under `runpy`. This faithfully reproduces the system-python3 environment that had no psycopg.

setup.sh contents (written for you — do not retype):

```bash
mkdir -p /tmp/spc72-smoke
```

```bash
printf '%s\n' "import runpy, sys" "from importlib.abc import MetaPathFinder" "class _Block(MetaPathFinder):" "    def find_spec(self, name, path, target=None):" "        if name == 'psycopg' or name.startswith('psycopg.'):" "            raise ModuleNotFoundError(\"No module named 'psycopg'\")" "        return None" "sys.meta_path.insert(0, _Block())" "runpy.run_path(sys.argv[1], run_name='__main__')" > /tmp/spc72-smoke/block_psycopg.py
```

---

## 1. Package imports without psycopg (the core regression)

```bash
printf '%s\n' "import sr2.memory" "print('IMPORT_OK')" > /tmp/spc72-smoke/s1.py
```

```bash
/home/shepard/git/sr2/.venv/bin/python /tmp/spc72-smoke/block_psycopg.py /tmp/spc72-smoke/s1.py
```

**Expect:** prints `IMPORT_OK`, exit code 0, NO `ModuleNotFoundError: No module named 'psycopg'` in output.

---

## 2. InMemoryMemoryStore still usable without psycopg

```bash
printf '%s\n' "from sr2.memory import InMemoryMemoryStore" "InMemoryMemoryStore()" "print('INMEMORY_OK')" > /tmp/spc72-smoke/s2.py
```

```bash
/home/shepard/git/sr2/.venv/bin/python /tmp/spc72-smoke/block_psycopg.py /tmp/spc72-smoke/s2.py
```

**Expect:** prints `INMEMORY_OK`, exit code 0. System survived the missing backend and degraded to in-memory.

---

## 3. Postgres backend selection fails clearly, only on use

```bash
printf '%s\n' "from sr2.memory import PostgresMemoryStore" "print('SYMBOL_OK', PostgresMemoryStore.__name__)" "import sys" "try:" "    PostgresMemoryStore('postgresql://unused')" "except ModuleNotFoundError as e:" "    print('RAW_MODULE_NOT_FOUND'); sys.exit(1)" "except Exception as e:" "    assert 'psycopg' in str(e), str(e)" "    print('CLEAR_ERROR_OK', type(e).__name__)" "else:" "    print('NO_ERROR'); sys.exit(1)" > /tmp/spc72-smoke/s3.py
```

```bash
/home/shepard/git/sr2/.venv/bin/python /tmp/spc72-smoke/block_psycopg.py /tmp/spc72-smoke/s3.py
```

**Expect:** prints `SYMBOL_OK PostgresMemoryStore` then `CLEAR_ERROR_OK ImportError`, exit code 0. The class symbol imported fine (no import-time crash); the error naming `psycopg` appeared only at instantiation. NO `RAW_MODULE_NOT_FOUND`, NO `NO_ERROR`.

---

## 4. Downstream sr2_spectre.runtime imports without crash

```bash
/home/shepard/git/sr2-spectre/.venv/bin/python -c "import sr2_spectre.runtime; print('SPECTRE_RUNTIME_OK')"
```

**Expect:** prints `SPECTRE_RUNTIME_OK` on the last line (LiteLLM botocore warnings above it are unrelated and fine). Exit code 0.

---

## 5. Automated regression suite (psycopg present in venv)

```bash
/home/shepard/git/sr2/.venv/bin/python -m pytest tests/test_memory_optional_psycopg.py -q
```

**Expect:** `6 passed`.

```bash
/home/shepard/git/sr2/.venv/bin/python -m pytest tests/test_memory_store.py tests/test_pg_memory_store.py tests/memory/ tests/resolvers/test_memory_resolver.py -q
```

**Expect:** all pass (the pg_store suite runs only if its test Postgres is reachable; otherwise it skips — both outcomes are green).

---

## 6. Teardown

```bash
rm -rf /tmp/spc72-smoke
```

---

## Pass criteria

| # | Expect |
|---|--------|
| 1 | `IMPORT_OK`, exit 0, no `ModuleNotFoundError: psycopg` |
| 2 | `INMEMORY_OK`, exit 0 |
| 3 | `SYMBOL_OK ...` then `CLEAR_ERROR_OK ImportError`, exit 0, no `RAW_MODULE_NOT_FOUND`/`NO_ERROR` |
| 4 | `SPECTRE_RUNTIME_OK`, exit 0 |
| 5 | `6 passed`; broader suites green |

All green → **spc-72 hardening verified; the dependent beads (spc-63/65/68/70 + image-gen work) no longer risk a startup crash from a missing psycopg in any sr2/sr2-spectre environment.**

**Caveat:** This smoke proves import-safety and clear-failure semantics. It does NOT exercise a live Postgres save/search round-trip — that path is covered by `tests/test_pg_memory_store.py` against a reachable test DB.
