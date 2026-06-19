"""Tests for lazy/optional psycopg backend (bead spc-72).

Regression guard for: a missing ``psycopg`` must NOT break importing the whole
``sr2.memory`` package (or downstream ``sr2_spectre.runtime``). Instead:

  * ``import sr2.memory`` succeeds and still exposes ``InMemoryMemoryStore`` and
    the ``PostgresMemoryStore`` symbol even when psycopg is unavailable.
  * The psycopg import is LAZY — it is only attempted when a Postgres-backed
    store is actually instantiated. Selecting the Postgres backend without
    psycopg installed raises a clear, actionable error that NAMES psycopg, only
    at instantiation time — never a raw ``ModuleNotFoundError`` at package
    import.
  * Existing in-memory selection behaviour is unaffected.

psycopg IS installed in the test venv, so "absence" is simulated by blocking the
import (subprocess with an import hook for the true package-import case; a
``sys.modules`` sentinel + reload for the in-process instantiation case).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# 1. Package import survives a missing psycopg (true subprocess, real absence)
# ---------------------------------------------------------------------------


def _run_with_psycopg_blocked(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter where ``import psycopg`` is impossible.

    A ``sys.meta_path`` finder raises ModuleNotFoundError for psycopg before any
    sr2 import happens, faithfully reproducing the system-python3 environment
    that had no psycopg installed.
    """
    script = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        class _BlockPsycopg(MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "psycopg" or name.startswith("psycopg."):
                    raise ModuleNotFoundError("No module named 'psycopg'")
                return None

        sys.meta_path.insert(0, _BlockPsycopg())

        # Hard-prove psycopg really is unavailable in this interpreter.
        try:
            import psycopg  # noqa: F401
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("psycopg was importable; block failed")

        """
    ) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def test_import_sr2_memory_succeeds_without_psycopg():
    """``import sr2.memory`` must not raise when psycopg is absent."""
    result = _run_with_psycopg_blocked(
        """
        import sr2.memory  # must NOT raise ModuleNotFoundError: psycopg
        print("IMPORT_OK")
        """
    )
    assert result.returncode == 0, (
        "importing sr2.memory crashed without psycopg:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "IMPORT_OK" in result.stdout
    assert "No module named 'psycopg'" not in result.stderr


def test_inmemory_store_usable_without_psycopg():
    """InMemoryMemoryStore is exposed and constructable when psycopg is absent."""
    result = _run_with_psycopg_blocked(
        """
        from sr2.memory import InMemoryMemoryStore
        store = InMemoryMemoryStore()
        print("INMEMORY_OK")
        """
    )
    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "INMEMORY_OK" in result.stdout


def test_postgres_symbol_present_without_psycopg():
    """The PostgresMemoryStore symbol is still importable when psycopg is absent.

    The class object must exist (so callers can reference / type-check it); only
    *using* it should require psycopg.
    """
    result = _run_with_psycopg_blocked(
        """
        from sr2.memory import PostgresMemoryStore
        assert PostgresMemoryStore is not None
        print("SYMBOL_OK")
        """
    )
    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "SYMBOL_OK" in result.stdout


def test_instantiating_postgres_without_psycopg_raises_clear_error():
    """Selecting the Postgres backend without psycopg fails clearly, only on use.

    The error must name 'psycopg' so the operator knows exactly what to install,
    and it must surface at INSTANTIATION — not at package import.
    """
    result = _run_with_psycopg_blocked(
        """
        from sr2.memory import PostgresMemoryStore
        try:
            PostgresMemoryStore("postgresql://unused")
        except ModuleNotFoundError as exc:
            # A bare ModuleNotFoundError with no guidance is the failure mode we
            # are fixing. It must be a clearer error type / message instead.
            print("RAW_MODULE_NOT_FOUND:" + str(exc))
        except Exception as exc:
            msg = str(exc)
            assert "psycopg" in msg, f"error did not name psycopg: {msg!r}"
            print("CLEAR_ERROR_OK:" + type(exc).__name__)
        else:
            print("NO_ERROR")
        """
    )
    assert result.returncode == 0, (
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "CLEAR_ERROR_OK:" in result.stdout, (
        "instantiating PostgresMemoryStore without psycopg did not raise a "
        f"clear, psycopg-naming error. Output:\n{result.stdout}"
    )
    assert "RAW_MODULE_NOT_FOUND:" not in result.stdout
    assert "NO_ERROR" not in result.stdout


# ---------------------------------------------------------------------------
# 2. In-process: lazy import (psycopg present, but symbol-level proof)
# ---------------------------------------------------------------------------


def test_pg_store_module_imports_without_touching_psycopg():
    """Importing the pg_store module must not eagerly import psycopg.

    Simulated by hiding psycopg via a sys.modules sentinel and reloading the
    module: it must import fine; only construction should fail.
    """
    import importlib

    import sr2.memory.pg_store as pg_store

    saved = sys.modules.get("psycopg")
    sys.modules["psycopg"] = None  # type: ignore[assignment]  # forces ImportError on import
    try:
        reloaded = importlib.reload(pg_store)
        assert hasattr(reloaded, "PostgresMemoryStore")

        with pytest.raises(Exception) as excinfo:
            reloaded.PostgresMemoryStore("postgresql://unused")
        assert "psycopg" in str(excinfo.value)
        # Must NOT be a bare ModuleNotFoundError with no guidance.
        assert not isinstance(excinfo.value, ModuleNotFoundError) or (
            "install" in str(excinfo.value).lower()
        )
    finally:
        if saved is None:
            sys.modules.pop("psycopg", None)
        else:
            sys.modules["psycopg"] = saved
        importlib.reload(pg_store)


# ---------------------------------------------------------------------------
# 3. Existing in-memory selection behaviour is unaffected (psycopg present)
# ---------------------------------------------------------------------------


def test_inmemory_selection_unaffected():
    """Normal in-process import path still exposes a working InMemory store."""
    from sr2.memory import InMemoryMemoryStore, Memory, MemoryScope

    store = InMemoryMemoryStore()
    mem = Memory(content="hello", scope=MemoryScope.PRIVATE, tags=["t"])
    saved = store.save(mem)
    assert saved.content == "hello"
