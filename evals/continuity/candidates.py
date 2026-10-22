"""Continuity index candidate registry for the versioned benchmark (sr2-98 / AC17).

Both candidates receive the same policy-eligible records, query, and token
budget. Store truth remains in the immutable ledger; index state is
rebuildable. This module publishes raw candidate metrics, not a selection
verdict.

The benchmark runner (sr2-91, sr2-100) imports this registry and runs each
candidate against identical scenarios with no candidate-specific inputs or
budgets.
"""

from __future__ import annotations

from typing import Callable

from sr2.continuity.index import ContinuityIndex
from sr2.continuity.indexes.graph import GraphIndex
from sr2.continuity.indexes.structured import StructuredIndex
from sr2.continuity.indexes.vector_hybrid import VectorHybridIndex


def _make_structured() -> ContinuityIndex:
    return StructuredIndex()


def _make_vector_hybrid() -> ContinuityIndex:
    return VectorHybridIndex()


def _make_graph() -> ContinuityIndex:
    return GraphIndex()


#: Registry of candidate index factories. Each entry is a zero-argument
#: callable that produces a fresh ContinuityIndex instance. The benchmark
#: runner calls each factory once per scenario to ensure isolation.
CANDIDATES: dict[str, Callable[[], ContinuityIndex]] = {
    "structured": _make_structured,
    "vector_hybrid": _make_vector_hybrid,
    "graph": _make_graph,
}
