"""sr2-99 / AC18 — graph continuity index (unit layer of the cage).

Case table rows: AC18-r1 through AC18-r13.

Red mechanism: the module-level import of ``sr2.continuity.indexes.graph``;
until C creates that module, collection of this file fails with ImportError.

WHAT THIS CAGE FIXES
====================

The spec's AC18: "The graph implementation satisfies the same ContinuityIndex
and benchmark contracts, publishes raw per-scenario results plus
ingestion/lineage diagnostics, and does not become a source of truth
independent from the immutable ledger."

The implementation class is located by name: ``GraphIndex`` /
``GraphContinuityIndex`` / ``Index`` in ``sr2.continuity.indexes.graph``.

Graph-specific surfaces beyond the base ContinuityIndex protocol:

* ``diagnostics() -> GraphDiagnostics`` — ingestion/lineage metrics.
* ``GraphDiagnostics`` — a frozen pydantic model with at minimum:
    - ``records_ingested: int`` — total records appended since last rebuild
    - ``entities_extracted: int`` — total entity nodes created
    - ``relations_extracted: int`` — total relation edges created
    - ``ingestion_loss: int`` — records yielding zero entities AND zero relations
    - ``lineage_coverage: float`` — fraction of returned candidates whose
      supporting_entry_ids are non-empty (computed after search)

The diagnostics are a projection of the index state, not a second truth store.
"""

from __future__ import annotations

import inspect
from typing import Any

from sr2.continuity.index import ContinuityCandidate, ContinuityHealth, ContinuityIndex
from sr2.continuity.indexes import graph as graph_module
from tests.continuity._fixtures_index import (
    CONTENT_MATCHING,
    CONTEXT,
    RECORD_MATCHING_ID,
    candidate_classes,
    construct,
    context_for_limit_rows,
    make_record,
    make_records,
    record_ids,
    records_extra,
    records_for_limit_rows,
    records_matching,
    records_iter,
)

GraphIndex = candidate_classes(
    graph_module, ("GraphIndex", "GraphContinuityIndex", "Index")
)[0]


def _in_backend_namespace(obj: Any) -> bool:
    """True when obj is DEFINED in the graph backend module."""
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str):
        return False
    return module == graph_module.__name__ or module.startswith(
        graph_module.__name__ + "."
    )


def _assert_candidate(candidate: Any) -> None:
    """AC18 candidate shape check."""
    assert isinstance(candidate, ContinuityCandidate), (
        f"search must yield ContinuityCandidate, got {type(candidate)!r}"
    )
    assert isinstance(candidate.score, float)
    assert isinstance(candidate.score_components, dict)
    for key, value in candidate.score_components.items():
        assert isinstance(key, str) and isinstance(value, float), (
            f"score_components[{key!r}] = {value!r} is not a float"
        )
    assert isinstance(candidate.supporting_entry_ids, tuple)
    assert all(isinstance(e, str) for e in candidate.supporting_entry_ids)
    assert isinstance(candidate.estimated_tokens, int)
    assert not isinstance(candidate.estimated_tokens, bool)


async def _index_with(*records: Any) -> Any:
    index = construct(GraphIndex, graph_module.__name__)
    await index.append(list(records))
    return index


# ---------------------------------------------------------------------------
# AC18-r1 — satisfies the common ContinuityIndex contract
# ---------------------------------------------------------------------------


def test_ac18_r1_graph_index_satisfies_continuity_index_protocol():
    index = construct(GraphIndex, graph_module.__name__)
    assert isinstance(index, ContinuityIndex), (
        "GraphIndex must satisfy the runtime-checkable ContinuityIndex protocol"
    )
    for verb in ("append", "rebuild", "search", "health"):
        method = getattr(index, verb, None)
        assert callable(method), f"GraphIndex.{verb} is missing or not callable"
        assert inspect.iscoroutinefunction(method), (
            f"GraphIndex.{verb} must be an async method"
        )


# ---------------------------------------------------------------------------
# AC18-r2 — append indexes records; search returns them
# ---------------------------------------------------------------------------


async def test_ac18_r2_append_then_search_returns_indexed_records():
    records = make_records()
    index = await _index_with(*records)

    results = await index.search(CONTEXT, limit=10)
    assert isinstance(results, list)
    hits = record_ids(results)
    assert RECORD_MATCHING_ID in hits, (
        f"appended record {RECORD_MATCHING_ID!r} is not retrievable; hits={hits}"
    )
    for candidate in results:
        _assert_candidate(candidate)


async def test_ac18_r2_support_append_scales_to_many_records():
    records = records_matching(12)
    index = await _index_with(*records)
    results = await index.search(CONTEXT, limit=50)
    assert record_ids(records).issubset(record_ids(results)), (
        "every appended matching record must be retrievable"
    )


# ---------------------------------------------------------------------------
# AC18-r3 — rebuild from async iterator; replaces prior contents
# ---------------------------------------------------------------------------


async def test_ac18_r3_rebuild_from_async_iterator_then_search():
    index = construct(GraphIndex, graph_module.__name__)
    await index.rebuild(records_iter(make_records()))

    results = await index.search(CONTEXT, limit=10)
    assert RECORD_MATCHING_ID in record_ids(results), (
        "records replayed through rebuild() must be searchable"
    )


async def test_ac18_r3_support_rebuild_replaces_prior_contents():
    old = [make_record("rec-old-0001", content=CONTENT_MATCHING)]
    new = [make_record("rec-new-0001", content=CONTENT_MATCHING)]

    index = await _index_with(*old)
    await index.rebuild(records_iter(new))

    hits = record_ids(await index.search(CONTEXT, limit=50))
    assert "rec-new-0001" in hits
    assert "rec-old-0001" not in hits, (
        f"rebuild() rebuilds from scratch; stale records must not survive (hits={hits})"
    )


# ---------------------------------------------------------------------------
# AC18-r4 — search respects limit
# ---------------------------------------------------------------------------


async def test_ac18_r4_search_returns_typed_candidates_and_respects_limit():
    records = records_for_limit_rows()
    index = await _index_with(*records)
    context = context_for_limit_rows()

    results = await index.search(context, limit=2)
    assert isinstance(results, list)
    assert len(results) == 2, f"limit=2 must yield exactly 2, got {len(results)}"
    for candidate in results:
        _assert_candidate(candidate)
    assert len({c.record.current_record.id for c in results}) == 2

    wide = await index.search(context, limit=5)
    assert len(wide) == 5


# ---------------------------------------------------------------------------
# AC18-r5 — health
# ---------------------------------------------------------------------------


async def test_ac18_r5_health_reports_status_and_failures():
    index = await _index_with(*make_records())
    health = await index.health()

    assert isinstance(health, ContinuityHealth)
    assert health.status in ("healthy", "degraded")
    assert isinstance(health.failures, tuple)


# ---------------------------------------------------------------------------
# AC18-r6 — backend neutrality: no graph-specific types in candidates
# ---------------------------------------------------------------------------


def _no_backend_attribute(obj: Any, seen: set[int]) -> str | None:
    if id(obj) in seen:
        return None
    seen.add(id(obj))
    if _in_backend_namespace(obj):
        return f":{obj!r}"
    if isinstance(obj, (str, bytes, int, float, complex, type(None), bool)):
        return None
    for name, value in (vars(obj).items() if hasattr(obj, "__dict__") else []):
        leak = _no_backend_attribute(value, seen)
        if leak is not None:
            return f".{name}{leak}"
    if isinstance(obj, dict):
        for key, value in obj.items():
            leak = _no_backend_attribute(key, seen)
            if leak is not None:
                return f"key{leak}"
            leak = _no_backend_attribute(value, seen)
            if leak is not None:
                return f"[{key!r}]{leak}"
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            leak = _no_backend_attribute(item, seen)
            if leak is not None:
                return f"{leak}"
    return None


async def test_ac18_r6_search_publishes_only_backend_neutral_types():
    records = make_records() + records_matching(3)
    index = await _index_with(*records)

    results = await index.search(CONTEXT, limit=10)
    assert results, "fixture must produce hits"
    for candidate in results:
        assert type(candidate) is ContinuityCandidate, (
            f"must return ContinuityCandidate, not a backend subclass {type(candidate)!r}"
        )
        leak = _no_backend_attribute(candidate, set())
        assert leak is None, (
            f"graph-backend-specific object leaked into candidate at {leak}"
        )


# ---------------------------------------------------------------------------
# AC18-r7 — determinism
# ---------------------------------------------------------------------------


async def test_ac18_r7_repeated_searches_publish_identical_scores_and_order():
    records = records_matching(4) + [
        make_record(
            "rec-filler-9001",
            content="Restock labels and inventory bins for the storage room.",
        )
    ]
    index = await _index_with(*records)

    first = await index.search(CONTEXT, limit=10)
    second = await index.search(CONTEXT, limit=10)

    assert [c.record.current_record.id for c in first] == [
        c.record.current_record.id for c in second
    ]
    assert [c.score for c in first] == [c.score for c in second]
    assert [dict(c.score_components) for c in first] == [
        dict(c.score_components) for c in second
    ]
    scores = [c.score for c in first]
    assert all(a >= b for a, b in zip(scores, scores[1:])), "scores must be non-increasing"


# ---------------------------------------------------------------------------
# AC18-r8 — empty index
# ---------------------------------------------------------------------------


async def test_ac18_r8_search_on_empty_index_returns_empty_list():
    index = construct(GraphIndex, graph_module.__name__)
    results = await index.search(CONTEXT, limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# AC18-r9 — ingestion diagnostics published
# ---------------------------------------------------------------------------


async def test_ac18_r9_diagnostics_reports_ingestion_counts():
    """The graph index publishes ingestion diagnostics beyond the base protocol."""
    records = make_records()
    index = await _index_with(*records)

    assert hasattr(index, "diagnostics"), (
        "GraphIndex must expose a diagnostics() method for AC18"
    )
    diag = index.diagnostics()

    assert hasattr(diag, "records_ingested"), "diagnostics must report records_ingested"
    assert hasattr(diag, "entities_extracted"), "diagnostics must report entities_extracted"
    assert hasattr(diag, "relations_extracted"), "diagnostics must report relations_extracted"
    assert hasattr(diag, "ingestion_loss"), "diagnostics must report ingestion_loss"

    assert diag.records_ingested == len(records)
    assert isinstance(diag.entities_extracted, int)
    assert isinstance(diag.relations_extracted, int)
    assert isinstance(diag.ingestion_loss, int)
    assert diag.records_ingested >= diag.ingestion_loss


# ---------------------------------------------------------------------------
# AC18-r10 — lineage diagnostics: provenance_ids visible in candidates
# ---------------------------------------------------------------------------


async def test_ac18_r10_candidates_carry_source_lineage():
    """supporting_entry_ids must trace back to the record's provenance_ids."""
    records = records_matching(3)
    index = await _index_with(*records)

    results = await index.search(CONTEXT, limit=10)
    assert results, "fixture must produce hits for lineage check"
    for candidate in results:
        record = candidate.record.current_record
        # The candidate's supporting_entry_ids must include the record's provenance.
        assert set(record.provenance_ids).issubset(set(candidate.supporting_entry_ids)), (
            f"candidate for {record.id!r} must include its provenance_ids "
            f"{record.provenance_ids} in supporting_entry_ids, "
            f"got {candidate.supporting_entry_ids}"
        )


# ---------------------------------------------------------------------------
# AC18-r11 — ingestion loss visible for records with no extractable structure
# ---------------------------------------------------------------------------


async def test_ac18_r11_ingestion_loss_counted_for_unstructured_records():
    """Records that yield zero entities AND zero relations count as ingestion loss."""
    # A record with content so devoid of structure that no entity or relation
    # can be extracted from it.
    unstructured = make_record(
        "rec-unstructured-0001",
        content="... !!! ??? ,,, ;;;",
    )
    structured = records_matching(2)

    index = await _index_with(unstructured, *structured)
    diag = index.diagnostics()

    # The unstructured record contributes to ingestion_loss.
    assert diag.ingestion_loss >= 1, (
        f"unstructured record should count as ingestion loss; got {diag.ingestion_loss}"
    )
    assert diag.records_ingested == 3  # 1 unstructured + 2 matching


# ---------------------------------------------------------------------------
# AC18-r12 — graph candidate registered in benchmark registry
# ---------------------------------------------------------------------------


def test_ac18_r12_graph_registered_in_benchmark_candidates():
    """The evals/continuity/candidates.py registry includes the graph backend."""
    from evals.continuity import candidates as candidates_module

    # Find the registry (same pattern as structured/vector tests).
    _REGISTRY_NAMES = (
        "CANDIDATES",
        "CANDIDATE_REGISTRY",
        "REGISTRY",
        "INDEX_CANDIDATES",
        "CONTINUITY_CANDIDATES",
    )
    registry = None
    for name in _REGISTRY_NAMES:
        value = getattr(candidates_module, name, None)
        if isinstance(value, dict) and len(value) >= 3:
            registry = value
            break
    assert registry is not None, (
        "evals/continuity/candidates.py must export a registry with >=3 entries"
    )
    # The registry must name a graph entry.
    graph_keys = [k for k in registry if "graph" in k.lower()]
    assert graph_keys, (
        f"registry must include a graph candidate; keys={list(registry.keys())}"
    )


async def test_ac18_r12_registry_graph_entry_speaks_the_contract():
    """The registered graph entry yields a working ContinuityIndex."""
    from evals.continuity import candidates as candidates_module

    registry = candidates_module.CANDIDATES
    graph_keys = [k for k in registry if "graph" in k.lower()]
    assert graph_keys
    factory = registry[graph_keys[0]]

    instance = factory()
    assert isinstance(instance, ContinuityIndex)

    await instance.append([make_record("rec-registry-graph-001", content=CONTENT_MATCHING)])
    results = await instance.search(CONTEXT, limit=5)
    assert "rec-registry-graph-001" in record_ids(results)
    assert all(isinstance(c, ContinuityCandidate) for c in results)


# ---------------------------------------------------------------------------
# AC18-r13 — graph is a projection, not a source of truth (rebuild fidelity)
# ---------------------------------------------------------------------------


async def test_ac18_r13_rebuild_from_ledger_produces_same_search_results():
    """Full rebuild from the same records produces identical results to append."""
    records = records_matching(4)

    # Build via append.
    index_append = await _index_with(*records)
    results_append = await index_append.search(CONTEXT, limit=10)

    # Build via rebuild.
    index_rebuild = construct(GraphIndex, graph_module.__name__)
    await index_rebuild.rebuild(records_iter(records))
    results_rebuild = await index_rebuild.search(CONTEXT, limit=10)

    assert [c.record.current_record.id for c in results_append] == [
        c.record.current_record.id for c in results_rebuild
    ], "rebuild must produce identical ordering to incremental append"
    assert [c.score for c in results_append] == [c.score for c in results_rebuild], (
        "rebuild must produce identical scores to incremental append"
    )
