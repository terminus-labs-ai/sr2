"""sr2-98 / AC17 — vector/hybrid continuity index (unit layer of the cage).

Case table rows transcribed here: AC17-r2, AC17-r4, AC17-r6, AC17-r8,
AC17-r10, AC17-r11 (vector half), AC17-r12 (vector half), AC17-r13, AC17-r15.

Red mechanism (deliberate): the module-level imports below of the
not-yet-existing modules; until C creates them, collection of this file
fails with an ImportError naming exactly ``sr2.continuity.index`` (the
sr2-94/95/96/97 pattern).

============================================================================
WHAT THIS CAGE FIXES
============================================================================

Same spec types and protocol contract as the structured test. The
implementation class is located by name: ``VectorHybridIndex`` /
``VectorHybridContinuityIndex`` / ``Index`` in
``sr2.continuity.indexes.vector_hybrid``.

Fixed, per the spec clause "publish raw per-scenario results without
leaking backend-specific types into context assembly":

* a search hit is exactly a ``ContinuityCandidate`` (AC17-r8);
* nothing anywhere in a returned candidate carries a name bound to the
  vector-hybrid backend module (AC17-r11);
* repeated searches over unchanged data publish the same raw numbers and
  order (AC17-r15), because the benchmark consumes them verbatim.

Free: the embedding model, similarity metric, storage layout, constructor
signature beyond the minimal neutral construction the fixture supplies.
"""

from __future__ import annotations

import inspect
from typing import Any

from sr2.continuity.index import ContinuityCandidate, ContinuityHealth, ContinuityIndex
from sr2.continuity.indexes import vector_hybrid as vector_module
from tests.continuity._fixtures_index import (
    CONTENT_MATCHING,
    CONTEXT,
    RECORD_MATCHING_ID,
    candidate_classes,
    construct,
    context_for_limit_rows,
    make_candidate_class,
    make_record,
    make_records,
    record_ids,
    records_extra,
    records_for_limit_rows,
    records_matching,
)

VectorHybridIndex = candidate_classes(
    vector_module,
    ("VectorHybridIndex", "VectorHybridContinuityIndex", "Index"),
)[0]


def _in_backend_namespace(obj: Any) -> bool:
    """True when ``obj`` is an object DEFINED in the vector-hybrid backend's
    module — as opposed to a shared type merely imported there."""
    module = getattr(obj, "__module__", None)
    if not isinstance(module, str):
        return False
    return module == vector_module.__name__ or module.startswith(
        vector_module.__name__ + "."
    )


def _assert_candidate(candidate: Any) -> None:
    """AC17-r8's shape check."""
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
    index = construct(VectorHybridIndex, vector_module.__name__)
    await index.append(list(records))
    return index


# ---------------------------------------------------------------------------
# AC17-r2 — satisfies the common ContinuityIndex contract
# ---------------------------------------------------------------------------


def test_ac17_r2_vector_hybrid_index_satisfies_continuity_index_protocol():
    index = construct(VectorHybridIndex, vector_module.__name__)
    assert isinstance(index, ContinuityIndex), (
        "VectorHybridIndex must satisfy the runtime-checkable ContinuityIndex"
        " protocol (append / rebuild / search / health)"
    )
    for verb in ("append", "rebuild", "search", "health"):
        method = getattr(index, verb, None)
        assert callable(method), f"VectorHybridIndex.{verb} is missing or not callable"
        assert inspect.iscoroutinefunction(method), (
            f"VectorHybridIndex.{verb} must be an async method per the spec shape"
        )


# ---------------------------------------------------------------------------
# AC17-r4 — append indexes records; search returns them
# ---------------------------------------------------------------------------


async def test_ac17_r4_append_then_search_returns_indexed_records():
    records = make_records()
    index = await _index_with(*records)

    results = await index.search(CONTEXT, limit=10)
    assert isinstance(results, list)
    hits = record_ids(results)
    assert RECORD_MATCHING_ID in hits, (
        f"appended record {RECORD_MATCHING_ID!r} (content overlaps the query) "
        f"is not retrievable; hits={hits}"
    )
    for candidate in results:
        _assert_candidate(candidate)


async def test_ac17_r4_support_append_scales_to_many_records():
    records = records_matching(12)
    index = await _index_with(*records)
    results = await index.search(CONTEXT, limit=50)
    assert record_ids(records).issubset(record_ids(results)), (
        "every appended matching record must be retrievable "
        "when the limit does not bind"
    )


# ---------------------------------------------------------------------------
# AC17-r6 — rebuild from an async iterator
# ---------------------------------------------------------------------------


async def test_ac17_r6_rebuild_from_async_iterator_then_search():
    from tests.continuity._fixtures_index import records_iter as _records_iter

    index = construct(VectorHybridIndex, vector_module.__name__)
    await index.rebuild(_records_iter(make_records()))

    results = await index.search(CONTEXT, limit=10)
    assert RECORD_MATCHING_ID in record_ids(results), (
        "records replayed through rebuild() must be searchable"
    )


async def test_ac17_r6_support_rebuild_replaces_prior_contents():
    """Rebuild is from scratch: replaying one record drops the old ones."""
    from tests.continuity._fixtures_index import records_iter as _records_iter

    old = [make_record("rec-old-0001", content=CONTENT_MATCHING)]
    new = [make_record("rec-new-0001", content=CONTENT_MATCHING)]

    index = await _index_with(*old)
    await index.rebuild(_records_iter(new))

    hits = record_ids(await index.search(CONTEXT, limit=50))
    assert "rec-new-0001" in hits
    assert "rec-old-0001" not in hits, (
        "rebuild() rebuilds the index from scratch; stale pre-rebuild "
        f"records must not be searchable (hits={hits})"
    )


# ---------------------------------------------------------------------------
# AC17-r8 — candidate shape and limit respected
# ---------------------------------------------------------------------------


async def test_ac17_r8_search_returns_typed_candidates_and_respects_limit():
    records = records_for_limit_rows()
    index = await _index_with(*records)
    context = context_for_limit_rows()

    results = await index.search(context, limit=2)
    assert isinstance(results, list)
    assert len(results) == 2, f"limit=2 must yield exactly 2 candidates, got {len(results)}"
    for candidate in results:
        _assert_candidate(candidate)
    assert len({c.record.current_record.id for c in results}) == 2, "no duplicate records in one result"

    # The head must be a lexically matching record.
    assert results[0].record.current_record.id in record_ids(records[:3])

    # Growing the limit grows the result set without breaking shape.
    wide = await index.search(context, limit=5)
    assert len(wide) == 5
    assert all(isinstance(c, ContinuityCandidate) for c in wide)


# ---------------------------------------------------------------------------
# AC17-r10 — health
# ---------------------------------------------------------------------------


async def test_ac17_r10_health_reports_status_and_failures():
    index = await _index_with(*make_records())
    health = await index.health()

    assert isinstance(health, ContinuityHealth)
    assert health.status in ("healthy", "degraded")
    assert isinstance(health.failures, tuple)
    for failure in health.failures:
        assert isinstance(failure.message, str) and failure.message
        assert isinstance(failure.retryable, bool)


# ---------------------------------------------------------------------------
# AC17-r11 (vector half) — backend-neutral candidate types
# ---------------------------------------------------------------------------


def _no_backend_attribute(obj: Any, seen: set[int]) -> str | None:
    """Walk a candidate's published surface; name any backend-defined attr."""
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


async def test_ac17_r11_search_publishes_only_backend_neutral_types():
    records = make_records() + records_matching(3)
    index = await _index_with(*records)

    results = await index.search(CONTEXT, limit=10)
    assert results, "fixture must produce hits for this check to mean anything"
    for candidate in results:
        assert type(candidate) is ContinuityCandidate, (
            f"raw per-scenario results must be ContinuityCandidate, not "
            f"a backend subclass {type(candidate)!r}"
        )
        leak = _no_backend_attribute(candidate, set())
        assert leak is None, (
            f"vector-hybrid-backend-specific object leaked into the candidate "
            f"at {leak} — context assembly must see only the shared types"
        )


# ---------------------------------------------------------------------------
# AC17-r13 — empty index
# ---------------------------------------------------------------------------


async def test_ac17_r13_search_on_empty_index_returns_empty_list():
    index = construct(VectorHybridIndex, vector_module.__name__)
    results = await index.search(CONTEXT, limit=5)
    assert results == []


# ---------------------------------------------------------------------------
# AC17-r15 — deterministic raw scores and ordering
# ---------------------------------------------------------------------------


async def test_ac17_r15_repeated_searches_publish_identical_scores_and_order():
    records = records_matching(4) + [
        make_record(
            "rec-filler-9001",
            content="Restock labels and inventory bins for the storage room.",
        )
    ]
    index = await _index_with(*records)

    first = await index.search(CONTEXT, limit=10)
    second = await index.search(CONTEXT, limit=10)

    assert [c.record.current_record.id for c in first] == [c.record.current_record.id for c in second], (
        "ordering must be deterministic across identical searches"
    )
    assert [c.score for c in first] == [c.score for c in second]
    assert [dict(c.score_components) for c in first] == [
        dict(c.score_components) for c in second
    ]
    assert [c.supporting_entry_ids for c in first] == [
        c.supporting_entry_ids for c in second
    ]
    # Scores are ranked (the benchmark consumes the raw ordering).
    scores = [c.score for c in first]
    assert all(a >= b for a, b in zip(scores, scores[1:])), "scores must be non-increasing"


# ---------------------------------------------------------------------------
# AC17-r12 — benchmark registration (vector half)
# ---------------------------------------------------------------------------

_REGISTRY_NAMES = (
    "CANDIDATES",
    "CANDIDATE_REGISTRY",
    "REGISTRY",
    "INDEX_CANDIDATES",
    "CONTINUITY_CANDIDATES",
    "candidates",
    "registry",
    "get_candidates",
    "candidate_registry",
)


def _registry_entries() -> list[Any]:
    """The evals registry: importable cold, holding both candidates."""
    from evals.continuity import candidates as candidates_module

    for name in _REGISTRY_NAMES:
        value = getattr(candidates_module, name, None)
        if value is None:
            continue
        if callable(value) and not isinstance(value, (list, tuple, dict)):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, dict):
            entries = list(value.values())
        elif isinstance(value, (list, tuple)):
            entries = list(value)
        else:
            continue
        if len(entries) >= 2:
            return entries
    raise AssertionError(
        "evals/continuity/candidates.py exports no registry (list, tuple, "
        "dict, or zero-arg factory returning one) containing both the "
        "structured and the vector/hybrid candidate"
    )


def _vector_entries(entries: list[Any]) -> list[Any]:
    return [e for e in entries if "vector" in str(e).lower() or "hybrid" in str(e).lower()]


def test_ac17_r12_registry_offers_a_vector_hybrid_candidate():
    entries = _registry_entries()
    assert _vector_entries(entries), (
        f"the candidates registry must name the vector/hybrid candidate; "
        f"registry={entries!r}"
    )


async def test_ac17_r12_vector_hybrid_registry_entry_speaks_the_contract():
    """The registry's vector/hybrid entry satisfies ContinuityIndex and
    answers the shared fixture query."""
    entries = _registry_entries()
    picked = _vector_entries(entries)
    assert picked, "no vector/hybrid entry in the registry"
    entry = picked[0]

    instance: Any = None
    if isinstance(entry, ContinuityIndex):
        instance = entry
    elif callable(entry):
        try:
            instance = entry()
        except TypeError:
            instance = construct(entry, vector_module.__name__)
    if instance is None:
        raise AssertionError(f"registry entry {entry!r} is neither an index nor a factory")

    assert isinstance(instance, ContinuityIndex), (
        f"registry entry {entry!r} does not yield a ContinuityIndex: {instance!r}"
    )
    await instance.append([make_record("rec-registry-0002", content=CONTENT_MATCHING)])
    results = await instance.search(CONTEXT, limit=5)
    assert "rec-registry-0002" in record_ids(results), (
        "the registered vector/hybrid candidate must answer the shared fixture "
        "query like any other ContinuityIndex"
    )
    assert all(isinstance(c, ContinuityCandidate) for c in results)
