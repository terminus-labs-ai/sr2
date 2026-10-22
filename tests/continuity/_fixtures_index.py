"""Shared fixtures for the sr2-98 continuity-index cage (A-owned).

Builds ONLY against types that already exist on the base commit
(``sr2.continuity.models``, ``sr2.continuity.policy``,
``sr2.pipeline.provenance``). The not-yet-existing modules
(``sr2.continuity.index`` and friends) are imported by the cage files
themselves, so a red run names those modules and not this helper.

============================================================================
INTERFACE CONTRACT THIS CAGE FIXES (everything else is free)
============================================================================

The bead quotes the spec verbatim. Two surfaces locate the implementation
without pinning its concrete class names (the first existing name wins,
sr2-97 pattern), applied per module:

``sr2.continuity.indexes.structured``
    ``StructuredIndex`` / ``StructuredContinuityIndex`` / ``Index``
``sr2.continuity.indexes.vector_hybrid``
    ``VectorHybridIndex`` / ``VectorHybridContinuityIndex`` / ``Index``

The classes must satisfy the spec shape of ``ContinuityIndex``:

    async def append(self, records: list[ContinuityRecord]) -> None
    async def rebuild(self, records: AsyncIterator[ContinuityRecord]) -> None
    async def search(self, context: RetrievalContext, limit: int)
        -> list[ContinuityCandidate]
    async def health(self) -> ContinuityHealth

A class additionally requires at least one MORE parameter than ``self`` in
``__init__`` (a connection string, a backend handle, ...) is constructed via
:func:`construct` with the argument named in ``_EXTRA_CTOR_ARGS``. Which
optional parameters a backend takes beyond those is free; nothing else is
injected, because the row AC17-r11 pins that the benchmark feeds both
candidates the same plain inputs — no candidate-specific wiring.

============================================================================
TEST DATA
============================================================================

* All five records are FORMED events authored by the SAME agent, in the
  SAME project, for the SAME owner, so indexing and search have no
  policy/isolation variable — search hits are driven by content overlap
  with the query alone.
* CONTENT_MATCHING pairs with QUERY ("preferred testing framework?") with a
  clearly above-noise lexical overlap, and also embeds RECORD_SUPPORTING_ID
  so a candidate's ``supporting_entry_ids`` has an answer other than "empty"
  available.
* CONTENT_NONMATCHING shares nothing lexically with QUERY: it exists so
  ``limit`` cannot be satisfied by every record, and so an index that
  returned everything fails the limit rows.
* All timestamps are fixed; no randomness, no clock.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Sequence

from sr2.continuity.models import (
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
    ContinuityView,
)
from sr2.continuity.policy import RetrievalContext
from sr2.pipeline.provenance import ContextAuthority, ContextScope

__all__ = [
    "CONTENT_MATCHING",
    "CONTENT_NONMATCHING",
    "CONTEXT",
    "NOW",
    "OWNER_ID",
    "PROJECT_ID",
    "QUERY",
    "RECORD_MATCHING_ID",
    "RECORD_NONMATCHING_ID",
    "RECORD_SUPPORTING_ID",
    "RECORD_SUPPORTING_TEXT",
    "TOKEN_BUDGET",
    "candidate_classes",
    "context_for_limit_rows",
    "context_with_extra_overlap",
    "construct",
    "make_candidate_class",
    "make_record",
    "make_records",
    "make_view",
    "records_extra",
    "records_for_limit_rows",
    "records_matching",
    "view_for_record",
]

NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
OWNER_ID = "owner-98"
PROJECT_ID = "sr2"
QUERY = "preferred testing framework for the project"
TOKEN_BUDGET = 4096

RECORD_MATCHING_ID = "rec-matching-0001"
RECORD_NONMATCHING_ID = "rec-nonmatching-0001"
RECORD_SUPPORTING_ID = "entry-supporting-0001"
RECORD_SUPPORTING_TEXT = "session log: pytest adoption discussion"

CONTENT_MATCHING = (
    "Decision: adopt pytest as the preferred testing framework for the sr2 "
    "project, with asyncio support enabled for all asynchronous tests."
)
CONTENT_NONMATCHING = (
    "Reminder: reorder the pantry shelves and rotate the canned goods before "
    "the quarterly inventory count."
)


def make_record(
    id: str,
    content: str = CONTENT_MATCHING,
    *,
    kind: ContinuityKind = ContinuityKind.DECISION,
    created_at: datetime = NOW,
    provenance_ids: tuple[str, ...] = (RECORD_SUPPORTING_ID,),
) -> ContinuityRecord:
    scope = ContextScope(
        owner_id=OWNER_ID,
        agent_id="agent-coding",
        projects=(PROJECT_ID,),
        topics=("testing",),
    )
    return ContinuityRecord(
        id=id,
        subject_id=OWNER_ID,
        event_kind=ContinuityEventKind.FORMED,
        kind=kind,
        content=content,
        scope=scope,
        authority=ContextAuthority.USER,
        provenance_ids=provenance_ids,
        confidence=0.9,
        salience=0.5,
        created_at=created_at,
    )


def make_view(record: ContinuityRecord) -> ContinuityView:
    return ContinuityView(
        subject_id=record.subject_id,
        current_record=record,
        status="active",
        history_ids=(record.id,),
    )


#: view_for_record is an alias the rows use when they pipe a record through
#: the projection surface.
view_for_record = make_view


def _context(**overrides: Any) -> RetrievalContext:
    base: dict[str, Any] = dict(
        owner_id=OWNER_ID,
        agent_id="agent-coding",
        agent_tags=("coding",),
        projects=(PROJECT_ID,),
        frame_id=None,
        query=QUERY,
        token_budget=TOKEN_BUDGET,
    )
    base.update(overrides)
    return RetrievalContext(**base)


CONTEXT = _context()

#: Context whose query lexically favors the first record of the limit-row
#: sets: ``"pytest"`` appears only in that record, so under any sane
#: relevance scorer it ranks first, and a limit below the record count has
#: one knowable answer for its first element.
CONTEXT_EXTRA = _context(query="pytest fixtures asyncio")


def context_for_limit_rows() -> RetrievalContext:
    return CONTEXT_EXTRA


def context_with_extra_overlap() -> RetrievalContext:
    return CONTEXT_EXTRA


def records_matching(count: int = 3) -> list[ContinuityRecord]:
    """``count`` matching records, distinct ids, same content family."""
    return [
        make_record(
            f"rec-matching-{i:04d}",
            content=(
                f"{CONTENT_MATCHING} Note {i}: test-coverage thresholds and "
                "CI lanes follow the same pytest framework decision."
            ),
            created_at=NOW,
        )
        for i in range(count)
    ]


def records_extra(count: int = 3) -> list[ContinuityRecord]:
    return records_matching(count)


def records_for_limit_rows() -> list[ContinuityRecord]:
    """Three matching records first, then a clearly nonmatching filler.

    The limit rows (AC17-r7/r8) search with ``limit=2`` against these five:
    the filler must not evict a matching record, and ``len == 2`` with the
    pytest-favoring record at the head is the assertable outcome.
    """
    return records_matching(3) + [
        make_record(
            f"rec-filler-{i:04d}",
            content=(
                f"{CONTENT_NONMATCHING} Batch {i}: labels, bins, and shelf "
                "maps for the storage room."
            ),
        )
        for i in range(2)
    ]


#: Default fixture: one matching and one nonmatching record. Rows that need
#: more index whatever they list.
def make_records() -> list[ContinuityRecord]:
    return [
        make_record(RECORD_MATCHING_ID, content=CONTENT_MATCHING),
        make_record(
            RECORD_NONMATCHING_ID,
            content=CONTENT_NONMATCHING,
            kind=ContinuityKind.NEXT_ACTION,
        ),
    ]


#: Constructor arguments beyond the minimal positional set, per module.
#: A backend that needs a connection string gets the in-memory/default one
#: named here — identical for both candidates except for the backend path
#: itself (AC17-r11: same records, context and budget, no per-candidate
#: budget or tuning knobs).
_EXTRA_CTOR_ARGS: dict[str, Sequence[Any]] = {
    "sr2.continuity.indexes.structured": (),
    "sr2.continuity.indexes.vector_hybrid": (),
}


def make_candidate_class(module, name="Index") -> type:
    """Return the candidate index class from a located module."""
    for attr in (name, f"{name}ContinuityIndex", "Index"):
        cls = getattr(module, attr, None)
        if inspect.isclass(cls):
            return cls
    raise AssertionError(f"no index class found in {module.__name__}")


def candidate_classes(module, names: Sequence[str]) -> list[type]:
    for attr in names:
        cls = getattr(module, attr, None)
        if inspect.isclass(cls):
            return [cls]
    # Fall back to any class whose name ends in "Index".
    found = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__name__.endswith("Index")
    ]
    if not found:
        raise AssertionError(
            f"module {module.__name__} exports no index class "
            f"(looked for {', '.join(names)})"
        )
    return found


def construct(cls: type, module_name: str) -> Any:
    """Instantiate an index class with the shared, candidate-neutral args."""
    extra = tuple(_EXTRA_CTOR_ARGS.get(module_name, ()))
    try:
        return cls(*extra)
    except TypeError:
        # The backend demands more configuration. Try the two conventional
        # in-memory spellings before giving up; nothing candidate-specific
        # is ever invented here.
        for kwargs in ({"in_memory": True}, {"connection_string": ":memory:"}):
            try:
                return cls(*extra, **kwargs)
            except TypeError:
                continue
        raise


def records_iter(records: Sequence[ContinuityRecord]) -> AsyncIterator[ContinuityRecord]:
    """Async iterator over ``records`` — the contract of ``rebuild``."""

    async def _gen() -> AsyncIterator[ContinuityRecord]:
        for record in records:
            yield record

    return _gen()


def record_ids(records: Sequence[Any]) -> set[str]:
    """Accept records, views or candidates; return the record ids."""
    out: set[str] = set()
    for item in records:
        if isinstance(item, ContinuityRecord):
            out.add(item.id)
        elif isinstance(item, ContinuityView):
            out.add(item.current_record.id)
        else:  # a candidate, duck-typed before the class exists
            candidate_record = getattr(item, "record", None)
            # candidate.record may be a ContinuityView (spec) or a ContinuityRecord
            if isinstance(candidate_record, ContinuityView):
                rid = candidate_record.current_record.id
            elif isinstance(candidate_record, ContinuityRecord):
                rid = candidate_record.id
            else:
                rid = getattr(candidate_record, "id", None) or getattr(item, "id", None)
            if not isinstance(rid, str):
                raise AssertionError(f"cannot read a record id from {item!r}")
            out.add(rid)
    return out
