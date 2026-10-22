"""Immutable continuity models (sr2-95 / FR4, FR13, AC3).

Continuity records are events in an append-only ledger: correcting,
suppressing or superseding information never mutates an earlier record, it
appends a new one. ``ContinuityRecord`` is therefore a frozen pydantic model
(AC3-r8) and ``ContinuityView`` — a rebuildable projection over the event
log — is frozen as well (AC3-r9).

``ContextScope`` / ``ContextAuthority`` are reused unchanged from
``sr2.pipeline.provenance`` (independent dimensions: project association,
personal subject matter, agent identity, access, authority); this module
does not re-declare them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, WrapValidator
from typing_extensions import Annotated

from sr2.pipeline.provenance import ContextAuthority, ContextScope


class FrozenDict(dict):
    """dict subclass whose mutation methods raise TypeError.

    Pydantic's ``frozen=True`` blocks attribute assignment but not mutation
    of contained mutable values (AC3-r8 pins that ``record.meta[k] = v``
    must raise). Reads, equality, hashing of containers and JSON
    serialization behave exactly like ``dict``, so byte-identity rows are
    unaffected.
    """

    __slots__ = ()

    def _readonly(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("meta is read-only on a frozen ContinuityRecord")

    __setitem__ = _readonly
    __delitem__ = _readonly
    update = _readonly
    setdefault = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly


def _freeze_dict(value: Any, handler: Any) -> Any:
    result = handler(value)
    return result if isinstance(result, FrozenDict) else FrozenDict(result)


#: ``dict[str, Any]`` that validates into a mutation-proof ``FrozenDict``.
FrozenDictMeta = Annotated[dict[str, Any], WrapValidator(_freeze_dict)]


class ContinuityKind(str, Enum):
    PERSONAL_FACT = "personal_fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    CORRECTION = "correction"
    GOAL = "goal"
    NEXT_ACTION = "next_action"
    COMMITMENT = "commitment"
    CONSTRAINT = "constraint"
    PROJECT_FACT = "project_fact"
    EPISODE = "episode"


class ContinuityEventKind(str, Enum):
    FORMED = "formed"
    REINFORCED = "reinforced"
    VERIFIED = "verified"
    SUPERSEDED = "superseded"
    DORMANT = "dormant"
    SUPPRESSED = "suppressed"
    UNSUPPRESSED = "unsuppressed"


class ContinuityRecord(BaseModel, frozen=True):
    """One immutable event in a subject's continuity chain (FR4).

    Never updated, never deleted: corrections append a record naming the
    superseded ids in ``supersedes``; suppression is a SUPPRESSED event.
    """

    id: str
    subject_id: str
    event_kind: ContinuityEventKind
    kind: ContinuityKind
    content: str
    scope: ContextScope
    authority: ContextAuthority
    provenance_ids: tuple[str, ...]
    supersedes: tuple[str, ...] = ()
    confidence: float
    salience: float
    created_at: datetime
    meta: FrozenDictMeta = {}

    def model_post_init(self, __context: Any) -> None:
        # Pydantic does not run field validators for an omitted field whose
        # default is injected verbatim — guarantee the stored ``meta`` is
        # always mutation-proof, default included.
        if not isinstance(self.__dict__.get("meta"), FrozenDict):
            self.__dict__["meta"] = FrozenDict(self.__dict__.get("meta") or {})


class ContinuityView(BaseModel, frozen=True):
    """Current projection for one subject, rebuilt from the event log.

    A cache, not truth: it is reconstructible from the immutable ledger
    (FR4) and excludes superseded / suppressed records by default while
    ``history_ids`` keeps the full chain for historical queries (FR13).
    """

    subject_id: str
    current_record: ContinuityRecord
    status: Literal["active", "tentative", "dormant", "superseded", "suppressed"]
    history_ids: tuple[str, ...]
