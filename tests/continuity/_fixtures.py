"""Shared fixtures for the sr2-95 continuity cage (A-owned).

Model layer under test (declared landing spot, fixed by the bead):

    sr2.continuity.models

 exposing exactly the spec-interface symbols:

    ContinuityKind, ContinuityEventKind, ContinuityRecord, ContinuityView

ContextScope / ContextAuthority are NOT re-declared by this bead — they are
reused from ``sr2.pipeline.provenance`` (the spec interface references them
directly), which exists on the base commit.

The module-level import below is the deliberate red mechanism: until the
implementer creates the module, collection of each test file fails with an
ImportError naming exactly that module (the pattern prior sr2 beads used,
e.g. tests/config/test_continuity_config.py for sr2-94).

Contract the rows fix (internals beyond the spec interface are free):

- ``ContinuityRecord`` / ``ContinuityView`` are pydantic models with
  ``frozen=True`` semantics (assignment to any field raises; mutation of the
  mutable ``meta`` default must not be possible through the public surface
  either — pydantic's frozen model rejects ``model.meta[...] = ...``).
- Serialized bytes are taken with ``model_dump_json()`` (a public pydantic
  API present on any pydantic v2 BaseModel), so the byte-identity rows do
  not pin a store-private serialization helper.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sr2.pipeline.provenance import ContextAuthority, ContextScope
from sr2.continuity.models import (
    ContinuityEventKind,
    ContinuityKind,
    ContinuityRecord,
    ContinuityView,
)

# Re-exports so each test file imports from one place.
__all__ = [
    "ContextAuthority",
    "ContextScope",
    "ContinuityEventKind",
    "ContinuityKind",
    "ContinuityRecord",
    "ContinuityView",
    "make_record",
    "serialize",
]

_BASE_TIME = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

DEFAULT_SCOPE = ContextScope(owner_id="owner-1", agent_id="agent-1")


def make_record(
    id: str,
    subject_id: str,
    event_kind: "ContinuityEventKind",
    *,
    kind: ContinuityKind = ContinuityKind.DECISION,
    content: str | None = None,
    scope: ContextScope | None = None,
    authority: ContextAuthority = ContextAuthority.USER,
    supersedes: tuple[str, ...] = (),
    confidence: float = 0.9,
    salience: float = 0.5,
    created_at_offset_seconds: int = 0,
) -> ContinuityRecord:
    """Build a valid ContinuityRecord with sensible defaults.

    Every field carries a deterministic value (fixed base clock, explicit
    ids) so serialized bytes are reproducible across calls and across
    store sessions — the byte-identity and deterministic-rebuild rows
    depend on that.
    """
    return ContinuityRecord(
        id=id,
        subject_id=subject_id,
        event_kind=event_kind,
        kind=kind,
        content=content if content is not None else f"content of {id}",
        scope=scope if scope is not None else DEFAULT_SCOPE,
        authority=authority,
        provenance_ids=(),
        supersedes=supersedes,
        confidence=confidence,
        salience=salience,
        created_at=_BASE_TIME + timedelta(seconds=created_at_offset_seconds),
    )


def serialize(record: "ContinuityRecord") -> str:
    """Canonical serialized form of a record for byte-identity assertions.

    ``model_dump_json`` is a public pydantic v2 API inherited by every
    BaseModel, so this pins no store-private serialization.
    """
    return record.model_dump_json()
