"""Backend-neutral ContinuityIndex protocol and shared value types (sr2-98 / AC17).

The spec defines one retrieval interface that every candidate backend
implements. Context assembly consumes only the types declared here; it never
sees a backend-specific object. Policy filters records before scoring —
candidates cannot bypass it.

Landing spot: ``sr2.continuity.index``.
"""

from __future__ import annotations

from typing import AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from sr2.continuity.models import ContinuityRecord, ContinuityView
from sr2.continuity.policy import RetrievalContext


# ---------------------------------------------------------------------------
# Value types (frozen pydantic models — no backend-specific fields)
# ---------------------------------------------------------------------------


class ContinuityCandidate(BaseModel, frozen=True):
    """One scored retrieval hit, backend-neutral by construction."""

    record: ContinuityView
    score: float
    score_components: dict[str, float]
    supporting_entry_ids: tuple[str, ...]
    estimated_tokens: int


class ContinuityFailure(BaseModel, frozen=True):
    """One named capability failure inside an index backend."""

    capability: Literal["archive", "consolidation", "situation", "policy", "index"]
    message: str
    retryable: bool


class ContinuityHealth(BaseModel, frozen=True):
    """Aggregate health of one index backend."""

    status: Literal["healthy", "degraded"]
    failures: tuple[ContinuityFailure, ...] = ()


# ---------------------------------------------------------------------------
# Protocol (runtime_checkable so isinstance works in tests)
# ---------------------------------------------------------------------------


@runtime_checkable
class ContinuityIndex(Protocol):
    """The common contract every retrieval backend satisfies.

    Implementations receive the same policy-eligible records, query, and
    token budget. Store truth remains in the immutable ledger; index state
    is rebuildable. Publish raw candidate metrics, not a selection verdict.
    """

    async def append(self, records: list[ContinuityRecord]) -> None: ...

    async def rebuild(self, records: AsyncIterator[ContinuityRecord]) -> None: ...

    async def search(
        self,
        context: RetrievalContext,
        limit: int,
    ) -> list[ContinuityCandidate]: ...

    async def health(self) -> ContinuityHealth: ...
