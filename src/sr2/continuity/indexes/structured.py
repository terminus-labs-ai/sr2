"""Structured continuity index — metadata-driven scoring, no embeddings (sr2-98 / AC17).

Scores records using structured signals only: lexical overlap between query
and content, record kind relevance, salience, confidence, recency, and
project affinity. Deterministic by construction: same data and query produce
identical scores and ordering every time.

The index holds records in memory; truth remains in the immutable ledger.
Callers rebuild from the ledger on startup.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import AsyncIterator

from sr2.continuity.index import ContinuityCandidate, ContinuityHealth
from sr2.continuity.models import ContinuityRecord, ContinuityView
from sr2.continuity.policy import RetrievalContext


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stripped of punctuation."""
    return [
        word.strip(".,;:!?()[]{}\"'")
        for word in text.lower().split()
        if word.strip(".,;:!?()[]{}\"'")
    ]


def _lexical_overlap(query_tokens: list[str], content_tokens: list[str]) -> float:
    """Fraction of query tokens found in content (Jaccard-like, asymmetric)."""
    if not query_tokens:
        return 0.0
    content_set = set(content_tokens)
    matches = sum(1 for t in query_tokens if t in content_set)
    return matches / len(query_tokens)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


class StructuredIndex:
    """In-memory structured metadata index.

    Scoring components (all in [0, 1] before weighting):
    - ``lexical``: overlap between query and record content
    - ``salience``: the record's own salience field
    - ``confidence``: the record's confidence field
    - ``recency``: exponential decay from the most recent record
    - ``project``: 1.0 if the record shares a project with the query context
    """

    def __init__(self) -> None:
        self._records: list[ContinuityRecord] = []
        self._views: dict[str, ContinuityView] = {}
        self._latest_time: datetime | None = None

    async def append(self, records: list[ContinuityRecord]) -> None:
        for record in records:
            self._records.append(record)
            view = ContinuityView(
                subject_id=record.subject_id,
                current_record=record,
                status="active",
                history_ids=(record.id,),
            )
            self._views[record.id] = view
            if self._latest_time is None or record.created_at > self._latest_time:
                self._latest_time = record.created_at

    async def rebuild(self, records: AsyncIterator[ContinuityRecord]) -> None:
        self._records = []
        self._views = {}
        self._latest_time = None
        async for record in records:
            await self.append([record])

    async def search(
        self,
        context: RetrievalContext,
        limit: int,
    ) -> list[ContinuityCandidate]:
        if not self._records or limit <= 0:
            return []

        query_tokens = _tokenize(context.query)
        scored: list[tuple[float, dict[str, float], ContinuityView]] = []

        for record in self._records:
            content_tokens = _tokenize(record.content)
            lexical = _lexical_overlap(query_tokens, content_tokens)

            salience = record.salience
            confidence = record.confidence

            recency = self._recency_score(record)
            project = self._project_score(record, context)

            # Weighted combination: lexical dominates, others modulate.
            score = (
                lexical * 0.50
                + salience * 0.15
                + confidence * 0.10
                + recency * 0.15
                + project * 0.10
            )

            components = {
                "lexical": lexical,
                "salience": salience,
                "confidence": confidence,
                "recency": recency,
                "project": project,
            }

            view = self._views[record.id]
            scored.append((score, components, view))

        # Sort by score descending, then by record id for stable ties.
        scored.sort(key=lambda x: (-x[0], x[2].current_record.id))

        results: list[ContinuityCandidate] = []
        for score, components, view in scored[:limit]:
            candidate = ContinuityCandidate(
                record=view,
                score=score,
                score_components=components,
                supporting_entry_ids=view.current_record.provenance_ids,
                estimated_tokens=_estimate_tokens(view.current_record.content),
            )
            results.append(candidate)
        return results

    async def health(self) -> ContinuityHealth:
        return ContinuityHealth(status="healthy")

    def _recency_score(self, record: ContinuityRecord) -> float:
        """Exponential decay: 1.0 for the newest, approaching 0 for old."""
        if self._latest_time is None:
            return 0.5
        if record.created_at >= self._latest_time:
            return 1.0
        age_seconds = (self._latest_time - record.created_at).total_seconds()
        # Half-life of ~7 days
        half_life = 7 * 24 * 3600
        return math.exp(-math.log(2) * age_seconds / half_life)

    def _project_score(self, record: ContinuityRecord, context: RetrievalContext) -> float:
        """1.0 if the record shares at least one project with the context."""
        if not context.projects:
            return 0.0
        record_projects = set(record.scope.projects)
        context_projects = set(context.projects)
        if record_projects & context_projects:
            return 1.0
        return 0.0
