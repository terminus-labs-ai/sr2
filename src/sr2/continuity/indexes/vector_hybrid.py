"""Vector/hybrid continuity index — TF-IDF similarity blended with structured signals (sr2-98 / AC17).

Combines a lightweight TF-IDF cosine similarity (computed in-memory over the
indexed corpus) with the same structured metadata signals the pure structured
index uses. No external embedding model or vector database required: the
vector component is a pure-Python sparse representation, making the index
fully deterministic, serializable, and free of external dependencies.

The hybrid weight between vector and structured components is fixed at
construction time (default 0.6 vector / 0.4 structured) and identical for
every query — no per-candidate tuning knobs.

Truth remains in the immutable ledger; the index is rebuildable.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
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


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


class _TfidfScorer:
    """Incremental TF-IDF over an in-memory corpus."""

    def __init__(self) -> None:
        self._doc_freqs: Counter[str] = Counter()
        self._doc_count: int = 0
        self._doc_tf: dict[str, Counter[str]] = {}
        self._doc_lengths: dict[str, float] = {}

    def add_document(self, doc_id: str, tokens: list[str]) -> None:
        tf = Counter(tokens)
        self._doc_tf[doc_id] = tf
        self._doc_lengths[doc_id] = math.sqrt(sum(v * v for v in tf.values()))
        for term in tf:
            self._doc_freqs[term] += 1
        self._doc_count += 1

    def clear(self) -> None:
        self._doc_freqs = Counter()
        self._doc_count = 0
        self._doc_tf = {}
        self._doc_lengths = {}

    def cosine_similarity(self, query_tokens: list[str], doc_id: str) -> float:
        """Cosine similarity between query TF-IDF vector and a document."""
        if not query_tokens or doc_id not in self._doc_tf:
            return 0.0

        query_tf = Counter(query_tokens)
        doc_tf = self._doc_tf[doc_id]

        # Compute dot product and norms simultaneously.
        dot = 0.0
        query_norm_sq = 0.0
        for term, q_count in query_tf.items():
            idf = self._idf(term)
            q_weight = q_count * idf
            query_norm_sq += q_weight * q_weight
            if term in doc_tf:
                d_weight = doc_tf[term] * idf
                dot += q_weight * d_weight

        doc_norm = self._doc_lengths.get(doc_id, 0.0)
        query_norm = math.sqrt(query_norm_sq)

        if doc_norm == 0.0 or query_norm == 0.0:
            return 0.0
        return dot / (query_norm * doc_norm)

    def _idf(self, term: str) -> float:
        df = self._doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + self._doc_count / df)


class VectorHybridIndex:
    """In-memory vector/hybrid index (TF-IDF + structured metadata).

    Scoring components:
    - ``vector``: TF-IDF cosine similarity between query and record content
    - ``lexical``: simple query-token coverage (same as structured)
    - ``salience``: the record's own salience field
    - ``confidence``: the record's confidence field
    - ``recency``: exponential decay from the most recent record
    - ``project``: 1.0 if the record shares a project with the query context

    The final score blends ``vector`` and the structured sub-score using the
    fixed ``hybrid_weight`` (default 0.6).
    """

    def __init__(self, hybrid_weight: float = 0.6) -> None:
        self._hybrid_weight = hybrid_weight
        self._records: list[ContinuityRecord] = []
        self._views: dict[str, ContinuityView] = {}
        self._tfidf = _TfidfScorer()
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
            tokens = _tokenize(record.content)
            self._tfidf.add_document(record.id, tokens)
            if self._latest_time is None or record.created_at > self._latest_time:
                self._latest_time = record.created_at

    async def rebuild(self, records: AsyncIterator[ContinuityRecord]) -> None:
        self._records = []
        self._views = {}
        self._tfidf.clear()
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
        query_set = set(query_tokens)
        scored: list[tuple[float, dict[str, float], ContinuityView]] = []

        for record in self._records:
            # Vector component: TF-IDF cosine similarity.
            vector = self._tfidf.cosine_similarity(query_tokens, record.id)

            # Structured components.
            content_tokens = _tokenize(record.content)
            lexical = self._lexical_overlap(query_set, content_tokens)
            salience = record.salience
            confidence = record.confidence
            recency = self._recency_score(record)
            project = self._project_score(record, context)

            # Structured sub-score (same weights as StructuredIndex).
            structured_score = (
                lexical * 0.50
                + salience * 0.15
                + confidence * 0.10
                + recency * 0.15
                + project * 0.10
            )

            # Hybrid blend.
            score = (
                self._hybrid_weight * vector
                + (1.0 - self._hybrid_weight) * structured_score
            )

            components = {
                "vector": vector,
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

    def _lexical_overlap(self, query_set: set[str], content_tokens: list[str]) -> float:
        """Fraction of query tokens present in content."""
        if not query_set:
            return 0.0
        content_set = set(content_tokens)
        matches = sum(1 for t in query_set if t in content_set)
        return matches / len(query_set)

    def _recency_score(self, record: ContinuityRecord) -> float:
        """Exponential decay: 1.0 for the newest, approaching 0 for old."""
        if self._latest_time is None:
            return 0.5
        if record.created_at >= self._latest_time:
            return 1.0
        age_seconds = (self._latest_time - record.created_at).total_seconds()
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
