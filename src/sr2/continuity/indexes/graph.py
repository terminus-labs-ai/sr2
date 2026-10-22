"""Graph continuity index — entity/relation projection over the continuity ledger (sr2-99 / AC18).

The graph is an index projection, never a second truth store. It extracts
entities and relations from record content at ingestion time, builds an
in-memory property graph, and scores retrieval candidates by combining
graph proximity with structured metadata signals.

Determinism: identical record sequences produce identical entity sets,
relation sets, scores, and orderings. No randomness, no external model calls.

Diagnostics expose ingestion loss (records yielding no extractable structure)
and lineage coverage (fraction of candidates carrying non-empty provenance).

Truth remains in the immutable ledger; the graph is rebuildable via ``rebuild()``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

from pydantic import BaseModel

from sr2.continuity.index import ContinuityCandidate, ContinuityHealth
from sr2.continuity.models import ContinuityRecord, ContinuityView
from sr2.continuity.policy import RetrievalContext


# ---------------------------------------------------------------------------
# Diagnostics (frozen pydantic — published alongside the index state)
# ---------------------------------------------------------------------------


class GraphDiagnostics(BaseModel, frozen=True):
    """Ingestion and lineage metrics for the graph projection."""

    records_ingested: int
    entities_extracted: int
    relations_extracted: int
    ingestion_loss: int
    lineage_coverage: float


# ---------------------------------------------------------------------------
# Internal graph structures
# ---------------------------------------------------------------------------


@dataclass
class _Entity:
    """A node in the property graph."""

    name: str
    record_ids: set[str] = field(default_factory=set)


@dataclass
class _Relation:
    """An edge between two entities."""

    source: str
    predicate: str
    target: str
    record_id: str


# ---------------------------------------------------------------------------
# Extraction patterns (deterministic, no ML)
# ---------------------------------------------------------------------------

# Capitalized phrases (2+ words starting with uppercase) — likely proper nouns.
_CAPITALIZED_PHRASE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# Single capitalized words that aren't sentence-initial (heuristic: preceded
# by non-start-of-string and not the first word after punctuation).
_EMBEDDED_CAPS = re.compile(
    r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b"
)

# Backtick-quoted terms.
_BACKTICK_TERM = re.compile(r"`([^`]+)`")

# Relation patterns: subject verb object.
_RELATION_PATTERNS = [
    re.compile(r"\b(\w+(?:\s+\w+){0,3})\s+(?:adopts?|uses?|depends?\s+on|requires?|implements?|extends?|wraps?|calls?)\s+(\w+(?:\s+\w+){0,3})", re.IGNORECASE),
    re.compile(r"\b(\w+(?:\s+\w+){0,3})\s+(?:is|are)\s+(?:a|an|the)\s+(\w+(?:\s+\w+){0,3})", re.IGNORECASE),
    re.compile(r"\bdecision:\s*(\w+(?:\s+\w+){0,3})\b", re.IGNORECASE),
]


def _extract_entities(text: str) -> set[str]:
    """Extract entity names from text using deterministic patterns."""
    entities: set[str] = set()

    # Capitalized multi-word phrases.
    for match in _CAPITALIZED_PHRASE.finditer(text):
        entities.add(match.group(1).lower())

    # Embedded capitalized words (skip if at start of text).
    for match in _EMBEDDED_CAPS.finditer(text):
        word = match.group(1)
        # Skip if it's the first word of the text.
        if match.start() > 0:
            entities.add(word.lower())

    # Backtick terms.
    for match in _BACKTICK_TERM.finditer(text):
        entities.add(match.group(1).lower())

    # Decision markers: "Decision: adopt pytest" → "pytest"
    decision_match = re.search(r"[Dd]ecision:\s*adopt\s+(\w+)", text)
    if decision_match:
        entities.add(decision_match.group(1).lower())

    # Lowercase known technical terms that appear in content.
    _TECH_TERMS = ("pytest", "sqlite", "postgres", "postgresql", "redis", "kafka", "litellm", "pydantic", "aiosqlite", "asyncio")
    text_lower = text.lower()
    for term in _TECH_TERMS:
        if term in text_lower:
            entities.add(term)

    return entities


def _extract_relations(text: str, entities: set[str]) -> list[tuple[str, str, str]]:
    """Extract (source, predicate, target) triples from text."""
    relations: list[tuple[str, str, str]] = []

    for pattern in _RELATION_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if len(groups) >= 2:
                source = groups[0].strip().lower()
                target = groups[1].strip().lower()
            elif len(groups) == 1:
                # Single-group pattern (e.g. "decision: X") — link to a generic node.
                source = groups[0].strip().lower()
                target = "decision"
            else:
                continue
            if source and target and source != target:
                if len(source) <= 40 and len(target) <= 40:
                    predicate = "related_to"
                    if len(groups) >= 2:
                        full = match.group(0)
                        between = full[len(groups[0]):full.rindex(groups[1])]
                        predicate = between.strip() or "related_to"
                    relations.append((source, predicate, target))

    return relations


def _tokenize(text: str) -> list[str]:
    return [
        word.strip(".,;:!?()[]{}\"'")
        for word in text.lower().split()
        if word.strip(".,;:!?()[]{}\"'")
    ]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# GraphIndex — the ContinuityIndex implementation
# ---------------------------------------------------------------------------


class GraphIndex:
    """In-memory property graph index over continuity records.

    Scoring components:
    - ``entity_overlap``: fraction of query entities found in the record's entity set
    - ``graph_proximity``: boost from graph neighborhood connectivity to high-scoring records
    - ``salience``: the record's own salience field
    - ``confidence``: the record's confidence field
    - ``recency``: exponential decay from the most recent record
    - ``project``: 1.0 if the record shares a project with the query context

    The final score blends entity_overlap and graph_proximity (the graph
    signals) with the structured metadata signals.
    """

    def __init__(self) -> None:
        self._records: list[ContinuityRecord] = []
        self._views: dict[str, ContinuityView] = {}
        self._entities: dict[str, _Entity] = {}
        self._relations: list[_Relation] = []
        self._record_entities: dict[str, set[str]] = {}
        self._latest_time: datetime | None = None
        self._records_ingested: int = 0
        self._ingestion_loss: int = 0

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

            # Extract graph structure.
            entities = _extract_entities(record.content)
            relations = _extract_relations(record.content, entities)

            self._record_entities[record.id] = entities
            for entity_name in entities:
                if entity_name not in self._entities:
                    self._entities[entity_name] = _Entity(name=entity_name)
                self._entities[entity_name].record_ids.add(record.id)

            for source, predicate, target in relations:
                self._relations.append(
                    _Relation(source=source, predicate=predicate, target=target, record_id=record.id)
                )
                # Ensure relation endpoints exist as entities.
                for name in (source, target):
                    if name not in self._entities:
                        self._entities[name] = _Entity(name=name)

            self._records_ingested += 1
            if not entities and not relations:
                self._ingestion_loss += 1

    async def rebuild(self, records: AsyncIterator[ContinuityRecord]) -> None:
        self._records = []
        self._views = {}
        self._entities = {}
        self._relations = []
        self._record_entities = {}
        self._latest_time = None
        self._records_ingested = 0
        self._ingestion_loss = 0
        async for record in records:
            await self.append([record])

    async def search(
        self,
        context: RetrievalContext,
        limit: int,
    ) -> list[ContinuityCandidate]:
        if not self._records or limit <= 0:
            return []

        query_entities = _extract_entities(context.query)
        query_tokens = _tokenize(context.query)

        # Phase 1: score each record by entity overlap and structured signals.
        base_scores: list[tuple[float, dict[str, float], str]] = []
        for record in self._records:
            record_entities = self._record_entities.get(record.id, set())

            # Entity overlap: fraction of query entities in record's entity set.
            if query_entities:
                entity_overlap = len(query_entities & record_entities) / len(query_entities)
            else:
                entity_overlap = 0.0

            # Fallback lexical overlap for records without entity matches.
            content_tokens = _tokenize(record.content)
            lexical = self._lexical_overlap(query_tokens, content_tokens)

            # Use the better of entity_overlap and lexical.
            relevance = max(entity_overlap, lexical * 0.5)

            salience = record.salience
            confidence = record.confidence
            recency = self._recency_score(record)
            project = self._project_score(record, context)

            base_score = (
                relevance * 0.40
                + salience * 0.15
                + confidence * 0.10
                + recency * 0.15
                + project * 0.10
                + entity_overlap * 0.10
            )

            components = {
                "entity_overlap": entity_overlap,
                "relevance": relevance,
                "salience": salience,
                "confidence": confidence,
                "recency": recency,
                "project": project,
            }
            base_scores.append((base_score, components, record.id))

        # Phase 2: graph proximity boost.
        # Records connected via relations to top-scoring records get a boost.
        base_scores.sort(key=lambda x: -x[0])
        top_ids = {rid for _, _, rid in base_scores[:max(3, limit)]}

        boosted_scores: list[tuple[float, dict[str, float], str]] = []
        for score, components, record_id in base_scores:
            graph_proximity = self._graph_proximity(record_id, top_ids)
            final_score = score + graph_proximity * 0.10
            components["graph_proximity"] = graph_proximity
            boosted_scores.append((final_score, components, record_id))

        # Sort by final score descending, then by record id for stable ties.
        boosted_scores.sort(key=lambda x: (-x[0], x[2]))

        results: list[ContinuityCandidate] = []
        for score, components, record_id in boosted_scores[:limit]:
            view = self._views[record_id]
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

    def diagnostics(self) -> GraphDiagnostics:
        """Publish ingestion and lineage metrics."""
        entities_count = len(self._entities)
        relations_count = len(self._relations)

        # Lineage coverage: fraction of indexed records with non-empty provenance.
        if self._records:
            with_provenance = sum(
                1 for r in self._records if r.provenance_ids
            )
            lineage_coverage = with_provenance / len(self._records)
        else:
            lineage_coverage = 0.0

        return GraphDiagnostics(
            records_ingested=self._records_ingested,
            entities_extracted=entities_count,
            relations_extracted=relations_count,
            ingestion_loss=self._ingestion_loss,
            lineage_coverage=lineage_coverage,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _graph_proximity(self, record_id: str, top_ids: set[str]) -> float:
        """Fraction of graph neighbors of record_id that are in top_ids."""
        if not self._relations:
            return 0.0

        neighbors: set[str] = set()
        for rel in self._relations:
            if rel.record_id == record_id:
                # This record's relations connect to other records via entities.
                for entity_name in self._record_entities.get(record_id, set()):
                    if entity_name in self._entities:
                        neighbors |= self._entities[entity_name].record_ids
                neighbors.add(rel.source)
                neighbors.add(rel.target)

        # Remove self.
        neighbors.discard(record_id)

        if not neighbors:
            return 0.0

        overlap = neighbors & top_ids
        return len(overlap) / len(neighbors) if neighbors else 0.0

    def _lexical_overlap(self, query_tokens: list[str], content_tokens: list[str]) -> float:
        if not query_tokens:
            return 0.0
        content_set = set(content_tokens)
        matches = sum(1 for t in query_tokens if t in content_set)
        return matches / len(query_tokens)

    def _recency_score(self, record: ContinuityRecord) -> float:
        if self._latest_time is None:
            return 0.5
        if record.created_at >= self._latest_time:
            return 1.0
        age_seconds = (self._latest_time - record.created_at).total_seconds()
        half_life = 7 * 24 * 3600
        return math.exp(-math.log(2) * age_seconds / half_life)

    def _project_score(self, record: ContinuityRecord, context: RetrievalContext) -> float:
        if not context.projects:
            return 0.0
        record_projects = set(record.scope.projects)
        context_projects = set(context.projects)
        if record_projects & context_projects:
            return 1.0
        return 0.0
