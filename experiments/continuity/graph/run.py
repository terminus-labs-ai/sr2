"""Graph retrieval experiment — run benchmark scenarios and publish raw results (sr2-99 / AC18).

Exercises the graph index against the shared benchmark fixture scenarios
(same records, queries, and budgets as structured and vector candidates).
Publishes raw per-scenario results with lineage and ingestion diagnostics.

Usage:
    python -m experiments.continuity.graph.run
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from sr2.continuity.indexes.graph import GraphIndex
from sr2.continuity.models import ContinuityEventKind, ContinuityKind, ContinuityRecord
from sr2.continuity.policy import RetrievalContext
from sr2.pipeline.provenance import ContextAuthority, ContextScope


def _make_fixture_records() -> list[ContinuityRecord]:
    """Shared benchmark fixture: records covering multiple scenarios."""
    now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        ContinuityRecord(
            id="rec-bench-0001",
            subject_id="owner-bench",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.DECISION,
            content="Decision: adopt pytest as the preferred testing framework for the sr2 project, with asyncio support enabled for all asynchronous tests.",
            scope=ContextScope(owner_id="owner-bench", agent_id="agent-coding", projects=("sr2",), topics=("testing",)),
            authority=ContextAuthority.USER,
            provenance_ids=("entry-bench-001",),
            confidence=0.9,
            salience=0.8,
            created_at=now,
        ),
        ContinuityRecord(
            id="rec-bench-0002",
            subject_id="owner-bench",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.DECISION,
            content="The pipeline engine depends on an EventBus for pub/sub. LiteLLM handles model routing.",
            scope=ContextScope(owner_id="owner-bench", agent_id="agent-coding", projects=("sr2",), topics=("architecture",)),
            authority=ContextAuthority.VERIFIED,
            provenance_ids=("entry-bench-002", "entry-bench-003"),
            confidence=0.85,
            salience=0.7,
            created_at=now,
        ),
        ContinuityRecord(
            id="rec-bench-0003",
            subject_id="owner-bench",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.PROJECT_FACT,
            content="SQLiteProvenanceStore uses aiosqlite for durable append-only provenance entries.",
            scope=ContextScope(owner_id="owner-bench", agent_id="agent-coding", projects=("sr2",), topics=("storage",)),
            authority=ContextAuthority.VERIFIED,
            provenance_ids=("entry-bench-004",),
            confidence=0.9,
            salience=0.6,
            created_at=now,
        ),
        ContinuityRecord(
            id="rec-bench-0004",
            subject_id="owner-bench",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.PREFERENCE,
            content="Prefers BLUF communication style and explicit over implicit patterns.",
            scope=ContextScope(owner_id="owner-bench", agent_id="agent-coding", projects=(), topics=("communication",)),
            authority=ContextAuthority.USER,
            provenance_ids=("entry-bench-005",),
            confidence=0.95,
            salience=0.5,
            created_at=now,
        ),
        ContinuityRecord(
            id="rec-bench-0005",
            subject_id="owner-bench",
            event_kind=ContinuityEventKind.FORMED,
            kind=ContinuityKind.NEXT_ACTION,
            content="Reminder: reorder the pantry shelves and rotate the canned goods before the quarterly inventory count.",
            scope=ContextScope(owner_id="owner-bench", agent_id="agent-coding", projects=(), topics=("personal",)),
            authority=ContextAuthority.AGENT_TENTATIVE,
            provenance_ids=(),
            confidence=0.4,
            salience=0.2,
            created_at=now,
        ),
    ]


SCENARIOS = [
    {
        "name": "testing_framework_recall",
        "query": "preferred testing framework for the project",
        "expected_top": "rec-bench-0001",
        "projects": ("sr2",),
    },
    {
        "name": "architecture_entity_query",
        "query": "EventBus pub/sub LiteLLM model routing",
        "expected_top": "rec-bench-0002",
        "projects": ("sr2",),
    },
    {
        "name": "storage_provenance_query",
        "query": "aiosqlite provenance store durable",
        "expected_top": "rec-bench-0003",
        "projects": ("sr2",),
    },
    {
        "name": "cross_project_personal",
        "query": "communication style preference BLUF",
        "expected_top": "rec-bench-0004",
        "projects": ("other-project",),
    },
    {
        "name": "low_relevance_noise",
        "query": "quantum computing applications in biology",
        "expected_top": None,
        "projects": ("sr2",),
    },
]


async def main() -> None:
    records = _make_fixture_records()
    index = GraphIndex()
    await index.append(records)

    results = []
    for scenario in SCENARIOS:
        context = RetrievalContext(
            owner_id="owner-bench",
            agent_id="agent-coding",
            agent_tags=("coding",),
            projects=scenario["projects"],
            frame_id=None,
            query=scenario["query"],
            token_budget=4096,
        )
        candidates = await index.search(context, limit=5)
        top_id = candidates[0].record.current_record.id if candidates else None
        results.append({
            "scenario": scenario["name"],
            "query": scenario["query"],
            "expected_top": scenario["expected_top"],
            "actual_top": top_id,
            "hit": top_id == scenario["expected_top"] if scenario["expected_top"] else True,
            "candidate_count": len(candidates),
            "scores": [c.score for c in candidates[:3]],
        })

    diag = index.diagnostics()
    report = {
        "experiment": "graph_retrieval_benchmark",
        "bead": "sr2-99",
        "ac": "AC18",
        "scenarios_run": len(SCENARIOS),
        "scenarios_hit": sum(1 for r in results if r["hit"]),
        "results": results,
        "diagnostics": {
            "records_ingested": diag.records_ingested,
            "entities_extracted": diag.entities_extracted,
            "relations_extracted": diag.relations_extracted,
            "ingestion_loss": diag.ingestion_loss,
            "lineage_coverage": diag.lineage_coverage,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
