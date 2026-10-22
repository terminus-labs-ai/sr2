"""Graph ingestion experiment — extract entities/relations and publish diagnostics (sr2-99 / AC18).

Demonstrates that the graph index ingests continuity records, builds an
entity/relation projection, and publishes ingestion diagnostics showing
extraction counts and loss. The graph is a rebuildable projection; truth
remains in the immutable ledger.

Usage:
    python -m experiments.continuity.graph.ingest
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from sr2.continuity.indexes.graph import GraphIndex
from sr2.continuity.models import ContinuityEventKind, ContinuityKind, ContinuityRecord
from sr2.pipeline.provenance import ContextAuthority, ContextScope


SAMPLE_RECORDS = [
    ContinuityRecord(
        id="rec-ingest-0001",
        subject_id="owner-exp",
        event_kind=ContinuityEventKind.FORMED,
        kind=ContinuityKind.DECISION,
        content="Decision: adopt pytest as the preferred testing framework for the sr2 project, with asyncio support.",
        scope=ContextScope(owner_id="owner-exp", agent_id="agent-coding", projects=("sr2",)),
        authority=ContextAuthority.USER,
        provenance_ids=("entry-001", "entry-002"),
        confidence=0.95,
        salience=0.7,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    ),
    ContinuityRecord(
        id="rec-ingest-0002",
        subject_id="owner-exp",
        event_kind=ContinuityEventKind.FORMED,
        kind=ContinuityKind.DECISION,
        content="The orchestrator uses LiteLLM for model routing and Pydantic for configuration validation.",
        scope=ContextScope(owner_id="owner-exp", agent_id="agent-coding", projects=("sr2",)),
        authority=ContextAuthority.USER,
        provenance_ids=("entry-003",),
        confidence=0.9,
        salience=0.6,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    ),
    ContinuityRecord(
        id="rec-ingest-0003",
        subject_id="owner-exp",
        event_kind=ContinuityEventKind.FORMED,
        kind=ContinuityKind.PROJECT_FACT,
        content="SQLite stores the provenance ledger. PostgresMemoryStore provides persistent cross-process memory.",
        scope=ContextScope(owner_id="owner-exp", agent_id="agent-coding", projects=("sr2",)),
        authority=ContextAuthority.VERIFIED,
        provenance_ids=("entry-004", "entry-005"),
        confidence=0.85,
        salience=0.5,
        created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
    ),
    ContinuityRecord(
        id="rec-ingest-0004",
        subject_id="owner-exp",
        event_kind=ContinuityEventKind.FORMED,
        kind=ContinuityKind.EPISODE,
        content="...",
        scope=ContextScope(owner_id="owner-exp", agent_id="agent-coding", projects=("sr2",)),
        authority=ContextAuthority.AGENT_TENTATIVE,
        provenance_ids=(),
        confidence=0.3,
        salience=0.1,
        created_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
    ),
]


async def main() -> None:
    index = GraphIndex()
    await index.append(SAMPLE_RECORDS)

    diag = index.diagnostics()
    report = {
        "experiment": "graph_ingestion",
        "bead": "sr2-99",
        "ac": "AC18",
        "records_ingested": diag.records_ingested,
        "entities_extracted": diag.entities_extracted,
        "relations_extracted": diag.relations_extracted,
        "ingestion_loss": diag.ingestion_loss,
        "lineage_coverage": diag.lineage_coverage,
        "notes": (
            "Record rec-ingest-0004 (content='...') yields zero entities and "
            "zero relations — counted as ingestion loss. The graph projection "
            "is rebuildable from the ledger and never a source of truth."
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
