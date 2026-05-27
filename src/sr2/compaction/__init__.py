"""sr2.compaction — rule-based and LLM-powered context compaction."""

from sr2.compaction.cost_gate import CostGate
from sr2.compaction.engine import CompactionEngine
from sr2.compaction.llm_strategy import LLMCompactionStrategy
from sr2.compaction.rules import (
    ReferenceRule,
    collapse,
    result_summary,
    schema_and_sample,
    supersede,
)

__all__ = [
    "CostGate",
    "CompactionEngine",
    "LLMCompactionStrategy",
    "ReferenceRule",
    "collapse",
    "result_summary",
    "schema_and_sample",
    "supersede",
]
