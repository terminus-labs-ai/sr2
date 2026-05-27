"""SR2 public facade.

Thin re-export of sr2.orchestrator.SR2. All logic lives in the orchestrator;
this module exists as the stable public import path.
"""

from sr2.orchestrator import SR2

__all__ = ["SR2"]
