"""sr2.degradation — graceful degradation subsystem.

Public API:
  ladder          — DegradationLadder, DegradationLevel
  shedding        — shed()
  circuit_breaker — CircuitBreaker, CircuitState
  fallback        — FallbackProvider
  registry        — DegradationPolicy, DegradationPolicyRegistry
"""

from sr2.degradation.circuit_breaker import CircuitBreaker, CircuitState
from sr2.degradation.fallback import FallbackProvider
from sr2.degradation.ladder import DegradationLadder, DegradationLevel
from sr2.degradation.registry import DegradationPolicy, DegradationPolicyRegistry
from sr2.degradation.shedding import shed

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "FallbackProvider",
    "DegradationLadder",
    "DegradationLevel",
    "DegradationPolicy",
    "DegradationPolicyRegistry",
    "shed",
]
