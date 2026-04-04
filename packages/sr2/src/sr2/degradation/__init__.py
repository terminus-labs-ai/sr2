"""Degradation subsystem — circuit breaker and degradation ladder."""

from sr2.degradation.ladder import DegradationLadder, DegradationLevel, DEGRADATION_ORDER
from sr2.degradation.circuit_breaker import CircuitBreaker
from sr2.degradation.registry import register_policy

# Register built-in degradation policy (the ladder itself).
register_policy("ladder", DegradationLadder)

__all__ = ["DegradationLadder", "DegradationLevel", "DEGRADATION_ORDER", "CircuitBreaker"]
