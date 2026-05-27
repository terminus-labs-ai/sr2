"""Policy registry for degradation configuration.

DegradationPolicy holds per-provider degradation settings.
DegradationPolicyRegistry stores and retrieves them by provider name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DegradationPolicy:
    """Per-provider degradation configuration.

    Attributes:
        provider_name:             Identifier for the content provider.
        fallback_mode:             One of 'none', 'static', or 'cached'.
        circuit_breaker_threshold: Consecutive failures before the breaker opens.
        priority:                  Shedding priority (lower = shed first).
    """

    provider_name: str
    fallback_mode: str
    circuit_breaker_threshold: int
    priority: int


class DegradationPolicyRegistry:
    """Stores DegradationPolicy objects keyed by provider name."""

    def __init__(self) -> None:
        self._policies: dict[str, DegradationPolicy] = {}

    def register(self, policy: DegradationPolicy) -> None:
        """Add or replace the policy for *policy.provider_name*."""
        self._policies[policy.provider_name] = policy

    def get(self, provider_name: str) -> DegradationPolicy | None:
        """Return the policy for *provider_name*, or None if not registered."""
        return self._policies.get(provider_name)

    def list_all(self) -> list[DegradationPolicy]:
        """Return all registered policies (order not guaranteed)."""
        return list(self._policies.values())
