"""Policy registry for degradation configuration.

DegradationPolicy holds per-provider degradation settings.
DegradationPolicyStore stores and retrieves them by provider name.
DegradationPolicyRegistry discovers strategy classes via entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sr2.plugins.registry import PluginRegistry


@runtime_checkable
class DegradationStrategyProtocol(Protocol):
    """Protocol that degradation strategy classes must satisfy.

    A strategy class must expose a ``build()`` classmethod that accepts
    a configuration dict and returns a strategy instance.
    """

    @classmethod
    def build(cls, config: dict) -> "DegradationStrategyProtocol": ...


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


class DegradationPolicyStore:
    """Stores DegradationPolicy config objects keyed by provider name.

    Single responsibility: runtime configuration object storage.
    Strategy-class discovery lives in :class:`DegradationPolicyRegistry`.
    """

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


class DegradationPolicyRegistry:
    """Discovers and validates degradation strategy classes via entry points.

    Single responsibility: plugin discovery for the ``sr2.degradation_policies``
    entry-point group via :class:`~sr2.plugins.registry.PluginRegistry`.

    For per-provider config storage use :class:`DegradationPolicyStore`.
    """

    _ENTRY_POINT_GROUP = "sr2.degradation_policies"

    def __init__(self) -> None:
        self._registry: PluginRegistry[DegradationStrategyProtocol] = PluginRegistry(
            self._ENTRY_POINT_GROUP,
            DegradationStrategyProtocol,
        )
        # Config-object store delegated to DegradationPolicyStore.
        # Kept here for backward compatibility — prefer DegradationPolicyStore directly.
        self._policies: dict[str, DegradationPolicy] = {}

    # ------------------------------------------------------------------
    # Backward-compat config-object API (delegates to internal dict)
    # ------------------------------------------------------------------

    def register(self, policy: DegradationPolicy) -> None:
        """Add or replace the policy for *policy.provider_name*."""
        self._policies[policy.provider_name] = policy

    def get(self, provider_name: str) -> DegradationPolicy | None:
        """Return the policy for *provider_name*, or None if not registered."""
        return self._policies.get(provider_name)

    def list_all(self) -> list[DegradationPolicy]:
        """Return all registered policies (order not guaranteed)."""
        return list(self._policies.values())

    # ------------------------------------------------------------------
    # Strategy-class API (entry-point discovery via PluginRegistry)
    # ------------------------------------------------------------------

    def get_strategy(self, name: str) -> type[DegradationStrategyProtocol]:
        """Return the strategy *class* registered under *name*.

        Raises
        ------
        PluginCollisionError
            If two distributions register the same strategy name.
        PluginNotFoundError
            If no entry point with *name* exists in the group.
        TypeError
            If the loaded class does not satisfy
            :class:`DegradationStrategyProtocol`.
        """
        return self._registry.get(name)

    def list_strategy_names(self) -> list[str]:
        """Return all non-colliding strategy names discovered from entry points."""
        return self._registry.names()
