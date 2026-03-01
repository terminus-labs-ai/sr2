from typing import Protocol
from dataclasses import dataclass


@dataclass
class PipelineState:
    """Snapshot of pipeline state for cache decisions."""
    turn_number: int = 0
    current_intent: str | None = None
    previous_intent: str | None = None
    state_hash: str | None = None
    previous_state_hash: str | None = None


class CachePolicy(Protocol):
    """Protocol for cache policies."""
    def should_recompute(
        self,
        layer_name: str,
        current_state: PipelineState,
        previous_state: PipelineState | None,
    ) -> bool:
        ...


class CachePolicyRegistry:
    """Registry mapping policy names to policy instances."""

    def __init__(self) -> None:
        self._policies: dict[str, CachePolicy] = {}

    def register(self, name: str, policy: CachePolicy) -> None:
        self._policies[name] = policy

    def get(self, name: str) -> CachePolicy:
        if name not in self._policies:
            raise KeyError(f"No cache policy registered: {name}")
        return self._policies[name]

    def has(self, name: str) -> bool:
        return name in self._policies

    @property
    def registered_policies(self) -> list[str]:
        return list(self._policies.keys())
