"""Tracks prefix stability across pipeline invocations for KV-cache optimization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefixSnapshot:
    """Snapshot of the prefix state at a single invocation."""

    full_hash: str
    layer_hashes: dict[str, str] = field(default_factory=dict)
    prefix_tokens: int = 0


@dataclass
class CacheReport:
    """Report comparing current prefix to previous invocation."""

    prefix_stable: bool
    changed_layers: list[str] = field(default_factory=list)
    expected_cached_tokens: int = 0
    actual_cached_tokens: int = 0
    first_invocation: bool = False

    @property
    def cache_efficiency(self) -> float:
        """Ratio of actual cached tokens to expected cached tokens."""
        if self.expected_cached_tokens == 0:
            return 0.0
        return self.actual_cached_tokens / self.expected_cached_tokens


class PrefixTracker:
    """Tracks prefix stability across invocations.

    Compares the current prefix snapshot with the previous one to determine
    whether the KV-cache prefix was invalidated and which layers changed.
    """

    def __init__(self) -> None:
        self._previous: PrefixSnapshot | None = None

    def snapshot(
        self,
        layer_hashes: dict[str, str],
        full_hash: str,
        prefix_tokens: int,
    ) -> PrefixSnapshot:
        """Create and store a new prefix snapshot."""
        snap = PrefixSnapshot(
            full_hash=full_hash,
            layer_hashes=layer_hashes,
            prefix_tokens=prefix_tokens,
        )
        return snap

    def compare(
        self,
        current: PrefixSnapshot,
        actual_cached_tokens: int,
    ) -> CacheReport:
        """Compare current snapshot with previous, then advance state.

        On the first call (no previous snapshot), reports prefix_stable=True
        since there's nothing to compare against.
        """
        if self._previous is None:
            self._previous = current
            return CacheReport(
                prefix_stable=True,
                changed_layers=[],
                expected_cached_tokens=current.prefix_tokens,
                actual_cached_tokens=actual_cached_tokens,
                first_invocation=True,
            )

        # Identify changed layers
        changed: list[str] = []
        all_layers = set(self._previous.layer_hashes) | set(current.layer_hashes)
        for layer_name in all_layers:
            prev_hash = self._previous.layer_hashes.get(layer_name)
            curr_hash = current.layer_hashes.get(layer_name)
            if prev_hash != curr_hash:
                changed.append(layer_name)

        prefix_stable = current.full_hash == self._previous.full_hash

        report = CacheReport(
            prefix_stable=prefix_stable,
            changed_layers=sorted(changed),
            expected_cached_tokens=self._previous.prefix_tokens,
            actual_cached_tokens=actual_cached_tokens,
        )

        self._previous = current
        return report

    def reset(self) -> None:
        """Clear tracked state (e.g. on session change)."""
        self._previous = None
