"""Tracks prefix stability across pipeline invocations for KV-cache optimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class PrefixSnapshot:
    """Snapshot of the prefix state at a single invocation."""

    full_hash: str
    layer_hashes: dict[str, str] = field(default_factory=dict)
    prefix_tokens: int = 0
    tools_hash: str | None = None


@dataclass
class CacheReport:
    """Report comparing current prefix to previous invocation."""

    prefix_stable: bool
    changed_layers: list[str] = field(default_factory=list)
    expected_cached_tokens: int = 0
    actual_cached_tokens: int = 0
    first_invocation: bool = False
    tools_changed: bool = False

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
        tool_schemas: list[dict] | None = None,
    ) -> PrefixSnapshot:
        """Create and store a new prefix snapshot.

        Args:
            tool_schemas: Optional list of tool JSON schemas. When provided,
                their hash is included in the snapshot for cache tracking.
        """
        tools_hash = None
        if tool_schemas is not None:
            tools_hash = self._hash_tools(tool_schemas)
        snap = PrefixSnapshot(
            full_hash=full_hash,
            layer_hashes=layer_hashes,
            prefix_tokens=prefix_tokens,
            tools_hash=tools_hash,
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
                tools_changed=False,
            )

        # Identify changed layers
        changed: list[str] = []
        all_layers = set(self._previous.layer_hashes) | set(current.layer_hashes)
        for layer_name in all_layers:
            prev_hash = self._previous.layer_hashes.get(layer_name)
            curr_hash = current.layer_hashes.get(layer_name)
            if prev_hash != curr_hash:
                changed.append(layer_name)

        # Check tool schema changes
        tools_changed = (
            current.tools_hash is not None
            and self._previous.tools_hash is not None
            and current.tools_hash != self._previous.tools_hash
        )

        prefix_stable = current.full_hash == self._previous.full_hash and not tools_changed

        report = CacheReport(
            prefix_stable=prefix_stable,
            changed_layers=sorted(changed),
            expected_cached_tokens=self._previous.prefix_tokens,
            actual_cached_tokens=actual_cached_tokens,
            tools_changed=tools_changed,
        )

        self._previous = current
        return report

    def reset(self) -> None:
        """Clear tracked state (e.g. on session change)."""
        self._previous = None

    @staticmethod
    def _hash_tools(tool_schemas: list[dict]) -> str:
        """Deterministic hash of tool schemas for cache tracking."""
        canonical = json.dumps(tool_schemas, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def suggest_tool_ordering(
        tool_schemas: list[dict],
        static_tools: set[str] | None = None,
        dynamic_tools: set[str] | None = None,
    ) -> list[dict]:
        """Suggest optimal tool ordering for cache efficiency.

        Ordering: static tools (never change) first, then session-stable
        tools, then dynamic tools (change per invocation) last.
        """
        static = static_tools or set()
        dynamic = dynamic_tools or set()

        static_group: list[dict] = []
        stable_group: list[dict] = []
        dynamic_group: list[dict] = []

        for schema in tool_schemas:
            name = schema.get("name", schema.get("function", {}).get("name", ""))
            if name in static:
                static_group.append(schema)
            elif name in dynamic:
                dynamic_group.append(schema)
            else:
                stable_group.append(schema)

        return static_group + stable_group + dynamic_group
