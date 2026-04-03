"""Conflict resolution pipeline for memory conflicts."""

import logging
from dataclasses import dataclass
from typing import Literal

from sr2.memory.conflicts import Conflict
from sr2.memory.schema import Memory
from sr2.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of resolving a conflict."""

    action: Literal["keep_new", "keep_existing", "keep_both_tagged", "archive_old"]
    winner: Memory
    loser: Memory | None = None
    reason: str = ""


class ConflictResolver:
    """Resolves memory conflicts based on configured strategies."""

    def __init__(
        self,
        store: MemoryStore,
        strategies: dict[str, str] | None = None,
    ):
        """Args:
        store: Memory store for archival operations.
        strategies: mapping of memory_type -> resolution strategy.
            Default: identity=latest_wins_archive, semi_stable=latest_wins_archive,
                     dynamic=latest_wins_discard
        """
        self._store = store
        self._strategies = strategies or {
            "identity": "latest_wins_archive",
            "semi_stable": "latest_wins_archive",
            "dynamic": "latest_wins_discard",
        }

    async def resolve(self, conflict: Conflict) -> ResolutionResult:
        """Resolve a single conflict.

        Resolution pipeline (in order):
        1. Get strategy for the new memory's type
        2. If "latest_wins_archive": new wins, archive old
        3. If "latest_wins_discard": new wins, delete old
        4. If "keep_both": tag both with conflicts_with
        """
        strategy = self._strategies.get(
            conflict.new_memory.memory_type,
            self._strategies.get("default", "latest_wins_archive"),
        )

        if strategy == "latest_wins_archive":
            return await self._latest_wins(conflict, archive=True)
        elif strategy == "latest_wins_discard":
            return await self._latest_wins(conflict, archive=False)
        elif strategy == "keep_both":
            return await self._keep_both(conflict)
        else:
            logger.warning(
                "Unknown conflict resolution strategy %r for memory_type %r, defaulting to latest_wins_archive",
                strategy, conflict.new_memory.memory_type,
            )
            return await self._latest_wins(conflict, archive=True)

    async def _latest_wins(self, conflict: Conflict, archive: bool) -> ResolutionResult:
        """New memory wins. Archive or delete the old one."""
        if archive:
            await self._store.archive(conflict.existing_memory.id)
            action: Literal["keep_new", "archive_old"] = "archive_old"
        else:
            await self._store.delete(conflict.existing_memory.id)
            action = "keep_new"

        return ResolutionResult(
            action=action,
            winner=conflict.new_memory,
            loser=conflict.existing_memory,
            reason=f"Strategy: {'archive' if archive else 'discard'} old. "
            f"New memory type: {conflict.new_memory.memory_type}",
        )

    async def _keep_both(self, conflict: Conflict) -> ResolutionResult:
        """Keep both memories, tag them as conflicting."""
        conflict.new_memory.conflicts_with = conflict.existing_memory.id
        conflict.existing_memory.conflicts_with = conflict.new_memory.id
        await self._store.save(conflict.new_memory)
        await self._store.save(conflict.existing_memory)

        return ResolutionResult(
            action="keep_both_tagged",
            winner=conflict.new_memory,
            loser=None,
            reason="Ambiguous conflict — both memories retained with conflict tags.",
        )

    async def resolve_all(self, conflicts: list[Conflict]) -> list[ResolutionResult]:
        """Resolve a list of conflicts. Returns results in same order."""
        return [await self.resolve(c) for c in conflicts]
