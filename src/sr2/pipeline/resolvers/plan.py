"""PlanResolver: dynamic, per-turn plan + knowledge injection for SR2.

Implements FR6/7/11 from the auto-decomposition spec:
  - Directory-per-plan layout under configurable plans_root.
  - Configurable knowledge_root for project-knowledge files.
  - Layered L1/L2/L3 injection with clear delimiters.
  - Per-turn re-read (not frozen at init) so mid-run status changes are
    reflected on the next turn.

Registered as entry point ``sr2.resolvers:plan`` — opt-in via pipeline config.
"""

from __future__ import annotations

import glob as _glob
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.plan.frontmatter import parse_file
from sr2.pipeline.plan.models import (
    KnowledgeFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    TaskFrontmatter,
    TaskStatus,
)
from sr2.pipeline.token_counting import CHARS_PER_TOKEN, CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_SUBSCRIPTION = EventSubscription(
    event_name="turn_start", phase=EventPhase.STARTING
)


# ---------------------------------------------------------------------------
# Layer delimiter constants
# ---------------------------------------------------------------------------

_LAYER1_HEADER = "## Project Knowledge"
_LAYER2_HEADER = "## Active Plan"
_LAYER3_HEADER = "## Current Task"

_LAYER_SEPARATOR = "\n---\n"


# ---------------------------------------------------------------------------
# PlanResolver
# ---------------------------------------------------------------------------


class PlanResolver:
    """Dynamically resolves plan + knowledge content on every turn.

    Config fields (in ``ResolverConfig.config``)
    -------------------------------------------
    plans_root : str
        Root directory holding one sub-directory per plan.
        Default: ``~/.sr2/plans``
    knowledge_root : str
        Directory holding project-knowledge markdown files.
        Default: ``~/.sr2/knowledge/<project>`` (expanded at init).
    project : str
        Active project name; used to filter L1 knowledge files.
        **Required.**
    max_tokens : int | None
        Optional token budget for the combined injection.
    """

    name: str = "plan"

    def __init__(self, config: ResolverConfig) -> None:
        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

        # Required: project name
        self._project: str = config.config.get("project")
        if not self._project:
            raise ValueError(
                "PlanResolver requires config['project'] to be set."
            )

        # Optional: roots with defaults
        plans_root_raw: str = config.config.get(
            "plans_root", str(Path.home() / ".sr2" / "plans")
        )
        self._plans_root = Path(plans_root_raw).expanduser().resolve()

        knowledge_root_raw: str = config.config.get(
            "knowledge_root",
            str(Path.home() / ".sr2" / "knowledge" / self._project),
        )
        self._knowledge_root = Path(knowledge_root_raw).expanduser().resolve()

        self._max_tokens: int | None = config.config.get("max_tokens")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> "PlanResolver":
        return cls(config)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        """Resolve L1 + L2 + L3 content dynamically from disk.

        Re-reads plan and knowledge directories on every call so that
        mid-run status changes are reflected immediately.
        """
        self.execution_count += 1

        layers: list[tuple[str, str]] = []  # (layer_header, content)

        # L1: project knowledge
        l1_content = self._resolve_layer1()
        if l1_content:
            layers.append((_LAYER1_HEADER, l1_content))

        # Resolve the single open plan (or none)
        open_plan_dir, plan_frontmatter = self._find_open_plan()

        if open_plan_dir and plan_frontmatter:
            # L2: plan-shared (_plan.md) — always injected when plan is open,
            # even if body is empty (signals to the model that a plan exists).
            l2_content = self._resolve_layer2(open_plan_dir)
            layers.append((_LAYER2_HEADER, l2_content))

            # L3: current task (lowest-order pending)
            l3_content = self._resolve_layer3(open_plan_dir)
            if l3_content.strip():
                layers.append((_LAYER3_HEADER, l3_content))

        # Build the combined text with delimiters
        combined_parts: list[str] = []
        for header, content in layers:
            combined_parts.append(header)
            combined_parts.append(content)

        combined = _LAYER_SEPARATOR.join(combined_parts)

        # Token budget enforcement
        tokens = len(combined) // CHARS_PER_TOKEN
        if self._max_tokens is not None and tokens > self._max_tokens:
            combined = self._truncate_to_budget(combined, self._max_tokens)
            tokens = len(combined) // CHARS_PER_TOKEN

        return ResolvedContent(
            resolver_name=self.name,
            source_layer="plan",
            content=[TextBlock(text=combined)],
            token_count=tokens,
        )

    # ------------------------------------------------------------------
    # Layer resolution helpers
    # ------------------------------------------------------------------

    def _resolve_layer1(self) -> str:
        """L1: Load all project-knowledge files matching the active project.

        Globs ``*.md`` under ``knowledge_root``, parses frontmatter, filters
        by ``kind: project-knowledge`` with matching ``project`` field.
        Returns concatenated content of matching files (body only, frontmatter
        stripped).
        """
        if not self._knowledge_root.is_dir():
            return ""

        pattern = str(self._knowledge_root / "*.md")
        paths = sorted(_glob.glob(pattern, recursive=True))

        parts: list[str] = []
        for path_str in paths:
            path = Path(path_str)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Cannot read %s: %s — skipping", path, exc)
                continue

            fm = parse_file(path)
            if isinstance(fm, KnowledgeFrontmatter) and fm.project == self._project:
                body = self._strip_frontmatter(text)
                if body.strip():
                    parts.append(body)

        return "\n\n".join(parts)

    def _find_open_plan(
        self,
    ) -> tuple[Path | None, PlanFrontmatter | None]:
        """Find the single open plan directory.

        Globs immediate sub-directories under ``plans_root``. Parses each
        ``_plan.md`` looking for ``status: open``.

        Returns:
            (plan_dir, plan_frontmatter) or (None, None) if no open plans,
            or raises PlanResolverError if multiple open plans found.
        """
        if not self._plans_root.is_dir():
            return None, None

        plan_dirs: list[Path] = sorted(
            p for p in self._plans_root.iterdir() if p.is_dir()
        )

        open_plans: list[tuple[Path, PlanFrontmatter]] = []

        for plan_dir in plan_dirs:
            plan_file = plan_dir / "_plan.md"
            if not plan_file.is_file():
                continue

            fm = parse_file(plan_file)
            if isinstance(fm, PlanFrontmatter) and fm.status == PlanStatus.OPEN:
                open_plans.append((plan_dir, fm))

        if len(open_plans) > 1:
            slugs = [str(p[0].name) for p in open_plans]
            raise PlanResolverError(
                f"Multiple open plans detected: {', '.join(slugs)}. "
                "PlanResolver v1 supports at most one open plan. "
                "Close or complete the plans you don't need."
            )

        if not open_plans:
            return None, None

        return open_plans[0]

    def _resolve_layer2(self, plan_dir: Path) -> str:
        """L2: Load the _plan.md body (goal + constraints contract)."""
        plan_file = plan_dir / "_plan.md"
        if not plan_file.is_file():
            return ""

        text = plan_file.read_text(encoding="utf-8")
        return self._strip_frontmatter(text)

    def _resolve_layer3(self, plan_dir: Path) -> str:
        """L3: Load the lowest-order pending task file body.

        Globs ``NN-*.md`` files in the plan directory, parses frontmatter,
        filters for ``kind: task`` with ``status: pending``, sorts by ``order``,
        returns the body of the first match.
        """
        pattern = str(plan_dir / "*.md")
        paths = sorted(_glob.glob(pattern, recursive=True))

        pending_tasks: list[tuple[int, str]] = []  # (order, body_text)

        for path_str in paths:
            path = Path(path_str)
            # Skip _plan.md
            if path.name == "_plan.md":
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Cannot read %s: %s — skipping", path, exc)
                continue

            fm = parse_file(path)
            if isinstance(fm, TaskFrontmatter) and fm.status == TaskStatus.PENDING:
                body = self._strip_frontmatter(text)
                pending_tasks.append((fm.order, body))

        if not pending_tasks:
            return ""

        pending_tasks.sort(key=lambda x: x[0])
        return pending_tasks[0][1]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """Strip the YAML frontmatter block (--- ... ---) from text.

        Returns the body after the closing ``---``. If no frontmatter is
        found, returns the original text.
        """
        stripped = text.strip()
        if not stripped.startswith("---"):
            return text

        rest = stripped[3:]  # skip opening ---
        try:
            idx = rest.index("\n---")
            return rest[idx + 4:]  # skip "\n---" + leading newline
        except ValueError:
            return text

    @staticmethod
    def _truncate_to_budget(text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens (chars // 4 approximation).

        Adds a truncation notice at the end. If the budget is too small even
        for the notice, the notice alone is returned.
        """
        max_chars = max_tokens * CHARS_PER_TOKEN
        notice = "\n\n⚠️ Content truncated — token budget exceeded."

        if len(text) <= max_chars:
            return text

        # Ensure cutoff leaves at least room for the notice
        cutoff = max(0, max_chars - len(notice))
        return text[:cutoff] + notice


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PlanResolverError(Exception):
    """Raised when PlanResolver encounters a non-tolerable error.

    Used for: multiple open plans (v1 limitation).
    """
