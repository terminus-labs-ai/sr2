"""Frozen dataclasses for parsed plan frontmatter.

Each kind (task, plan, project-knowledge) has its own dataclass with the
required fields from the spec. All are frozen for immutability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"


class PlanStatus(str, Enum):
    OPEN = "open"
    DONE = "done"


@dataclass(frozen=True)
class TaskFrontmatter:
    """Parsed frontmatter for a task file (NN-slug.md)."""

    kind: str = "task"
    plan: str = ""
    order: int = 0
    status: TaskStatus = TaskStatus.PENDING
    verify: str = ""
    title: str = ""


@dataclass(frozen=True)
class PlanFrontmatter:
    """Parsed frontmatter for a plan-shared file (_plan.md)."""

    kind: str = "plan"
    slug: str = ""
    status: PlanStatus = PlanStatus.OPEN
    goal: str = ""


@dataclass(frozen=True)
class KnowledgeFrontmatter:
    """Parsed frontmatter for a project-knowledge file."""

    kind: str = "project-knowledge"
    project: str = ""


# Union of all recognized frontmatter types.
PlanFileFrontmatter = TaskFrontmatter | PlanFrontmatter | KnowledgeFrontmatter

# Mapping from kind string to the corresponding dataclass.
_FRONTMATTER_KINDS: dict[str, type[PlanFileFrontmatter]] = {
    "task": TaskFrontmatter,
    "plan": PlanFrontmatter,
    "project-knowledge": KnowledgeFrontmatter,
}


def get_frontmatter_class(kind: str) -> type[PlanFileFrontmatter] | None:
    """Return the dataclass for a recognized kind, or None."""
    return _FRONTMATTER_KINDS.get(kind)


RECOGNIZED_KINDS: frozenset[str] = frozenset(_FRONTMATTER_KINDS.keys())
