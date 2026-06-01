"""Frontmatter parsing for plan, task, and project-knowledge markdown files.

Extracts the YAML block between the opening `---` and the next `---` from a
markdown file and parses it into the appropriate frozen dataclass per kind.

Tolerance guarantees (FR5):
  - Files lacking a recognized `kind` are skipped (logged).
  - Files that fail to parse (bad YAML, missing fields) are skipped (logged).
  - Never crashes resolve — returns None on any failure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from sr2.pipeline.plan.models import (
    KnowledgeFrontmatter,
    PlanFileFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    RECOGNIZED_KINDS,
    TaskFrontmatter,
    TaskStatus,
    get_frontmatter_class,
)

logger = logging.getLogger(__name__)


def extract_raw_frontmatter(text: str) -> Optional[str]:
    """Extract the raw YAML string between the first `---` and next `---`.

    Returns None if no valid frontmatter block is found.
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return None

    # Find the closing delimiter. Search after the opening `---`.
    rest = stripped[3:]  # skip opening ---
    if not rest.lstrip().startswith("\n") and not rest.startswith("\n"):
        # Edge case: `---` at start but no newline (not valid frontmatter)
        # Some markdown tools treat `---` without a newline as a thematic break
        # rather than frontmatter. We still try to parse it if there's a second
        # `---` somewhere, but require at least the second delimiter.
        pass

    try:
        idx = rest.index("\n---")
    except ValueError:
        # No closing delimiter found
        return None

    # Extract content between delimiters.
    raw = rest[:idx].strip()
    return raw if raw else None


def parse_frontmatter(
    text: str,
    file_path: Optional[Path] = None,
) -> Optional[PlanFileFrontmatter]:
    """Parse frontmatter from a markdown string into a frozen dataclass.

    Args:
        text: Full markdown file content.
        file_path: Optional path for logging context.

    Returns:
        A frozen frontmatter dataclass for the recognized kind, or None if
        the file has no frontmatter, an unrecognized kind, or fails to parse.
    """
    label = f"{file_path}" if file_path else "inline text"

    raw = extract_raw_frontmatter(text)
    if raw is None:
        logger.debug("No frontmatter found in %s — skipping", label)
        return None

    # Parse YAML
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("YAML parse error in %s: %s — skipping", label, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Frontmatter in %s is not a mapping — skipping", label)
        return None

    # Check kind
    kind = data.get("kind")
    if kind is None:
        logger.warning("No 'kind' in frontmatter of %s — skipping", label)
        return None

    kind = str(kind).strip().lower()
    if kind not in RECOGNIZED_KINDS:
        logger.warning(
            "Unrecognized kind %r in frontmatter of %s — skipping", kind, label
        )
        return None

    # Build the appropriate dataclass
    cls = get_frontmatter_class(kind)
    if cls is None:
        # Shouldn't happen if kind is in RECOGNIZED_KINDS, but defensive.
        logger.warning("No dataclass for kind %r in %s — skipping", kind, label)
        return None

    try:
        return _build_frontmatter(cls, data, kind, label)
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("Frontmatter field error in %s: %s — skipping", label, exc)
        return None


def _build_frontmatter(
    cls: type[PlanFileFrontmatter],
    data: dict,
    kind: str,
    label: str,
) -> PlanFileFrontmatter:
    """Instantiate the correct frontmatter dataclass from parsed YAML data.

    Normalizes enum fields and provides defaults for optional fields.
    """
    if kind == "task":
        return _parse_task(data, label)
    elif kind == "plan":
        return _parse_plan(data, label)
    elif kind == "project-knowledge":
        return _parse_knowledge(data, label)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def _parse_task(data: dict, label: str) -> TaskFrontmatter:
    """Parse task frontmatter with validation."""
    plan = data.get("plan", "")
    if plan is None:
        plan = ""
    plan = str(plan).strip()

    order = data.get("order", 0)
    try:
        order = int(order)
    except (TypeError, ValueError):
        logger.warning("Invalid 'order' in %s — defaulting to 0", label)
        order = 0

    status_raw = data.get("status", "pending")
    if status_raw is None:
        status_raw = "pending"
    status_raw = str(status_raw).strip().lower()
    try:
        status = TaskStatus(status_raw)
    except ValueError:
        logger.warning(
            "Invalid 'status' %r in %s — defaulting to pending", status_raw, label
        )
        status = TaskStatus.PENDING

    verify = data.get("verify", "")
    if verify is None:
        verify = ""
    verify = str(verify)

    title = data.get("title", "")
    if title is None:
        title = ""
    title = str(title)

    return TaskFrontmatter(
        kind="task",
        plan=plan,
        order=order,
        status=status,
        verify=verify,
        title=title,
    )


def _parse_plan(data: dict, label: str) -> PlanFrontmatter:
    """Parse plan frontmatter with validation."""
    slug = data.get("slug", "")
    if slug is None:
        slug = ""
    slug = str(slug).strip()

    status_raw = data.get("status", "open")
    if status_raw is None:
        status_raw = "open"
    status_raw = str(status_raw).strip().lower()
    try:
        status = PlanStatus(status_raw)
    except ValueError:
        logger.warning(
            "Invalid 'status' %r in %s — defaulting to open", status_raw, label
        )
        status = PlanStatus.OPEN

    goal = data.get("goal", "")
    if goal is None:
        goal = ""
    goal = str(goal)

    return PlanFrontmatter(
        kind="plan",
        slug=slug,
        status=status,
        goal=goal,
    )


def _parse_knowledge(data: dict, label: str) -> KnowledgeFrontmatter:
    """Parse project-knowledge frontmatter."""
    project = data.get("project", "")
    if project is None:
        project = ""
    project = str(project).strip()

    return KnowledgeFrontmatter(
        kind="project-knowledge",
        project=project,
    )


def parse_file(
    path: Path,
) -> Optional[PlanFileFrontmatter]:
    """Read a file from disk and parse its frontmatter.

    Wraps parse_frontmatter with file I/O. Returns None on any error
    (file not found, permission denied, parse failure).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Cannot read %s: %s — skipping", path, exc)
        return None

    return parse_frontmatter(text, file_path=path)
