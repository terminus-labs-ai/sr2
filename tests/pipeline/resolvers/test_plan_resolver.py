"""Tests for PlanResolver.

Covers all acceptance criteria from the auto-decomposition spec (FR6/7/11):
  - Layered L1/L2/L3 injection with clear delimiters
  - Per-turn dynamic discovery (re-read from disk)
  - Open-plan resolution & ambiguity
  - Frontmatter tolerance (malformed files skipped)
  - Token budget enforcement
  - Read-only resolver (never writes to plan files)
  - Planning guide NOT auto-injected
  - Task status advancement (pending → done → next task injected)
  - Zero open plans → only L1 injected
  - Multiple open plans → clear error raised
  - Resolver protocol compliance
  - Mid-run plan discovery
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.plan.models import TaskStatus
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.resolvers.plan import (
    PlanResolver,
    PlanResolverError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT = "test-project"


def make_config(
    plans_root: str | None = None,
    knowledge_root: str | None = None,
    project: str = PROJECT,
    max_tokens: int | None = None,
) -> ResolverConfig:
    cfg: dict = {"project": project}
    if plans_root is not None:
        cfg["plans_root"] = plans_root
    if knowledge_root is not None:
        cfg["knowledge_root"] = knowledge_root
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    return ResolverConfig(type="plan", config=cfg)


def make_turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def write_plan_file(
    plan_dir: Path,
    slug: str = "test-plan",
    status: str = "open",
    goal: str = "Test goal",
    body: str = "Understanding & constraints contract.",
) -> Path:
    plan_file = plan_dir / "_plan.md"
    plan_file.write_text(
        f"""---
kind: plan
slug: {slug}
status: {status}
goal: "{goal}"
---

{body}
"""
    )
    return plan_file


def write_task_file(
    plan_dir: Path,
    order: int,
    slug: str,
    status: str = "pending",
    verify: str = "echo verify",
    title: str = "Test task",
    body: str = "Task body content.",
) -> Path:
    task_file = plan_dir / f"{order:02d}-{slug}.md"
    task_file.write_text(
        f"""---
kind: task
plan: test-plan
order: {order}
status: {status}
verify: "{verify}"
title: "{title}"
---

{body}
"""
    )
    return task_file


def write_knowledge_file(
    knowledge_dir: Path,
    filename: str,
    project: str = PROJECT,
    body: str = "Knowledge content.",
) -> Path:
    kfile = knowledge_dir / filename
    kfile.write_text(
        f"""---
kind: project-knowledge
project: {project}
---

{body}
"""
    )
    return kfile


# ---------------------------------------------------------------------------
# 1. Resolver protocol compliance
# ---------------------------------------------------------------------------


class TestPlanResolverProtocol:
    def test_has_name_attribute(self, tmp_path):
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        assert resolver.name == "plan"

    def test_has_subscriptions(self, tmp_path):
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        assert hasattr(resolver, "subscriptions")
        assert isinstance(resolver.subscriptions, list)

    def test_has_max_executions(self, tmp_path):
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_execution_count_starts_at_zero(self, tmp_path):
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        assert resolver.execution_count == 0

    def test_satisfies_resolver_protocol(self, tmp_path):
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        assert isinstance(resolver, Resolver)

    def test_default_subscription_is_turn_start(self, tmp_path):
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        names = [s.event_name for s in resolver.subscriptions]
        assert "turn_start" in names

    def test_build_classmethod(self, tmp_path):
        config = make_config(str(tmp_path), str(tmp_path))
        resolver = PlanResolver.build(config, Dependencies())
        assert isinstance(resolver, PlanResolver)

    def test_build_satisfies_resolver_protocol(self, tmp_path):
        config = make_config(str(tmp_path), str(tmp_path))
        resolver = PlanResolver.build(config, Dependencies())
        assert isinstance(resolver, Resolver)


# ---------------------------------------------------------------------------
# 2. Required config validation
# ---------------------------------------------------------------------------


class TestPlanResolverConfigValidation:
    def test_missing_project_raises(self, tmp_path):
        config = ResolverConfig(
            type="plan", config={"plans_root": str(tmp_path)}
        )
        with pytest.raises(ValueError, match="project"):
            PlanResolver(config)


# ---------------------------------------------------------------------------
# 3. Zero open plans → only L1 injected
# ---------------------------------------------------------------------------


class TestPlanResolverNoPlans:
    @pytest.mark.asyncio
    async def test_no_plans_dir_returns_empty(self, tmp_path):
        """When plans_root doesn't exist, resolve returns empty content."""
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result, ResolvedContent)
        assert result.resolver_name == "plan"
        assert result.source_layer == "plan"
        # No plans = empty content
        assert result.content[0].text == ""

    @pytest.mark.asyncio
    async def test_empty_plans_dir_returns_empty(self, tmp_path):
        """Empty plans_root yields no L2/L3 content."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(tmp_path))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" not in text
        assert "## Current Task" not in text


# ---------------------------------------------------------------------------
# 4. L1 — Project knowledge injection
# ---------------------------------------------------------------------------


class TestPlanResolverLayer1:
    @pytest.mark.asyncio
    async def test_injects_matching_knowledge_files(self, tmp_path):
        """Knowledge files matching the project are injected as L1."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        write_knowledge_file(knowledge_dir, "architecture.md", body="SOLID principles apply.")

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Project Knowledge" in text
        assert "SOLID principles apply." in text

    @pytest.mark.asyncio
    async def test_excludes_wrong_project_knowledge(self, tmp_path):
        """Knowledge files for other projects are excluded from L1."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        write_knowledge_file(
            knowledge_dir, "other.md", project="other-project", body="Other stuff."
        )

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir),
                knowledge_root=str(knowledge_dir),
                project=PROJECT,
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Other stuff." not in text

    @pytest.mark.asyncio
    async def test_non_knowledge_md_files_skipped(self, tmp_path):
        """Files without kind: project-knowledge are not injected as L1."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        # A file without recognized frontmatter
        (knowledge_dir / "random.md").write_text("# Just a readme\nNothing here.")

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Project Knowledge" not in text

    @pytest.mark.asyncio
    async def test_multiple_knowledge_files_concatenated(self, tmp_path):
        """Multiple knowledge files are concatenated with blank line separator."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        write_knowledge_file(knowledge_dir, "01-architecture.md", body="Arch stuff.")
        write_knowledge_file(knowledge_dir, "02-conventions.md", body="Style guide.")

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Arch stuff." in text
        assert "Style guide." in text


# ---------------------------------------------------------------------------
# 5. L2 — Active plan injection
# ---------------------------------------------------------------------------


class TestPlanResolverLayer2:
    @pytest.mark.asyncio
    async def test_injects_open_plan(self, tmp_path):
        """An open plan's _plan.md body is injected as L2."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "rename-plugin"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Must preserve public API.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" in text
        assert "Must preserve public API." in text

    @pytest.mark.asyncio
    async def test_done_plan_not_injected(self, tmp_path):
        """A plan with status: done is not injected."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "old-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="done", body="Completed work.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" not in text
        assert "Completed work." not in text

    @pytest.mark.asyncio
    async def test_plan_frontmatter_stripped(self, tmp_path):
        """L2 content excludes the YAML frontmatter block."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Body content.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "kind: plan" not in text
        assert "status: open" not in text


# ---------------------------------------------------------------------------
# 6. L3 — Current task injection
# ---------------------------------------------------------------------------


class TestPlanResolverLayer3:
    @pytest.mark.asyncio
    async def test_injects_lowest_order_pending(self, tmp_path):
        """The lowest-order pending task is injected as L3."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1, slug="dir-move", body="Move the directory.")
        write_task_file(plan_dir, order=2, slug="cli-flag", body="Update CLI flags.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Current Task" in text
        assert "Move the directory." in text
        assert "Update CLI flags." not in text

    @pytest.mark.asyncio
    async def test_done_tasks_skipped(self, tmp_path):
        """Tasks with status: done are skipped; next pending is injected."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1, slug="dir-move", status="done", body="Done work.")
        write_task_file(plan_dir, order=2, slug="cli-flag", body="Current work.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Current work." in text
        assert "Done work." not in text

    @pytest.mark.asyncio
    async def test_all_done_no_l3(self, tmp_path):
        """When all tasks are done, no L3 is injected."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1, slug="task1", status="done", body="All done.")
        write_task_file(plan_dir, order=2, slug="task2", status="done", body="All done.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Current Task" not in text

    @pytest.mark.asyncio
    async def test_task_frontmatter_stripped(self, tmp_path):
        """L3 content excludes YAML frontmatter."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        write_task_file(plan_dir, order=1, slug="task1", body="Task body.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "kind: task" not in text
        assert "order: 1" not in text


# ---------------------------------------------------------------------------
# 7. Task advancement (status flip → next task)
# ---------------------------------------------------------------------------


class TestPlanResolverTaskAdvancement:
    @pytest.mark.asyncio
    async def test_flipping_status_advances_to_next(self, tmp_path):
        """Flipping a task's status to done causes next turn to inject the next task."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open")
        task1_file = write_task_file(plan_dir, order=1, slug="task1", body="First task.")
        task2_file = write_task_file(plan_dir, order=2, slug="task2", body="Second task.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))

        # Turn 1: task1 is pending
        result = await resolver.resolve([make_turn_start_event()])
        assert "First task." in result.content[0].text
        assert "Second task." not in result.content[0].text

        # Simulate edi flipping status:done on task1
        task1_text = task1_file.read_text()
        task1_text = task1_text.replace("status: pending", "status: done")
        task1_file.write_text(task1_text)

        # Turn 2: should now inject task2 (dynamic re-read)
        result = await resolver.resolve([make_turn_start_event()])
        assert "Second task." in result.content[0].text
        assert "First task." not in result.content[0].text


# ---------------------------------------------------------------------------
# 8. Layer ordering & delimiters
# ---------------------------------------------------------------------------


class TestPlanResolverLayerOrdering:
    @pytest.mark.asyncio
    async def test_layers_ordered_l1_l2_l3(self, tmp_path):
        """L1 appears before L2, which appears before L3."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        write_knowledge_file(knowledge_dir, "arch.md", body="Knowledge content.")
        write_plan_file(plan_dir, status="open", body="Plan contract.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task details.")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir), knowledge_root=str(knowledge_dir)
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text

        pos_l1 = text.index("## Project Knowledge")
        pos_l2 = text.index("## Active Plan")
        pos_l3 = text.index("## Current Task")
        assert pos_l1 < pos_l2 < pos_l3

    @pytest.mark.asyncio
    async def test_layer_delimiters_present(self, tmp_path):
        """Layer headers and separators are present in the output."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()

        write_knowledge_file(knowledge_dir, "arch.md", body="Knowledge.")
        write_plan_file(plan_dir, status="open", body="Plan.")
        write_task_file(plan_dir, order=1, slug="task1", body="Task.")

        resolver = PlanResolver(
            make_config(
                plans_root=str(plans_dir), knowledge_root=str(knowledge_dir)
            )
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Project Knowledge" in text
        assert "## Active Plan" in text
        assert "## Current Task" in text
        assert "---" in text


# ---------------------------------------------------------------------------
# 9. Multiple open plans → error
# ---------------------------------------------------------------------------


class TestPlanResolverMultipleOpenPlans:
    @pytest.mark.asyncio
    async def test_multiple_open_plans_raises(self, tmp_path):
        """More than one open plan raises PlanResolverError."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        plan_a = plans_dir / "plan-a"
        plan_a.mkdir()
        write_plan_file(plan_a, slug="plan-a", status="open", goal="Plan A.")

        plan_b = plans_dir / "plan-b"
        plan_b.mkdir()
        write_plan_file(plan_b, slug="plan-b", status="open", goal="Plan B.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        with pytest.raises(PlanResolverError, match="Multiple open plans"):
            await resolver.resolve([make_turn_start_event()])

    @pytest.mark.asyncio
    async def test_error_message_lists_slugs(self, tmp_path):
        """Error message includes the plan directory names."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        plan_a = plans_dir / "plan-alpha"
        plan_a.mkdir()
        write_plan_file(plan_a, slug="plan-alpha", status="open", goal="Plan A.")

        plan_b = plans_dir / "plan-beta"
        plan_b.mkdir()
        write_plan_file(plan_b, slug="plan-beta", status="open", goal="Plan B.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        with pytest.raises(PlanResolverError) as exc_info:
            await resolver.resolve([make_turn_start_event()])
        msg = str(exc_info.value)
        assert "plan-alpha" in msg
        assert "plan-beta" in msg


# ---------------------------------------------------------------------------
# 10. Frontmatter tolerance
# ---------------------------------------------------------------------------


class TestPlanResolverFrontmatterTolerance:
    @pytest.mark.asyncio
    async def test_malformed_frontmatter_skipped(self, tmp_path):
        """Files with bad YAML frontmatter are skipped without crashing."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Malformed knowledge file
        kfile = knowledge_dir / "bad.md"
        kfile.write_text(
            "---\nkind: project-knowledge\nproject: test-project\n  bad_indent: [\n---\n\nBody.\n"
        )

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        # Should not crash — just skip the file
        assert isinstance(result, ResolvedContent)

    @pytest.mark.asyncio
    async def test_missing_kind_skipped(self, tmp_path):
        """Files without a 'kind' field are skipped."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        (knowledge_dir / "no-kind.md").write_text(
            "---\nproject: test-project\n---\n\nSome content.\n"
        )

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Some content." not in text

    @pytest.mark.asyncio
    async def test_unknown_kind_skipped(self, tmp_path):
        """Files with unrecognized 'kind' are skipped."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        (knowledge_dir / "weird.md").write_text(
            "---\nkind: recipe\n---\n\nPancake recipe.\n"
        )

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Pancake recipe." not in text

    @pytest.mark.asyncio
    async def test_no_frontmatter_file_skipped(self, tmp_path):
        """Files without any frontmatter are skipped."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        (knowledge_dir / "plain.md").write_text("# Just a heading\nNo frontmatter.")

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "No frontmatter" not in text


# ---------------------------------------------------------------------------
# 11. Token budget enforcement
# ---------------------------------------------------------------------------


class TestPlanResolverTokenBudget:
    @pytest.mark.asyncio
    async def test_within_budget_no_truncation(self, tmp_path):
        """Content within max_tokens budget is returned intact."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "small-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Short content.")

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), max_tokens=10000)
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert "Short content." in result.content[0].text

    @pytest.mark.asyncio
    async def test_budget_exceeded_truncates(self, tmp_path):
        """Content exceeding max_tokens is truncated."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "big-plan"
        plan_dir.mkdir()
        # Create a very long body
        write_plan_file(plan_dir, status="open", body="X" * 4000)

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), max_tokens=10)
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        # Should have a truncation notice
        assert "truncated" in text.lower()
        # Should be much shorter than original
        assert len(text) < 4000

    @pytest.mark.asyncio
    async def test_no_budget_no_truncation(self, tmp_path):
        """Without max_tokens, any size content is returned."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "big-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="X" * 50000)

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "X" * 50000 in text


# ---------------------------------------------------------------------------
# 12. Read-only resolver
# ---------------------------------------------------------------------------


class TestPlanResolverReadOnly:
    @pytest.mark.asyncio
    async def test_resolver_never_writes(self, tmp_path):
        """PlanResolver never modifies plan files on disk."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Original content.")
        task_file = write_task_file(plan_dir, order=1, slug="task1", body="Task body.")

        # Record file contents before resolve
        original_plan = (plan_dir / "_plan.md").read_text()
        original_task = task_file.read_text()

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        await resolver.resolve([make_turn_start_event()])
        await resolver.resolve([make_turn_start_event()])
        await resolver.resolve([make_turn_start_event()])

        # Verify nothing changed
        assert (plan_dir / "_plan.md").read_text() == original_plan
        assert task_file.read_text() == original_task


# ---------------------------------------------------------------------------
# 13. Planning guide NOT auto-injected
# ---------------------------------------------------------------------------


class TestPlanResolverPlanningGuide:
    @pytest.mark.asyncio
    async def test_planning_guide_not_injected(self, tmp_path):
        """A file named planning-guide.md is NOT auto-injected as L1.

        The planning guide doesn't carry kind: project-knowledge, so it
        should be excluded from L1 selection.
        """
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # planning-guide.md without kind: project-knowledge
        (knowledge_dir / "planning-guide.md").write_text(
            "# Planning Guide\n\nFollow this protocol for multi-step tasks.\n"
        )

        resolver = PlanResolver(
            make_config(plans_root=str(plans_dir), knowledge_root=str(knowledge_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "Planning Guide" not in text
        assert "multi-step tasks" not in text


# ---------------------------------------------------------------------------
# 14. Mid-run discovery
# ---------------------------------------------------------------------------


class TestPlanResolverMidRunDiscovery:
    @pytest.mark.asyncio
    async def test_new_plan_discovered_on_next_turn(self, tmp_path):
        """A plan directory created mid-run is discovered on the next resolve."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))

        # Turn 1: no plans
        result = await resolver.resolve([make_turn_start_event()])
        assert "## Active Plan" not in result.content[0].text

        # Create a plan mid-run
        plan_dir = plans_dir / "new-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="New plan contract.")
        write_task_file(plan_dir, order=1, slug="task1", body="New task.")

        # Turn 2: should discover the new plan
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" in text
        assert "New plan contract." in text
        assert "## Current Task" in text
        assert "New task." in text


# ---------------------------------------------------------------------------
# 15. Execution count
# ---------------------------------------------------------------------------


class TestPlanResolverExecutionCount:
    @pytest.mark.asyncio
    async def test_execution_count_increments(self, tmp_path):
        """execution_count increments after each resolve() call."""
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        assert resolver.execution_count == 0
        await resolver.resolve([make_turn_start_event()])
        assert resolver.execution_count == 1
        await resolver.resolve([make_turn_start_event()])
        assert resolver.execution_count == 2


# ---------------------------------------------------------------------------
# 16. ResolvedContent structure
# ---------------------------------------------------------------------------


class TestPlanResolverResolvedContent:
    @pytest.mark.asyncio
    async def test_resolved_content_structure(self, tmp_path):
        """ResolvedContent has correct resolver_name and source_layer."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan body.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])

        assert result.resolver_name == "plan"
        assert result.source_layer == "plan"
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)

    @pytest.mark.asyncio
    async def test_token_count_populated(self, tmp_path):
        """ResolvedContent.token_count reflects actual content size."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "test-plan"
        plan_dir.mkdir()
        write_plan_file(plan_dir, status="open", body="Plan body.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result.token_count, int)
        assert result.token_count > 0


# ---------------------------------------------------------------------------
# 17. Default root paths
# ---------------------------------------------------------------------------


class TestPlanResolverDefaultPaths:
    def test_default_plans_root(self, tmp_path, monkeypatch):
        """Default plans_root expands to ~/.sr2/plans."""
        # Use a fake home
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        resolver = PlanResolver(make_config())
        assert resolver._plans_root == tmp_path / ".sr2" / "plans"

    def test_default_knowledge_root(self, tmp_path, monkeypatch):
        """Default knowledge_root expands to ~/.sr2/knowledge/<project>."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        resolver = PlanResolver(make_config(project="myproject"))
        assert resolver._knowledge_root == tmp_path / ".sr2" / "knowledge" / "myproject"


# ---------------------------------------------------------------------------
# 18. Edge cases
# ---------------------------------------------------------------------------


class TestPlanResolverEdgeCases:
    @pytest.mark.asyncio
    async def test_plan_without_plan_md_skipped(self, tmp_path):
        """A plan directory without _plan.md is ignored."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "no-plan-md"
        plan_dir.mkdir()
        write_task_file(plan_dir, order=1, slug="task1", body="Orphan task.")

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" not in text

    @pytest.mark.asyncio
    async def test_plan_md_without_body_returns_empty_l2(self, tmp_path):
        """A _plan.md with only frontmatter (no body) returns empty L2 content."""
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()
        plan_dir = plans_dir / "empty-plan"
        plan_dir.mkdir()
        (plan_dir / "_plan.md").write_text(
            "---\nkind: plan\nslug: empty\nstatus: open\ngoal: \"Empty\"\n---\n"
        )

        resolver = PlanResolver(make_config(plans_root=str(plans_dir)))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "## Active Plan" in text

    @pytest.mark.asyncio
    async def test_resolved_content_has_entries_field(self, tmp_path):
        """ResolvedContent includes entries field (for provenance)."""
        resolver = PlanResolver(make_config(str(tmp_path), str(tmp_path)))
        result = await resolver.resolve([make_turn_start_event()])
        assert hasattr(result, "entries")
        assert isinstance(result.entries, list)
