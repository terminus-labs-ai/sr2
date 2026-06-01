"""Tests for frontmatter parsing (FR5).

Covers:
  - extract_raw_frontmatter: boundary cases for the --- --- extraction.
  - parse_frontmatter: all three kinds, tolerance, error handling.
  - parse_file: disk I/O wrapping.
  - Models: frozen dataclasses, enum values.

Acceptance criteria:
  - Files lacking a recognized 'kind' or that fail to parse are skipped (logged),
    never crash resolve.
  - YAML parser reuses SR2-bundled PyYAML (no new dep).
"""

import logging
import pytest
import yaml as _yaml
from pathlib import Path
from textwrap import dedent

from sr2.pipeline.plan.frontmatter import (
    extract_raw_frontmatter,
    parse_file,
    parse_frontmatter,
)
from sr2.pipeline.plan.models import (
    KnowledgeFrontmatter,
    PlanFrontmatter,
    PlanStatus,
    RECOGNIZED_KINDS,
    TaskFrontmatter,
    TaskStatus,
    get_frontmatter_class,
)


# =========================================================================
# 1. extract_raw_frontmatter
# =========================================================================


class TestExtractRawFrontmatter:
    def test_simple_block(self):
        """Basic frontmatter block between --- delimiters."""
        text = "---\nkind: task\norder: 1\n---\n# Body\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None
        assert "kind: task" in raw
        assert "order: 1" in raw

    def test_no_frontmatter(self):
        """Text without --- prefix returns None."""
        text = "# Just a heading\n\nSome body text.\n"
        assert extract_raw_frontmatter(text) is None

    def test_single_dash_no_close(self):
        """Opening --- but no closing --- returns None."""
        text = "---\nkind: task\norder: 1\n# No close\n"
        assert extract_raw_frontmatter(text) is None

    def test_empty_block(self):
        """Empty frontmatter (--- followed immediately by ---) returns None."""
        text = "---\n---\n# Body\n"
        assert extract_raw_frontmatter(text) is None

    def test_whitespace_before_block(self):
        """Leading whitespace is stripped before checking for ---."""
        text = "\n\n---\nkind: plan\n---\n# Body\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None

    def test_multiline_yaml(self):
        """Multi-line YAML values are extracted correctly."""
        text = "---\nkind: task\nverify: |\n  uv run pytest\n  uv run mypy\n---\n# Body\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None
        assert "uv run pytest" in raw

    def test_block_with_horizontal_rules(self):
        """Body contains --- (horizontal rules) after frontmatter closes."""
        text = "---\nkind: task\n---\n# Body\n\n---\n\nMore text\n"
        raw = extract_raw_frontmatter(text)
        assert raw is not None
        assert "kind: task" in raw


# =========================================================================
# 2. parse_frontmatter — task kind
# =========================================================================


class TestParseTaskFrontmatter:
    def _task_text(self, yaml_block: str, body: str = "# Body\n") -> str:
        return f"---\n{yaml_block}\n---\n{body}"

    def test_minimal_task(self):
        """Only kind: task — all other fields get defaults."""
        text = self._task_text("kind: task")
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)
        assert result.kind == "task"
        assert result.plan == ""
        assert result.order == 0
        assert result.status == TaskStatus.PENDING
        assert result.verify == ""
        assert result.title == ""

    def test_full_task(self):
        """All fields populated."""
        text = self._task_text(
            "kind: task\nplan: rename-plugin\norder: 2\n"
            "status: pending\nverify: uv run pytest\n"
            'title: "Rename Plugin to Interface"'
        )
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)
        assert result.plan == "rename-plugin"
        assert result.order == 2
        assert result.status == TaskStatus.PENDING
        assert result.verify == "uv run pytest"
        assert result.title == "Rename Plugin to Interface"

    def test_done_status(self):
        """status: done is parsed correctly."""
        text = self._task_text("kind: task\nstatus: done")
        result = parse_frontmatter(text)
        assert result.status == TaskStatus.DONE

    def test_invalid_status_defaults_to_pending(self):
        """Invalid status value falls back to pending (tolerance)."""
        text = self._task_text("kind: task\nstatus: banana")
        result = parse_frontmatter(text)
        assert result is not None
        assert result.status == TaskStatus.PENDING

    def test_missing_optional_fields(self):
        """Missing optional fields don't crash — get defaults."""
        text = self._task_text("kind: task\norder: 5")
        result = parse_frontmatter(text)
        assert result.order == 5
        assert result.plan == ""
        assert result.verify == ""

    def test_negative_order(self):
        """Negative order is accepted (int)."""
        text = self._task_text("kind: task\norder: -1")
        result = parse_frontmatter(text)
        assert result.order == -1

    def test_non_integer_order_defaults_to_zero(self):
        """Non-integer order falls back to 0 (tolerance)."""
        text = self._task_text("kind: task\norder: abc")
        result = parse_frontmatter(text)
        assert result.order == 0


# =========================================================================
# 3. parse_frontmatter — plan kind
# =========================================================================


class TestParsePlanFrontmatter:
    def _plan_text(self, yaml_block: str) -> str:
        return f"---\n{yaml_block}\n---\n# Body\n"

    def test_minimal_plan(self):
        text = self._plan_text("kind: plan")
        result = parse_frontmatter(text)
        assert isinstance(result, PlanFrontmatter)
        assert result.kind == "plan"
        assert result.slug == ""
        assert result.status == PlanStatus.OPEN
        assert result.goal == ""

    def test_full_plan(self):
        text = self._plan_text(
            "kind: plan\nslug: rename-plugin\nstatus: open\n"
            'goal: "Rename Plugin to Interface across 9 files"'
        )
        result = parse_frontmatter(text)
        assert result.slug == "rename-plugin"
        assert result.status == PlanStatus.OPEN
        assert result.goal == "Rename Plugin to Interface across 9 files"

    def test_done_plan(self):
        text = self._plan_text("kind: plan\nstatus: done")
        result = parse_frontmatter(text)
        assert result.status == PlanStatus.DONE

    def test_invalid_status_defaults_to_open(self):
        text = self._plan_text("kind: plan\nstatus: woot")
        result = parse_frontmatter(text)
        assert result is not None
        assert result.status == PlanStatus.OPEN


# =========================================================================
# 4. parse_frontmatter — project-knowledge kind
# =========================================================================


class TestParseKnowledgeFrontmatter:
    def _knowledge_text(self, yaml_block: str) -> str:
        return f"---\n{yaml_block}\n---\n# Body\n"

    def test_minimal_knowledge(self):
        text = self._knowledge_text("kind: project-knowledge")
        result = parse_frontmatter(text)
        assert isinstance(result, KnowledgeFrontmatter)
        assert result.kind == "project-knowledge"
        assert result.project == ""

    def test_full_knowledge(self):
        text = self._knowledge_text(
            "kind: project-knowledge\nproject: sr2-spectre"
        )
        result = parse_frontmatter(text)
        assert result.project == "sr2-spectre"

    def test_knowledge_matches_project(self):
        """Knowledge project field is used for matching in resolver."""
        text = self._knowledge_text("kind: project-knowledge\nproject: my-project")
        result = parse_frontmatter(text)
        assert result.project == "my-project"


# =========================================================================
# 5. Tolerance — skip, never crash
# =========================================================================


class TestParseFrontmatterTolerance:
    def test_no_frontmatter_returns_none(self):
        """Files without frontmatter return None (skip)."""
        result = parse_frontmatter("# Just a heading\n\nBody text.\n")
        assert result is None

    def test_missing_kind_returns_none(self):
        """Frontmatter without 'kind' returns None."""
        text = "---\norder: 1\nstatus: pending\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_unrecognized_kind_returns_none(self):
        """Unrecognized kind returns None (skip, no crash)."""
        text = "---\nkind: recipe\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_invalid_yaml_returns_none(self):
        """Malformed YAML returns None (skip, no crash)."""
        text = "---\nkind: task\n  order: [invalid yaml\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_non_mapping_yaml_returns_none(self):
        """YAML that parses to a non-dict (e.g., a list) returns None."""
        text = "---\n- item1\n- item2\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        result = parse_frontmatter("")
        assert result is None

    def test_only_dashes_returns_none(self):
        """Just `---` with no body returns None."""
        result = parse_frontmatter("---")
        assert result is None

    def test_frontmatter_with_body_preserved_in_text(self):
        """parse_frontmatter only extracts frontmatter; the body remains in the
        original text. The function doesn't strip or modify the input."""
        text = "---\nkind: task\n---\n# Important Body\n"
        result = parse_frontmatter(text)
        assert result is not None
        assert "# Important Body" in text  # input unchanged

    def test_case_insensitive_kind(self):
        """Kind matching is case-insensitive."""
        text = "---\nkind: TASK\n---\n# Body\n"
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)

    def test_kind_with_whitespace(self):
        """Kind with surrounding whitespace is trimmed."""
        text = "---\nkind:  task  \n---\n# Body\n"
        result = parse_frontmatter(text)
        assert isinstance(result, TaskFrontmatter)


# =========================================================================
# 6. parse_file — disk I/O
# =========================================================================


class TestParseFile:
    def test_parses_valid_file(self, tmp_path):
        content = "---\nkind: task\norder: 1\nstatus: pending\n---\n# Body\n"
        file = tmp_path / "01-test.md"
        file.write_text(content)
        result = parse_file(file)
        assert isinstance(result, TaskFrontmatter)
        assert result.order == 1

    def test_missing_file_returns_none(self, tmp_path):
        result = parse_file(tmp_path / "nonexistent.md")
        assert result is None

    def test_unreadable_file_returns_none(self, tmp_path):
        file = tmp_path / "locked.md"
        file.write_text("---\nkind: task\n---\n")
        file.chmod(0o000)
        try:
            result = parse_file(file)
            assert result is None
        finally:
            file.chmod(0o644)  # cleanup for deletion

    def test_file_without_frontmatter_returns_none(self, tmp_path):
        file = tmp_path / "readme.md"
        file.write_text("# Just a README\n\nNo frontmatter here.\n")
        result = parse_file(file)
        assert result is None


# =========================================================================
# 7. Models — frozen dataclasses & enums
# =========================================================================


class TestModels:
    def test_task_frontmatter_frozen(self):
        """TaskFrontmatter is frozen (immutable)."""
        fm = TaskFrontmatter(plan="test", order=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            fm.plan = "other"

    def test_plan_frontmatter_frozen(self):
        fm = PlanFrontmatter(slug="test")
        with pytest.raises(Exception):
            fm.slug = "other"

    def test_knowledge_frontmatter_frozen(self):
        fm = KnowledgeFrontmatter(project="test")
        with pytest.raises(Exception):
            fm.project = "other"

    def test_recommended_kinds_set(self):
        """RECOGNIZED_KINDS contains all three kinds."""
        assert "task" in RECOGNIZED_KINDS
        assert "plan" in RECOGNIZED_KINDS
        assert "project-knowledge" in RECOGNIZED_KINDS

    def test_get_frontmatter_class(self):
        assert get_frontmatter_class("task") is TaskFrontmatter
        assert get_frontmatter_class("plan") is PlanFrontmatter
        assert get_frontmatter_class("project-knowledge") is KnowledgeFrontmatter
        assert get_frontmatter_class("unknown") is None

    def test_task_status_enum(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.DONE.value == "done"

    def test_plan_status_enum(self):
        assert PlanStatus.OPEN.value == "open"
        assert PlanStatus.DONE.value == "done"

    def test_task_status_from_string(self):
        assert TaskStatus("pending") == TaskStatus.PENDING
        assert TaskStatus("done") == TaskStatus.DONE

    def test_plan_status_from_string(self):
        assert PlanStatus("open") == PlanStatus.OPEN
        assert PlanStatus("done") == PlanStatus.DONE


# =========================================================================
# 8. Logging — warnings emitted on skip
# =========================================================================


class TestLoggingOnSkip:
    def test_warning_on_unrecognized_kind(self, caplog):
        """A warning is logged when kind is unrecognized."""
        caplog.set_level(logging.WARNING, logger="sr2.pipeline.plan.frontmatter")
        text = "---\nkind: recipe\n---\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "Unrecognized kind" in caplog.text

    def test_warning_on_missing_kind(self, caplog):
        """A warning is logged when kind is missing."""
        caplog.set_level(logging.WARNING, logger="sr2.pipeline.plan.frontmatter")
        text = "---\norder: 1\n---\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "No 'kind'" in caplog.text

    def test_warning_on_yaml_error(self, caplog):
        """A warning is logged on YAML parse errors."""
        caplog.set_level(logging.WARNING, logger="sr2.pipeline.plan.frontmatter")
        text = "---\nkind: [broken yaml\n---\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "YAML parse error" in caplog.text

    def test_debug_on_no_frontmatter(self, caplog):
        """Debug-level log when no frontmatter found."""
        caplog.set_level(logging.DEBUG, logger="sr2.pipeline.plan.frontmatter")
        text = "# Just a heading\n"
        result = parse_frontmatter(text, file_path=Path("/test.md"))
        assert result is None
        assert "No frontmatter" in caplog.text


# =========================================================================
# 9. PyYAML reuse (no new dependency)
# =========================================================================


class TestPyYAMLReuse:
    def test_uses_builtin_yaml(self):
        """The frontmatter module uses yaml from SR2's existing PyYAML dep."""
        from sr2.pipeline.plan import frontmatter
        # Verify yaml module is the standard PyYAML
        assert frontmatter.yaml is not None
        # Verify it can parse (smoke test)
        data = frontmatter.yaml.safe_load("key: value")
        assert data == {"key": "value"}

    def test_no_external_deps_needed(self):
        """frontmatter.py only imports yaml (PyYAML) which is already in sr2 deps."""
        import ast
        import inspect

        source = inspect.getsource(__import__("sr2.pipeline.plan.frontmatter", fromlist=[""]))
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        # The only external (non-sr2) import should be yaml
        external = [i for i in imports if not i.startswith("sr2.") and i not in (
            "__future__", "logging", "pathlib", "typing"
        )]
        assert external == ["yaml"], f"Unexpected external deps: {external}"
