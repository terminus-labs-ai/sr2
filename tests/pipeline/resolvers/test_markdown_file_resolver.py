"""Tests for MarkdownFileResolver.

Covers:
  - Single file: loaded and returned as a single TextBlock
  - Multiple files via glob: sorted alphabetically, concatenated with \\n separator
  - Empty glob match: raises FileNotFoundError with a clear message
  - max_tokens budget: within budget → no error at init
  - max_tokens budget exceeded → raises MarkdownTokenBudgetError at init time
  - Budget error message: itemizes each file path + token count, total, budget, null hint
  - max_tokens=None → no enforcement, any size accepted
  - resolve() subscribes to turn_start/STARTING by default
  - declaring_dir: relative glob resolved against declaring_dir
"""

from __future__ import annotations

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.protocols import Resolver
from sr2.pipeline.resolvers.markdown_file import (
    MarkdownFileResolver,
    MarkdownTokenBudgetError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    path: str,
    max_tokens: int | None = None,
    declaring_dir: str | None = None,
    **kwargs,
) -> ResolverConfig:
    """Build a ResolverConfig for MarkdownFileResolver."""
    cfg: dict = {"path": path}
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    if declaring_dir is not None:
        cfg["declaring_dir"] = declaring_dir
    return ResolverConfig(type="markdown_file", config=cfg, **kwargs)


def make_turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


# ---------------------------------------------------------------------------
# 1. Single file
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverSingleFile:
    @pytest.mark.asyncio
    async def test_single_file_returns_resolved_content(self, tmp_path):
        """A single file path returns ResolvedContent."""
        f = tmp_path / "doc.md"
        f.write_text("# Hello\nWorld")
        resolver = MarkdownFileResolver(make_config(str(f)))
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result, ResolvedContent)

    @pytest.mark.asyncio
    async def test_single_file_content_has_one_text_block(self, tmp_path):
        """A single file produces exactly one TextBlock."""
        f = tmp_path / "doc.md"
        f.write_text("# Hello\nWorld")
        resolver = MarkdownFileResolver(make_config(str(f)))
        result = await resolver.resolve([make_turn_start_event()])
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)

    @pytest.mark.asyncio
    async def test_single_file_text_matches_file_contents(self, tmp_path):
        """TextBlock text must match the file's content exactly."""
        f = tmp_path / "doc.md"
        content = "# System Prompt\nYou are a helpful assistant."
        f.write_text(content)
        resolver = MarkdownFileResolver(make_config(str(f)))
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == content

    @pytest.mark.asyncio
    async def test_resolver_name_is_markdown_file(self, tmp_path):
        """ResolvedContent.resolver_name must be 'markdown_file'."""
        f = tmp_path / "doc.md"
        f.write_text("content")
        resolver = MarkdownFileResolver(make_config(str(f)))
        result = await resolver.resolve([make_turn_start_event()])
        assert result.resolver_name == "markdown_file"

    @pytest.mark.asyncio
    async def test_source_layer_is_markdown_file(self, tmp_path):
        """ResolvedContent.source_layer must be 'markdown_file'."""
        f = tmp_path / "doc.md"
        f.write_text("content")
        resolver = MarkdownFileResolver(make_config(str(f)))
        result = await resolver.resolve([make_turn_start_event()])
        assert result.source_layer == "markdown_file"


# ---------------------------------------------------------------------------
# 2. Multiple files via glob — sorted alphabetically, joined with \\n
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverGlob:
    @pytest.mark.asyncio
    async def test_glob_returns_multiple_files(self, tmp_path):
        """Glob expansion loads all matching files."""
        (tmp_path / "a.md").write_text("AAA")
        (tmp_path / "b.md").write_text("BBB")
        (tmp_path / "c.md").write_text("CCC")
        resolver = MarkdownFileResolver(make_config(str(tmp_path / "*.md")))
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result.content[0], TextBlock)
        combined = result.content[0].text
        assert "AAA" in combined
        assert "BBB" in combined
        assert "CCC" in combined

    @pytest.mark.asyncio
    async def test_glob_results_are_sorted_alphabetically(self, tmp_path):
        """Files from glob are sorted alphabetically by absolute path."""
        (tmp_path / "z.md").write_text("LAST")
        (tmp_path / "a.md").write_text("FIRST")
        (tmp_path / "m.md").write_text("MIDDLE")
        resolver = MarkdownFileResolver(make_config(str(tmp_path / "*.md")))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        pos_a = text.index("FIRST")
        pos_m = text.index("MIDDLE")
        pos_z = text.index("LAST")
        assert pos_a < pos_m < pos_z

    @pytest.mark.asyncio
    async def test_glob_files_joined_with_newline_separator(self, tmp_path):
        """Files are concatenated with \\n between them (not within)."""
        (tmp_path / "a.md").write_text("AAA")
        (tmp_path / "b.md").write_text("BBB")
        resolver = MarkdownFileResolver(make_config(str(tmp_path / "*.md")))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        # Separator must be exactly \\n between file contents
        assert text == "AAA\nBBB"

    @pytest.mark.asyncio
    async def test_glob_non_matching_extensions_excluded(self, tmp_path):
        """Glob only matches specified pattern — other files are excluded."""
        (tmp_path / "a.md").write_text("MARKDOWN")
        (tmp_path / "b.txt").write_text("TEXT")
        resolver = MarkdownFileResolver(make_config(str(tmp_path / "*.md")))
        result = await resolver.resolve([make_turn_start_event()])
        text = result.content[0].text
        assert "MARKDOWN" in text
        assert "TEXT" not in text


# ---------------------------------------------------------------------------
# 3. Empty glob match — raises FileNotFoundError
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverEmptyGlob:
    def test_empty_glob_raises_at_init(self, tmp_path):
        """Empty glob match raises FileNotFoundError at __init__ time."""
        pattern = str(tmp_path / "*.md")
        # tmp_path has no .md files
        with pytest.raises(FileNotFoundError):
            MarkdownFileResolver(make_config(pattern))

    def test_empty_glob_error_message_includes_pattern(self, tmp_path):
        """FileNotFoundError message includes the glob pattern."""
        pattern = str(tmp_path / "*.md")
        with pytest.raises(FileNotFoundError, match=r"\*\.md"):
            MarkdownFileResolver(make_config(pattern))

    def test_nonexistent_single_file_raises_at_init(self, tmp_path):
        """A single path that does not exist raises FileNotFoundError at init."""
        missing = str(tmp_path / "missing.md")
        with pytest.raises(FileNotFoundError):
            MarkdownFileResolver(make_config(missing))


# ---------------------------------------------------------------------------
# 4. max_tokens budget enforcement at init
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverTokenBudget:
    def test_within_budget_does_not_raise(self, tmp_path):
        """Files within max_tokens budget: no error at init."""
        f = tmp_path / "small.md"
        f.write_text("x" * 100)  # 100 chars ≈ 25 tokens
        # Budget is comfortably above 25 tokens
        resolver = MarkdownFileResolver(make_config(str(f), max_tokens=1000))
        assert resolver is not None

    def test_budget_exceeded_raises_at_init(self, tmp_path):
        """Files exceeding max_tokens raise MarkdownTokenBudgetError at init."""
        f = tmp_path / "big.md"
        f.write_text("x" * 400)  # 400 chars ≈ 100 tokens
        with pytest.raises(MarkdownTokenBudgetError):
            MarkdownFileResolver(make_config(str(f), max_tokens=10))

    def test_multi_file_glob_total_budget_exceeded(self, tmp_path):
        """Multiple files: total token count is checked against max_tokens."""
        # Each file is within budget individually, but combined they exceed it.
        (tmp_path / "a.md").write_text("x" * 80)  # 20 tokens
        (tmp_path / "b.md").write_text("x" * 80)  # 20 tokens
        # Combined = 40 tokens, budget = 30 → should raise
        with pytest.raises(MarkdownTokenBudgetError):
            MarkdownFileResolver(make_config(str(tmp_path / "*.md"), max_tokens=30))

    def test_budget_none_no_enforcement(self, tmp_path):
        """max_tokens=None disables budget enforcement — any size is OK."""
        f = tmp_path / "big.md"
        f.write_text("x" * 40_000)  # large file
        resolver = MarkdownFileResolver(make_config(str(f), max_tokens=None))
        assert resolver is not None

    def test_budget_exact_boundary_no_error(self, tmp_path):
        """Exactly at the token budget does not raise."""
        # CharacterTokenCounter: tokens = chars // 4
        # 40 chars → 10 tokens. Budget = 10 → should be OK (<=).
        f = tmp_path / "doc.md"
        f.write_text("x" * 40)
        resolver = MarkdownFileResolver(make_config(str(f), max_tokens=10))
        assert resolver is not None

    def test_budget_one_over_raises(self, tmp_path):
        """One token over the budget raises MarkdownTokenBudgetError."""
        # 44 chars → 11 tokens. Budget = 10 → should raise.
        f = tmp_path / "doc.md"
        f.write_text("x" * 44)
        with pytest.raises(MarkdownTokenBudgetError):
            MarkdownFileResolver(make_config(str(f), max_tokens=10))


# ---------------------------------------------------------------------------
# 5. Budget error message content
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverBudgetErrorMessage:
    def test_error_message_includes_file_path(self, tmp_path):
        """Error message includes each file path."""
        f = tmp_path / "big.md"
        f.write_text("x" * 400)
        with pytest.raises(MarkdownTokenBudgetError) as exc_info:
            MarkdownFileResolver(make_config(str(f), max_tokens=10))
        assert str(f) in str(exc_info.value)

    def test_error_message_includes_token_count(self, tmp_path):
        """Error message includes token count for each file."""
        f = tmp_path / "big.md"
        f.write_text("x" * 400)  # 400 chars // 4 = 100 tokens
        with pytest.raises(MarkdownTokenBudgetError) as exc_info:
            MarkdownFileResolver(make_config(str(f), max_tokens=10))
        assert "100" in str(exc_info.value)

    def test_error_message_includes_total(self, tmp_path):
        """Error message includes total token count."""
        f = tmp_path / "big.md"
        f.write_text("x" * 400)
        with pytest.raises(MarkdownTokenBudgetError) as exc_info:
            MarkdownFileResolver(make_config(str(f), max_tokens=10))
        msg = str(exc_info.value)
        # Total and budget both appear
        assert "100" in msg
        assert "10" in msg

    def test_error_message_includes_null_hint(self, tmp_path):
        """Error message mentions max_tokens: null to disable budget."""
        f = tmp_path / "big.md"
        f.write_text("x" * 400)
        with pytest.raises(MarkdownTokenBudgetError) as exc_info:
            MarkdownFileResolver(make_config(str(f), max_tokens=10))
        msg = str(exc_info.value)
        assert "max_tokens" in msg
        assert "null" in msg or "None" in msg

    def test_error_message_multi_file_itemizes_each(self, tmp_path):
        """With multiple files exceeding budget, each file is itemized."""
        (tmp_path / "a.md").write_text("x" * 200)  # 50 tokens
        (tmp_path / "b.md").write_text("x" * 200)  # 50 tokens
        with pytest.raises(MarkdownTokenBudgetError) as exc_info:
            MarkdownFileResolver(make_config(str(tmp_path / "*.md"), max_tokens=10))
        msg = str(exc_info.value)
        assert "a.md" in msg
        assert "b.md" in msg


# ---------------------------------------------------------------------------
# 6. Default subscriptions (turn_start / STARTING)
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverSubscriptions:
    def test_default_subscription_is_turn_start(self, tmp_path):
        """Default subscription fires on turn_start."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        names = [s.event_name for s in resolver.subscriptions]
        assert "turn_start" in names

    def test_default_subscription_phase_is_starting(self, tmp_path):
        """Default turn_start subscription uses EventPhase.STARTING."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        turn_start_subs = [
            s for s in resolver.subscriptions if s.event_name == "turn_start"
        ]
        assert any(s.phase == EventPhase.STARTING for s in turn_start_subs)

    def test_satisfies_resolver_protocol(self, tmp_path):
        """MarkdownFileResolver satisfies isinstance(x, Resolver)."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        assert isinstance(resolver, Resolver)

    def test_has_name_attribute(self, tmp_path):
        """Resolver exposes a non-empty name attribute."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        assert hasattr(resolver, "name")
        assert resolver.name

    def test_has_max_executions_attribute(self, tmp_path):
        """Resolver exposes max_executions from config."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        assert hasattr(resolver, "max_executions")
        assert isinstance(resolver.max_executions, int)

    def test_has_execution_count_attribute(self, tmp_path):
        """Resolver exposes execution_count initialized to 0."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        assert resolver.execution_count == 0

    @pytest.mark.asyncio
    async def test_execution_count_increments_on_resolve(self, tmp_path):
        """execution_count increments after each resolve() call."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        resolver = MarkdownFileResolver(make_config(str(f)))
        await resolver.resolve([make_turn_start_event()])
        assert resolver.execution_count == 1
        await resolver.resolve([make_turn_start_event()])
        assert resolver.execution_count == 2


# ---------------------------------------------------------------------------
# 7. declaring_dir: relative glob resolved against it
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverDeclaringDir:
    def test_relative_glob_resolved_against_declaring_dir(self, tmp_path):
        """Relative glob pattern is resolved relative to declaring_dir."""
        subdir = tmp_path / "prompts"
        subdir.mkdir()
        (subdir / "sys.md").write_text("SYSTEM")
        # Path is relative — declaring_dir provides the base
        resolver = MarkdownFileResolver(
            make_config("prompts/*.md", declaring_dir=str(tmp_path))
        )
        assert resolver is not None

    @pytest.mark.asyncio
    async def test_relative_glob_loads_correct_file(self, tmp_path):
        """Relative glob loads the right file when resolved against declaring_dir."""
        subdir = tmp_path / "prompts"
        subdir.mkdir()
        (subdir / "sys.md").write_text("SYSTEM PROMPT")
        resolver = MarkdownFileResolver(
            make_config("prompts/*.md", declaring_dir=str(tmp_path))
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "SYSTEM PROMPT"

    def test_absolute_path_ignores_declaring_dir(self, tmp_path):
        """When path is absolute, declaring_dir has no effect."""
        f = tmp_path / "doc.md"
        f.write_text("ABSOLUTE")
        # declaring_dir points somewhere else — should be ignored
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        resolver = MarkdownFileResolver(
            make_config(str(f), declaring_dir=str(other_dir))
        )
        assert resolver is not None

    @pytest.mark.asyncio
    async def test_absolute_path_with_declaring_dir_loads_correct_file(self, tmp_path):
        """Absolute path loads the correct file regardless of declaring_dir."""
        f = tmp_path / "doc.md"
        f.write_text("ABSOLUTE CONTENT")
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        resolver = MarkdownFileResolver(
            make_config(str(f), declaring_dir=str(other_dir))
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "ABSOLUTE CONTENT"


# ---------------------------------------------------------------------------
# 8. build() classmethod
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverBuild:
    def test_build_returns_instance(self, tmp_path):
        """build() classmethod returns a MarkdownFileResolver instance."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        config = make_config(str(f))
        result = MarkdownFileResolver.build(config, Dependencies())
        assert isinstance(result, MarkdownFileResolver)

    def test_build_satisfies_resolver_protocol(self, tmp_path):
        """Instance from build() satisfies the Resolver protocol."""
        f = tmp_path / "doc.md"
        f.write_text("hello")
        config = make_config(str(f))
        result = MarkdownFileResolver.build(config, Dependencies())
        assert isinstance(result, Resolver)


# ===========================================================================
# SLICE 2: {area} templated-path mode (AC 13-22, FR 16-25)
# ===========================================================================
#
# These tests extend the resolver with resolve-time {area} templating. They
# are RED against the current init-time-only implementation and become GREEN
# once the feature lands.

import logging
import os
import pathlib
import re


# ---------------------------------------------------------------------------
# Helpers for templated-mode tests
# ---------------------------------------------------------------------------


def make_area_config(
    path: str,
    on_missing: str | None = None,
    max_tokens: int | None = None,
    declaring_dir: str | None = None,
    **kwargs,
) -> ResolverConfig:
    """Build a ResolverConfig, placing on_missing inside the config dict."""
    cfg: dict = {"path": path}
    if on_missing is not None:
        cfg["on_missing"] = on_missing
    if max_tokens is not None:
        cfg["max_tokens"] = max_tokens
    if declaring_dir is not None:
        cfg["declaring_dir"] = declaring_dir
    return ResolverConfig(type="markdown_file", config=cfg, **kwargs)


def area_deps(area) -> Dependencies:
    """Dependencies whose run_context_provider yields the given area.

    - area is a str  -> provider returns {"area": area}
    - area is None   -> provider returns None (no run context)
    - area == "ABSENT" sentinel handled by caller via absent_deps()
    """
    if area is None:
        return Dependencies(run_context_provider=lambda: None)
    return Dependencies(run_context_provider=lambda: {"area": area})


def absent_area_deps() -> Dependencies:
    """Provider returns a dict without the 'area' key (interface resolves no area)."""
    return Dependencies(run_context_provider=lambda: {"mode": "headless"})


def combined_text(result: ResolvedContent) -> str:
    """Concatenate the text of all TextBlocks in a ResolvedContent."""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "".join(parts)


def assert_empty_content(result: ResolvedContent) -> None:
    """Assert the resolved content carries no markdown text.

    'Empty' is satisfied either by an empty content list or by content whose
    concatenated TextBlock text is the empty string.
    """
    assert combined_text(result) == "", (
        f"expected empty content, got {combined_text(result)!r}"
    )


def warning_records(caplog) -> list:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# AC 14 / AC 15 — config validation of the on_missing / {area} pairing
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverTemplatedConfigValidation:
    def test_area_path_without_on_missing_raises_at_init(self, tmp_path):
        """AC 14: {area} in path but on_missing omitted -> config error at __init__."""
        templated = str(tmp_path / "{area}" / "CLAUDE.md")
        with pytest.raises(ValueError):
            MarkdownFileResolver(make_area_config(templated))

    def test_area_path_without_on_missing_message_names_resolver_and_path(
        self, tmp_path
    ):
        """AC 14: the config error names the resolver and the offending path."""
        templated = str(tmp_path / "{area}" / "CLAUDE.md")
        with pytest.raises(ValueError) as exc_info:
            MarkdownFileResolver(make_area_config(templated))
        msg = str(exc_info.value)
        assert "MarkdownFileResolver" in msg
        assert templated in msg

    def test_area_path_without_on_missing_via_build_raises(self, tmp_path):
        """AC 14: validation also triggers through build()."""
        templated = str(tmp_path / "{area}" / "CLAUDE.md")
        with pytest.raises(ValueError):
            MarkdownFileResolver.build(
                make_area_config(templated), area_deps("alpha")
            )

    def test_on_missing_without_area_path_raises_at_init(self, tmp_path):
        """AC 15: on_missing supplied on a non-templated path -> config error at __init__."""
        f = tmp_path / "doc.md"
        f.write_text("content")
        with pytest.raises(ValueError):
            MarkdownFileResolver(make_area_config(str(f), on_missing="skip"))

    def test_on_missing_error_without_area_path_raises_at_init(self, tmp_path):
        """AC 15: same for on_missing='error' on a non-templated path."""
        f = tmp_path / "doc.md"
        f.write_text("content")
        with pytest.raises(ValueError):
            MarkdownFileResolver(make_area_config(str(f), on_missing="error"))


# ---------------------------------------------------------------------------
# AC 13 — one instance, two turns, area changes -> different file loaded
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverAreaSwitching:
    @pytest.mark.asyncio
    async def test_area_path_loads_different_file_across_turns(self, tmp_path):
        """AC 13: a single resolver instance loads a different area's file each
        turn as the run context's area changes, with no rebuild and no config edit."""
        base = tmp_path / "projects"
        (base / "alpha").mkdir(parents=True)
        (base / "beta").mkdir(parents=True)
        (base / "alpha" / "CLAUDE.md").write_text("ALPHA CONTENT")
        (base / "beta" / "CLAUDE.md").write_text("BETA CONTENT")

        holder = {"area": "alpha"}
        deps = Dependencies(run_context_provider=lambda: {"area": holder["area"]})
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), deps
        )

        first = await resolver.resolve([make_turn_start_event()])
        assert combined_text(first) == "ALPHA CONTENT"

        holder["area"] = "beta"
        second = await resolver.resolve([make_turn_start_event()])
        assert combined_text(second) == "BETA CONTENT"

        assert combined_text(first) != combined_text(second)

    @pytest.mark.asyncio
    async def test_area_path_resolves_current_area_content(self, tmp_path):
        """AC 13: resolve-time mode reads the file for the provider's current area."""
        base = tmp_path / "projects"
        (base / "gamma").mkdir(parents=True)
        (base / "gamma" / "CLAUDE.md").write_text("GAMMA")
        deps = area_deps("gamma")
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), deps
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert combined_text(result) == "GAMMA"


# ---------------------------------------------------------------------------
# AC 16 — on_missing: skip
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverOnMissingSkip:
    @pytest.mark.asyncio
    async def test_skip_no_match_returns_empty_content(self, tmp_path):
        """AC 16: on_missing='skip' with no matching file returns empty content."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), area_deps("nope")
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert_empty_content(result)

    @pytest.mark.asyncio
    async def test_skip_no_match_does_not_raise(self, tmp_path):
        """AC 16: on_missing='skip' completes the turn (no exception)."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), area_deps("nope")
        )
        # Must not raise.
        await resolver.resolve([make_turn_start_event()])

    @pytest.mark.asyncio
    async def test_skip_no_match_logs_exactly_one_warning(self, tmp_path, caplog):
        """AC 16: on_missing='skip' emits exactly one WARNING log line."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), area_deps("nope")
        )
        with caplog.at_level(logging.WARNING):
            await resolver.resolve([make_turn_start_event()])
        assert len(warning_records(caplog)) == 1


# ---------------------------------------------------------------------------
# AC 17 — on_missing: error
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverOnMissingError:
    @pytest.mark.asyncio
    async def test_error_no_match_raises_from_resolve(self, tmp_path):
        """AC 17: on_missing='error' with no matching file raises from resolve()."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), area_deps("nope")
        )
        with pytest.raises(FileNotFoundError):
            await resolver.resolve([make_turn_start_event()])

    def test_error_missing_file_does_not_raise_at_init(self, tmp_path):
        """AC 17/16: in templated mode a missing file is not an init-time error;
        the on_missing branch is evaluated at resolve() time, so build() succeeds."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        # Should build without raising even though nothing matches yet.
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), area_deps("nope")
        )
        assert resolver is not None


# ---------------------------------------------------------------------------
# AC 18 — no area available -> on_missing branch, never an empty-segment glob
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverNoArea:
    @pytest.mark.asyncio
    async def test_provider_none_skip_returns_empty(self, tmp_path):
        """AC 18: run_context_provider is None -> skip branch, empty content."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), Dependencies()
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert_empty_content(result)

    @pytest.mark.asyncio
    async def test_provider_returns_none_skip_returns_empty(self, tmp_path):
        """AC 18: provider returns None -> skip branch, empty content."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), area_deps(None)
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert_empty_content(result)

    @pytest.mark.asyncio
    async def test_area_key_absent_skip_returns_empty(self, tmp_path):
        """AC 18: 'area' key absent from run context -> skip branch, empty content."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), absent_area_deps()
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert_empty_content(result)

    @pytest.mark.asyncio
    async def test_empty_area_skip_returns_empty(self, tmp_path):
        """AC 18: explicit empty area ('') -> skip branch, empty content."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), area_deps("")
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert_empty_content(result)

    @pytest.mark.asyncio
    async def test_no_area_skip_reports_the_area_not_the_pattern(
        self, tmp_path, caplog
    ):
        """No area is not a templating failure and must not be reported as one.

        This branch used to reuse the missing-file WARNING, so an interface
        that resolves no areas logged
        "no files matched pattern '.../{area}/CLAUDE.md'" on every turn — the
        raw, un-substituted path, which reads like broken templating. It is
        the expected steady state for such an interface, so it belongs at
        DEBUG and must name the missing area instead.
        """
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), absent_area_deps()
        )
        with caplog.at_level(logging.DEBUG):
            result = await resolver.resolve([make_turn_start_event()])

        assert_empty_content(result)
        assert warning_records(caplog) == []
        messages = [r.getMessage() for r in caplog.records]
        assert any("supplies no area" in m for m in messages)
        assert not any("no files matched pattern" in m for m in messages)

    @pytest.mark.asyncio
    async def test_no_area_error_raises_from_resolve(self, tmp_path):
        """AC 18: with on_missing='error', no area takes the same branch as a
        missing file and raises from resolve()."""
        base = tmp_path / "projects"
        base.mkdir()
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), area_deps("")
        )
        with pytest.raises(FileNotFoundError):
            await resolver.resolve([make_turn_start_event()])

    @pytest.mark.asyncio
    async def test_empty_area_does_not_substitute_empty_segment(self, tmp_path):
        """AC 18 (sharp): an empty area must NOT be substituted into the path.

        If the resolver replaced {area} with '', the glob of
        '<base>//CLAUDE.md' would match '<base>/CLAUDE.md' and leak it. The
        resolver must instead take the on_missing branch and return nothing.
        """
        base = tmp_path / "projects"
        base.mkdir()
        # This file sits exactly where an empty substitution would resolve to.
        (base / "CLAUDE.md").write_text("LEAKED EMPTY-SEGMENT CONTENT")
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="skip"), area_deps("")
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert "LEAKED" not in combined_text(result)
        assert_empty_content(result)


# ---------------------------------------------------------------------------
# AC 19 — non-templated path keeps today's init-time behavior (regression)
# ---------------------------------------------------------------------------
#
# The pre-existing tests are the primary AC-19 proof:
#   TestMarkdownFileResolverEmptyGlob::test_empty_glob_raises_at_init
#   TestMarkdownFileResolverEmptyGlob::test_nonexistent_single_file_raises_at_init
#   TestMarkdownFileResolverTokenBudget::test_budget_exceeded_raises_at_init
# The tests below add backward-compat guards specific to the new feature's
# presence (provider wired but path is non-templated).


class TestMarkdownFileResolverNonTemplatedUnchanged:
    @pytest.mark.asyncio
    async def test_non_templated_ignores_run_context_provider(self, tmp_path):
        """AC 19: a non-templated path loads at init and ignores the area provider."""
        f = tmp_path / "doc.md"
        f.write_text("STATIC CONTENT")
        resolver = MarkdownFileResolver.build(
            make_area_config(str(f)), area_deps("some-area")
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert combined_text(result) == "STATIC CONTENT"

    def test_non_templated_empty_glob_still_raises_at_init(self, tmp_path):
        """AC 19: empty glob on a non-templated path still raises at __init__."""
        pattern = str(tmp_path / "*.md")
        with pytest.raises(FileNotFoundError):
            MarkdownFileResolver.build(make_area_config(pattern), area_deps("x"))

    def test_non_templated_budget_still_enforced_at_init(self, tmp_path):
        """AC 19: max_tokens on a non-templated path is still enforced at __init__."""
        f = tmp_path / "big.md"
        f.write_text("x" * 400)  # ~100 tokens
        with pytest.raises(MarkdownTokenBudgetError):
            MarkdownFileResolver.build(
                make_area_config(str(f), max_tokens=10), area_deps("x")
            )


# ---------------------------------------------------------------------------
# AC 20 — max_tokens enforced at resolve() in templated mode
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverTemplatedMaxTokens:
    def test_over_budget_does_not_raise_at_init_in_templated_mode(self, tmp_path):
        """AC 20: in templated mode, an over-budget file is not caught at init;
        enforcement is deferred to resolve()."""
        base = tmp_path / "projects"
        (base / "alpha").mkdir(parents=True)
        (base / "alpha" / "CLAUDE.md").write_text("x" * 400)  # ~100 tokens
        templated = str(base / "{area}" / "CLAUDE.md")
        # build must succeed even though the eventual file exceeds the budget.
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error", max_tokens=10),
            area_deps("alpha"),
        )
        assert resolver is not None

    @pytest.mark.asyncio
    async def test_over_budget_raises_at_resolve_in_templated_mode(self, tmp_path):
        """AC 20: max_tokens is enforced at resolve() against the file loaded that turn."""
        base = tmp_path / "projects"
        (base / "alpha").mkdir(parents=True)
        (base / "alpha" / "CLAUDE.md").write_text("x" * 400)  # ~100 tokens
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error", max_tokens=10),
            area_deps("alpha"),
        )
        with pytest.raises(MarkdownTokenBudgetError):
            await resolver.resolve([make_turn_start_event()])

    @pytest.mark.asyncio
    async def test_within_budget_resolves_in_templated_mode(self, tmp_path):
        """AC 20: a file within budget resolves normally in templated mode."""
        base = tmp_path / "projects"
        (base / "alpha").mkdir(parents=True)
        (base / "alpha" / "CLAUDE.md").write_text("small")  # ~1 token
        templated = str(base / "{area}" / "CLAUDE.md")
        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error", max_tokens=1000),
            area_deps("alpha"),
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert combined_text(result) == "small"


# ---------------------------------------------------------------------------
# AC 21 — per-path/mtime caching: unchanged file is re-stat'd but not re-read
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverCaching:
    @pytest.mark.asyncio
    async def test_unchanged_file_read_once_across_two_resolves(
        self, tmp_path, monkeypatch
    ):
        """AC 21: two consecutive turns on an unchanged file read it only once."""
        base = tmp_path / "projects"
        (base / "alpha").mkdir(parents=True)
        target = base / "alpha" / "CLAUDE.md"
        target.write_text("CACHED CONTENT")
        templated = str(base / "{area}" / "CLAUDE.md")

        reads: list[str] = []
        orig_read_text = pathlib.Path.read_text

        def counting_read_text(self, *args, **kwargs):
            reads.append(str(self))
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)

        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), area_deps("alpha")
        )

        first = await resolver.resolve([make_turn_start_event()])
        second = await resolver.resolve([make_turn_start_event()])

        target_reads = [p for p in reads if p.endswith("CLAUDE.md")]
        assert combined_text(first) == "CACHED CONTENT"
        assert combined_text(second) == "CACHED CONTENT"
        assert len(target_reads) == 1, (
            f"expected the file to be read once across two resolves, "
            f"got {len(target_reads)} reads: {target_reads}"
        )

    @pytest.mark.asyncio
    async def test_changed_mtime_triggers_reread(self, tmp_path, monkeypatch):
        """AC 21: a change to the file's mtime invalidates the cache and re-reads."""
        base = tmp_path / "projects"
        (base / "alpha").mkdir(parents=True)
        target = base / "alpha" / "CLAUDE.md"
        target.write_text("VERSION ONE")
        templated = str(base / "{area}" / "CLAUDE.md")

        reads: list[str] = []
        orig_read_text = pathlib.Path.read_text

        def counting_read_text(self, *args, **kwargs):
            reads.append(str(self))
            return orig_read_text(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)

        resolver = MarkdownFileResolver.build(
            make_area_config(templated, on_missing="error"), area_deps("alpha")
        )

        await resolver.resolve([make_turn_start_event()])

        # Change content and bump mtime clearly into the future.
        target.write_text("VERSION TWO")
        future = os.stat(target).st_mtime + 1000
        os.utime(target, (future, future))

        second = await resolver.resolve([make_turn_start_event()])

        target_reads = [p for p in reads if p.endswith("CLAUDE.md")]
        assert combined_text(second) == "VERSION TWO"
        assert len(target_reads) == 2, (
            f"expected a re-read after mtime change, got {len(target_reads)} reads"
        )


# ---------------------------------------------------------------------------
# AC 22 — docstring no longer claims ${VAR} interpolation of `path`
# ---------------------------------------------------------------------------


class TestMarkdownFileResolverDocstring:
    def test_docstring_does_not_claim_var_interpolation(self):
        """AC 22: the resolver docstring must not claim ${VAR} interpolation of path."""
        doc = MarkdownFileResolver.__doc__ or ""
        assert "${VAR}" not in doc, (
            "docstring still claims ${VAR} interpolation of `path`, which is false"
        )
