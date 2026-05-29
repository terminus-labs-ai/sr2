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
