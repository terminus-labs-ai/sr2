"""MarkdownFileResolver: loads markdown files from a path or glob pattern."""

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import TYPE_CHECKING

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.token_counting import CHARS_PER_TOKEN, CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    pass

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)


class MarkdownTokenBudgetError(Exception):
    """Raised at resolver init when loaded markdown files exceed max_tokens."""


class MarkdownFileResolver:
    """Loads one or more markdown files at init; returns concatenated content at resolve time.

    Config fields
    -------------
    path : str
        Absolute path or glob pattern. ${VAR} interpolation and file-relative
        resolution are applied by the config loader before this resolver sees
        the value. Glob expansion and file loading are handled here.
    max_tokens : int | None
        Optional token budget (CharacterTokenCounter approximation: chars // 4).
        Enforced at init — fails fast so misconfigured pipelines surface errors
        on startup, not mid-conversation.
    declaring_dir : str | None
        Directory of the declaring config file. Used to resolve relative glob
        patterns that were not expanded by the config loader. Ignored when
        ``path`` is absolute.

    Empty glob
    ----------
    Raises ``FileNotFoundError`` at init if no files match the pattern (or the
    explicit path does not exist). Resolvers must have something to return.
    """

    name: str = "markdown_file"

    def __init__(self, config: ResolverConfig) -> None:
        if "path" not in config.config:
            raise ValueError("MarkdownFileResolver requires config['path'] to be set.")

        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )

        raw_path: str = config.config["path"]
        max_tokens: int | None = config.config.get("max_tokens")
        declaring_dir: str | None = config.config.get("declaring_dir")

        # Resolve the path.  If the raw_path is not absolute and a
        # declaring_dir is provided, prepend it so relative globs work.
        resolved_path = self._resolve_path(raw_path, declaring_dir)

        # Expand glob and load files.
        self._files: list[Path] = self._expand_and_sort(resolved_path)

        if not self._files:
            raise FileNotFoundError(
                f"MarkdownFileResolver: no files matched pattern {resolved_path!r}"
            )

        # Read all file contents now (at init) so token counting can happen.
        self._contents: list[tuple[Path, str]] = [
            (p, p.read_text(encoding="utf-8")) for p in self._files
        ]

        # Enforce token budget if set.
        if max_tokens is not None:
            self._check_budget(self._contents, max_tokens)

        # Pre-build the concatenated text.
        self._combined: str = "\n".join(content for _, content in self._contents)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "MarkdownFileResolver":
        return cls(config)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="markdown_file",
            content=[TextBlock(text=self._combined)],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(raw_path: str, declaring_dir: str | None) -> str:
        """Return the effective path string to use for glob expansion.

        If raw_path is absolute (or an absolute glob), return it unchanged.
        Otherwise, if declaring_dir is provided, prepend it to form a full path.
        """
        p = Path(raw_path)
        if p.is_absolute():
            return raw_path
        if declaring_dir is not None:
            return str(Path(declaring_dir) / raw_path)
        return raw_path

    @staticmethod
    def _expand_and_sort(path_or_pattern: str) -> list[Path]:
        """Expand a glob pattern (or exact path) and return sorted absolute Paths."""
        matches = _glob.glob(path_or_pattern, recursive=True)
        return sorted(Path(m).resolve() for m in matches)

    @staticmethod
    def _check_budget(
        contents: list[tuple[Path, str]], max_tokens: int
    ) -> None:
        """Raise MarkdownTokenBudgetError if total tokens exceed max_tokens."""
        counter = CharacterTokenCounter()
        per_file: list[tuple[Path, int]] = []
        total = 0
        for path, text in contents:
            tokens = len(text) // CHARS_PER_TOKEN
            per_file.append((path, tokens))
            total += tokens

        if total <= max_tokens:
            return

        lines = ["MarkdownFileResolver: token budget exceeded."]
        lines.append("")
        for path, tokens in per_file:
            lines.append(f"  {path}: {tokens} tokens")
        lines.append("")
        lines.append(f"Total: {total} tokens")
        lines.append(f"Budget: {max_tokens} tokens")
        lines.append("")
        lines.append(
            "Hint: set max_tokens: null in your config to disable budget enforcement."
        )
        raise MarkdownTokenBudgetError("\n".join(lines))
