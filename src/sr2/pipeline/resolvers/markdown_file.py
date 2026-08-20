"""MarkdownFileResolver: loads markdown files from a path or glob pattern."""

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
from sr2.pipeline.token_counting import CHARS_PER_TOKEN, CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)

_AREA_PLACEHOLDER = "{area}"


class MarkdownTokenBudgetError(Exception):
    """Raised when loaded markdown files exceed max_tokens."""


class MarkdownFileResolver:
    """Loads one or more markdown files; returns concatenated content at resolve time.

    Config fields
    -------------
    path : str
        Absolute path or glob pattern. May contain the literal placeholder
        ``{area}``, which is substituted with the current run context's area
        at resolve time (see "Templated mode" below). Relative patterns are
        resolved against ``declaring_dir``; ``{area}`` is the only supported
        placeholder — no other environment-style interpolation of ``path``
        happens here.
    on_missing : "skip" | "error" | None
        Required if and only if ``path`` contains ``{area}``. "skip" makes
        an unmatched path resolve to empty content and log one WARNING;
        "error" raises ``FileNotFoundError`` from ``resolve()``. Supplying
        it on a path without ``{area}``, or omitting it on a path with
        ``{area}``, is a configuration error raised at ``__init__``.
    max_tokens : int | None
        Optional token budget (CharacterTokenCounter approximation: chars // 4).
        For a static path this is enforced at ``__init__`` (fails fast). For
        a templated path the files are unknown until resolve time, so it is
        enforced on every ``resolve()`` call instead.
    declaring_dir : str | None
        Directory of the declaring config file. Used to resolve relative glob
        patterns that were not expanded by the config loader. Ignored when
        ``path`` is absolute.

    Static mode (no ``{area}`` in path)
    ------------------------------------
    Unchanged from before: the glob is expanded and files are read once at
    ``__init__``; an empty glob raises ``FileNotFoundError`` at ``__init__``,
    and ``max_tokens`` is enforced at ``__init__``.

    Templated mode (``{area}`` in path)
    ------------------------------------
    The path is substituted, globbed and read on every ``resolve()`` call
    against the run context's current area, read from
    ``deps.run_context_provider()["area"]``. No area available — the
    provider is ``None``, returns ``None``, or the returned dict lacks the
    key or maps it to ``""`` — takes the same branch as a missing file and
    never substitutes an empty string into ``path``. File contents are
    cached per resolved path, keyed on path and mtime, so an unchanged file
    is re-``stat``'d every turn but not re-read.
    """

    name: str = "markdown_file"

    def __init__(
        self,
        config: ResolverConfig,
        run_context_provider: "Callable[[], dict[str, str] | None] | None" = None,
    ) -> None:
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
        on_missing: str | None = config.config.get("on_missing")

        self._raw_path = raw_path
        self._declaring_dir = declaring_dir
        self._max_tokens = max_tokens
        self._run_context_provider = run_context_provider
        self._templated: bool = _AREA_PLACEHOLDER in raw_path
        self._file_cache: dict[Path, tuple[float, str]] = {}

        if self._templated:
            if on_missing not in ("skip", "error"):
                raise ValueError(
                    "MarkdownFileResolver: on_missing ('skip' or 'error') is "
                    f"required when path contains {{area}}; path={raw_path!r}."
                )
            self._on_missing = on_missing
            # Resolve-time mode: no glob/read/budget check happens here.
            return

        if on_missing is not None:
            raise ValueError(
                "MarkdownFileResolver: on_missing is only valid when path "
                f"contains {{area}}; path={raw_path!r} does not."
            )
        self._on_missing = None

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
        provider = deps.run_context_provider if deps is not None else None
        return cls(config, run_context_provider=provider)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        text = self._resolve_templated() if self._templated else self._combined
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="markdown_file",
            content=[TextBlock(text=text)],
        )

    # ------------------------------------------------------------------
    # Internal helpers — templated (resolve-time) mode
    # ------------------------------------------------------------------

    def _current_area(self) -> str | None:
        """Return the current non-empty area, or None if none is available."""
        if self._run_context_provider is None:
            return None
        ctx = self._run_context_provider()
        if not ctx:
            return None
        area = ctx.get("area")
        return area or None

    def _handle_missing(self, pattern: str) -> str:
        """Apply the on_missing branch: raise (error) or warn+empty (skip)."""
        if self._on_missing == "error":
            raise FileNotFoundError(
                f"MarkdownFileResolver: no files matched pattern {pattern!r}"
            )
        logger.warning(
            "MarkdownFileResolver: no files matched pattern %r; returning empty content",
            pattern,
        )
        return ""

    def _resolve_templated(self) -> str:
        area = self._current_area()
        if area is None:
            return self._handle_missing(self._raw_path)

        substituted = self._raw_path.replace(_AREA_PLACEHOLDER, area)
        resolved_path = self._resolve_path(substituted, self._declaring_dir)
        files = self._expand_and_sort(resolved_path)
        if not files:
            return self._handle_missing(resolved_path)

        contents = [(p, self._read_cached(p)) for p in files]
        if self._max_tokens is not None:
            self._check_budget(contents, self._max_tokens)
        return "\n".join(content for _, content in contents)

    def _read_cached(self, path: Path) -> str:
        """Read path's text, using a path+mtime cache to avoid re-reading."""
        mtime = path.stat().st_mtime
        cached = self._file_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        text = path.read_text(encoding="utf-8")
        self._file_cache[path] = (mtime, text)
        return text

    # ------------------------------------------------------------------
    # Internal helpers — shared
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
