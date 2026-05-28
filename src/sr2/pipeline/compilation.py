"""Position strategies and compilation target strategies for the SR2 pipeline.

FR15: PositionStrategy protocol with PrefixStrategy and AppendStrategy built-ins.
FR20: Protocol-based — new position strategies addable without modifying engine code.
FR39: TargetCompiler protocol with per-target strategy registry — adding a new
      CompilationTarget only requires adding a new entry to _COMPILATION_TARGETS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2.models import Message, TextBlock, ToolDefinition
    from sr2.pipeline.models import CompilationTarget


@runtime_checkable
class PositionStrategy(Protocol):
    """Determines how new content is placed relative to existing content."""

    def place(self, existing: list, new: list) -> list: ...


class AppendStrategy:
    """Appends new items after existing items."""

    def place(self, existing: list, new: list) -> list:
        return existing + new


class PrefixStrategy:
    """Prepends new items before existing items."""

    def place(self, existing: list, new: list) -> list:
        return new + existing


# ---------------------------------------------------------------------------
# Compilation target strategies
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetCompiler(Protocol):
    """Appends a compiled layer's output into the appropriate accumulator lists.

    Each CompilationTarget maps to one TargetCompiler. The compiler receives
    the pre-compiled content and the three accumulator lists; it mutates the
    appropriate list in place.

    Implementing a new target: create a class with this shape and add it to
    ``_COMPILATION_TARGETS`` — no other code needs to change.
    """

    def collect(
        self,
        compiled: list,
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None: ...


class _SystemCollector:
    def collect(
        self,
        compiled: list,
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        system_blocks.extend(compiled)


class _MessagesCollector:
    def collect(
        self,
        compiled: list,
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        messages.extend(compiled)


class _ToolsCollector:
    def collect(
        self,
        compiled: list,
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        tools.extend(compiled)


def _build_compilation_targets() -> dict[CompilationTarget, TargetCompiler]:
    """Build the canonical target -> compiler mapping.

    Deferred import avoids a circular dependency: compilation.py is imported by
    layer.py and models.py; importing models.py at module level would create a
    cycle.  This function is called once at first use via the cached module-level
    constant ``COMPILATION_TARGETS``.
    """
    from sr2.pipeline.models import CompilationTarget  # local to break cycle

    return {
        CompilationTarget.SYSTEM: _SystemCollector(),
        CompilationTarget.MESSAGES: _MessagesCollector(),
        CompilationTarget.TOOLS: _ToolsCollector(),
    }


# Populated on first access via get_compilation_targets()
_compilation_targets: dict[CompilationTarget, TargetCompiler] | None = None


def get_compilation_targets() -> dict[CompilationTarget, TargetCompiler]:
    """Return the singleton target-compiler registry (initialised lazily)."""
    global _compilation_targets
    if _compilation_targets is None:
        _compilation_targets = _build_compilation_targets()
    return _compilation_targets
