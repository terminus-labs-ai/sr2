"""Position strategies and compilation target strategies for the SR2 pipeline.

FR15: PositionStrategy protocol with PrefixStrategy and AppendStrategy built-ins.
FR20: Protocol-based — new position strategies addable without modifying engine code.
FR39: TargetCompiler protocol with per-target strategy registry — adding a new
      CompilationTarget only requires adding a new entry to _COMPILATION_TARGETS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sr2.models import ContentBlock, Message, TextBlock, ToolDefinition
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
    """Compiles raw layer content into the appropriate accumulator lists.

    Each CompilationTarget maps to one TargetCompiler. The compiler receives
    the raw content blocks and tool definitions; it performs type narrowing
    (e.g. filtering to TextBlock for SYSTEM, grouping into Message for
    MESSAGES) and mutates the appropriate accumulator list in place.

    Layer.compile() dispatches through this registry — there is no separate
    _COMPILE_DISPATCH table. Implementing a new target: create a class with
    this shape and add it to ``_COMPILATION_TARGETS``.
    """

    def collect(
        self,
        content: list[ContentBlock | Message],
        tool_definitions: list[ToolDefinition],
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None: ...


class _SystemCollector:
    """Compile SYSTEM target: filter content to TextBlocks."""

    def collect(
        self,
        content: list[ContentBlock | Message],
        tool_definitions: list[ToolDefinition],
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        from sr2.models import TextBlock

        out: list[TextBlock] = []
        for block in content:
            if isinstance(block, TextBlock):
                out.append(block)
        system_blocks.extend(out)


class _MessagesCollector:
    """Compile MESSAGES target: group raw ContentBlocks into Messages."""

    def collect(
        self,
        content: list[ContentBlock | Message],
        tool_definitions: list[ToolDefinition],
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        from sr2.models import ContentBlock, Message

        if not content:
            return

        msgs: list[Message] = []
        raw_blocks: list[ContentBlock] = []

        for item in content:
            if isinstance(item, Message):
                # Flush any accumulated raw blocks first
                if raw_blocks:
                    msgs.append(Message(role="user", content=raw_blocks))
                    raw_blocks = []
                msgs.append(item)
            else:
                raw_blocks.append(item)

        # Flush trailing raw blocks
        if raw_blocks:
            msgs.append(Message(role="user", content=raw_blocks))

        messages.extend(msgs)


class _ToolsCollector:
    """Compile TOOLS target: return tool definitions as-is."""

    def collect(
        self,
        content: list[ContentBlock | Message],
        tool_definitions: list[ToolDefinition],
        system_blocks: list[TextBlock],
        messages: list[Message],
        tools: list[ToolDefinition],
    ) -> None:
        tools.extend(tool_definitions)


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
