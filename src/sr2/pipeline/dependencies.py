"""Pipeline dependency container."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from sr2.protocols.llm import LLMCallable

if TYPE_CHECKING:
    from sr2.memory.protocol import MemoryExtractor, MemoryStore
    from sr2.pipeline.protocols import ToolSource


@dataclasses.dataclass(frozen=True)
class Dependencies:
    """Immutable container for runtime dependencies injected into pipeline components."""

    llm: dict[str, LLMCallable] | None = None
    memory_store: "MemoryStore | None" = None
    memory_extractor: "MemoryExtractor | None" = None
    tool_source: "ToolSource | None" = None
    """Harness-provided tool definition source.

    A ToolProvider reads this to surface the harness's tools into the
    pipeline (see ``sr2.pipeline.protocols.ToolSource``).  When ``None``
    (default), no harness tools are injected.  Replaces the removed untyped
    ``extras`` service-locator bag with a typed optional dependency.
    """
    session_id: str = ""
    active_frame_provider: Callable[[str], str | None] | None = None
    """Origin-aware active-frame provider.

    When present, the orchestrator calls ``provider(origin)`` to resolve the
    active frame for the current turn's *origin*.  The provider returns the
    work-frame id if one is open on that origin, or the ambient frame id
    bound to the origin.  When ``None`` (default), no stamping occurs and
    core behaviour is unchanged (regression-safe).

    The *origin* parameter is typically a transport-identifier string
    (e.g. ``"tui"``, ``"discord:channel_id"``).
    """
    run_context_provider: Callable[[], dict[str, str] | None] | None = None
    """Run-context provider for pipeline components.

    When present, a resolver/transformer calls ``provider()`` to retrieve a
    dict of run-context strings (e.g. ``{"mode": "headless", "source": "discord"}``).
    The harness supplies the concrete provider; core owns only the generic
    callable contract.  When ``None`` (default), no run-context is available
    and component behaviour is unchanged (regression-safe).

    Mirrors ``active_frame_provider``: generic callable, no harness-type
    import in core.  Spectre's ``RunContext`` is a spectre type — core
    MUST NOT import it.
    """
