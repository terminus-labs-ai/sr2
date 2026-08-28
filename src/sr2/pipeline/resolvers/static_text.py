"""StaticTextResolver: emits literal config text with run-context interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)

_AREA_PLACEHOLDER = "{area}"


class StaticTextResolver:
    """Emits literal text from config into a layer, interpolating run-context placeholders.

    Config fields
    -------------
    text : str
        Literal text to emit. May contain the literal placeholder ``{area}``,
        which is substituted with the current run context's area at resolve
        time (read from ``deps.run_context_provider()["area"]``). Text with
        no placeholders is emitted verbatim.

    The resolver is intentionally dumb: it only produces text. A missing or
    empty area substitutes to the empty string — suppressing the whole
    layer when there is no area is a layer-condition concern, not a
    resolver concern.
    """

    name: str = "static_text"

    def __init__(
        self,
        config: ResolverConfig,
        run_context_provider: "Callable[[], dict[str, str] | None] | None" = None,
    ) -> None:
        if "text" not in config.config:
            raise ValueError("StaticTextResolver requires config['text'] to be set.")

        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, [_DEFAULT_SUBSCRIPTION]
        )
        self._text: str = config.config["text"]
        self._run_context_provider = run_context_provider

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "StaticTextResolver":
        provider = deps.run_context_provider if deps is not None else None
        return cls(config, run_context_provider=provider)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        text = self._text
        if _AREA_PLACEHOLDER in text:
            text = text.replace(_AREA_PLACEHOLDER, self._current_area())
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="static_text",
            content=[TextBlock(text=text)],
        )

    def _current_area(self) -> str:
        """Return the current area as a string (empty string when unavailable)."""
        if self._run_context_provider is None:
            return ""
        ctx = self._run_context_provider()
        if not ctx:
            return ""
        area = ctx.get("area")
        return area if isinstance(area, str) else ""
