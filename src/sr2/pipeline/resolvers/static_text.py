"""StaticTextResolver: emits config text with run-context interpolation."""

from __future__ import annotations

import datetime as _datetime
import types as _types
from typing import TYPE_CHECKING

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_date_namespace() -> _types.ModuleType:
    """Mutable ``date`` namespace so tests can patch ``date.today`` here.

    ``datetime.date`` is an immutable C type, so monkeypatching
    ``datetime.date.today`` directly is impossible.  This module exposes a
    plain module object under the name ``date`` whose ``today`` attribute can
    be set and restored.  A module object (not the class) is required because
    pytest's monkeypatch refuses to set attributes on any ``type``.
    """
    namespace = _types.ModuleType("date")
    namespace.today = _datetime.date.today  # type: ignore[attr-defined]
    return namespace


date = _make_date_namespace()

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)
_AREA_PLACEHOLDER = "{area}"
_DATE_PLACEHOLDER = "{date}"


class StaticTextResolver:
    """Emits config text, replacing ``{area}`` from the current run context."""

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
        self._run_context_provider = run_context_provider

    @classmethod
    def build(cls, config: ResolverConfig, deps: "Dependencies") -> "StaticTextResolver":
        provider = deps.run_context_provider if deps is not None else None
        return cls(config, run_context_provider=provider)

    async def resolve(self, events: list[Event]) -> ResolvedContent:
        self.execution_count += 1
        text = self._config.config["text"]
        if _AREA_PLACEHOLDER in text:
            text = text.replace(_AREA_PLACEHOLDER, self._current_area())
        if _DATE_PLACEHOLDER in text:
            text = text.replace(_DATE_PLACEHOLDER, self._today_iso())
        return ResolvedContent(
            resolver_name=self.name,
            source_layer="static_text",
            content=[TextBlock(text=text)],
        )

    def _current_area(self) -> str:
        """Return the current area, or an empty string when unavailable."""
        if self._run_context_provider is None:
            return ""
        context = self._run_context_provider()
        if not context:
            return ""
        area = context.get("area")
        return area if isinstance(area, str) else ""

    @staticmethod
    def _today_iso() -> str:
        """Return the local host date as an ISO-8601 string (YYYY-MM-DD)."""
        return date.today().isoformat()
