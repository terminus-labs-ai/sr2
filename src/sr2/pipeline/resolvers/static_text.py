"""StaticTextResolver: emits config text with run-context interpolation."""

from __future__ import annotations

import datetime as _datetime
import types as _types
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_date_namespace() -> _types.ModuleType:
    """Mutable ``datetime`` namespace so tests can patch ``datetime.now`` here.

    ``datetime.datetime`` is an immutable C type, so monkeypatching
    ``datetime.datetime.now`` directly is impossible.  This module exposes a
    plain module object under the name ``date`` (historical name; kept for
    backward compatibility with existing patches of ``date.today``) whose
    attributes can be set and restored.  A module object (not the class) is
    required because pytest's monkeypatch refuses to set attributes on any
    ``type``.
    """
    namespace = _types.ModuleType("date")
    namespace.today = _datetime.date.today  # type: ignore[attr-defined]
    namespace.now = _datetime.datetime.now  # type: ignore[attr-defined]
    return namespace


date = _make_date_namespace()

_DEFAULT_SUBSCRIPTION = EventSubscription(event_name="turn_start", phase=EventPhase.STARTING)
_AREA_PLACEHOLDER = "{area}"
_DATE_PLACEHOLDER = "{date}"
_TIME_PLACEHOLDER = "{time}"
_WEEKDAY_PLACEHOLDER = "{weekday}"
_ENGLISH_WEEKDAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
)


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

        # FR1 (sr2-spectre clock): optional IANA timezone.  An invalid name
        # fails at build time, not at resolve time.
        timezone_name = config.config.get("timezone")
        self._timezone: ZoneInfo | None = None
        if timezone_name is not None:
            try:
                self._timezone = ZoneInfo(str(timezone_name))
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"StaticTextResolver: invalid IANA timezone {timezone_name!r}."
                ) from exc

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
        if _TIME_PLACEHOLDER in text or _WEEKDAY_PLACEHOLDER in text:
            now = self._now_local()
            if _TIME_PLACEHOLDER in text:
                text = text.replace(_TIME_PLACEHOLDER, self._format_time(now))
            if _WEEKDAY_PLACEHOLDER in text:
                text = text.replace(_WEEKDAY_PLACEHOLDER, _ENGLISH_WEEKDAYS[now.weekday()])
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

    def _today_iso(self) -> str:
        """Return today's date as an ISO-8601 string (YYYY-MM-DD).

        In the configured zone when ``timezone`` is set.  Without a zone this
        delegates to the patchable ``date.today`` seam, so ``{date}`` behaves
        exactly as before (host local).
        """
        if self._timezone is None:
            return date.today().isoformat()
        return date.now(self._timezone).date().isoformat()

    def _now_local(self) -> _datetime.datetime:
        """Return the current wall-clock time in the configured zone.

        ``date.now`` is the patchable seam (see ``_make_date_namespace``).
        """
        now = date.now(self._timezone)
        return now.astimezone() if self._timezone is None else now

    @staticmethod
    def _format_time(now: _datetime.datetime) -> str:
        """Format *now* as ``HH:MM <abbr> (UTC±hh:mm)``, e.g. ``16:40 MST (UTC-07:00)``."""
        offset = now.utcoffset() or _datetime.timedelta(0)
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        abs_minutes = abs(total_minutes)
        return (
            f"{now.strftime('%H:%M')} {now.tzname()} "
            f"(UTC{sign}{abs_minutes // 60:02d}:{abs_minutes % 60:02d})"
        )
