"""Tests for sr2-117 — clock placeholders in StaticTextResolver (AC1).

Covers spec FR1–FR3 (sr2-spectre/agent-reminders):
  FR1: optional ``timezone`` config key holding an IANA name; an invalid
       name raises ValueError when the resolver is built.
  FR2: with ``timezone`` set, ``{date}`` renders the ISO date in that zone;
       without it ``{date}`` behaves exactly as before (host local).
  FR3: ``{time}`` renders ``HH:MM <abbr> (UTC±hh:mm)`` and ``{weekday}``
       renders the English weekday name, both in ``timezone`` when set,
       else host local.

Time is injected through the module-level patchable ``date`` namespace
(``date.now`` seam) — no real clock is read in any assertion.
"""

from __future__ import annotations

import datetime
import re
from zoneinfo import ZoneInfo

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.resolvers import static_text as static_text_module
from sr2.pipeline.resolvers.static_text import StaticTextResolver


@pytest.fixture
def frozen_now(monkeypatch):
    """Patch the clock seam so ``date.now(tz)`` returns a fixed instant in *tz*."""

    def _freeze(dt: datetime.datetime):
        monkeypatch.setattr(
            static_text_module.date, "now", lambda tz=None: dt.astimezone(tz)
        )

    return _freeze


def make_config(text: str, timezone: str | None = None) -> ResolverConfig:
    config: dict = {"text": text}
    if timezone is not None:
        config["timezone"] = timezone
    return ResolverConfig(type="static_text", config=config)


def turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


async def resolve_text(resolver: StaticTextResolver) -> str:
    result = await resolver.resolve([turn_start_event()])
    assert isinstance(result.content[0], TextBlock)
    return result.content[0].text


# ---------------------------------------------------------------------------
# FR1 — timezone key validation at build time
# ---------------------------------------------------------------------------


class TestTimezoneBuildValidation:
    def test_valid_iana_timezone_accepted_at_build(self):
        resolver = StaticTextResolver(make_config("x", timezone="America/Phoenix"))
        assert resolver is not None

    def test_invalid_timezone_raises_value_error_at_build(self):
        with pytest.raises(ValueError, match="[Tt]imezone"):
            StaticTextResolver(make_config("x", timezone="Mars/Olympus_Mons"))

    def test_invalid_timezone_raises_through_build_classmethod(self):
        # The orchestrator builds resolvers via .build(config, deps).
        with pytest.raises(ValueError, match="[Tt]imezone"):
            StaticTextResolver.build(make_config("x", timezone="Not/AZone"), None)

    def test_timezone_absent_is_valid(self):
        resolver = StaticTextResolver(make_config("x"))
        assert resolver is not None


# ---------------------------------------------------------------------------
# FR3 — {time} format
# ---------------------------------------------------------------------------


class TestTimePlaceholder:
    @pytest.mark.asyncio
    async def test_time_renders_hhmm_abbr_offset(self, frozen_now):
        frozen_now(datetime.datetime(2026, 7, 22, 23, 40, tzinfo=ZoneInfo("America/Phoenix")))
        resolver = StaticTextResolver(make_config("Now: {time}", timezone="America/Phoenix"))

        text = await resolve_text(resolver)

        assert text == "Now: 23:40 MST (UTC-07:00)"

    @pytest.mark.asyncio
    async def test_time_offset_has_colon_and_sign(self, frozen_now):
        frozen_now(datetime.datetime(2026, 1, 22, 23, 40, tzinfo=ZoneInfo("America/New_York")))
        resolver = StaticTextResolver(
            make_config("{time}", timezone="America/New_York")
        )

        text = await resolve_text(resolver)

        assert text == "23:40 EST (UTC-05:00)"

    @pytest.mark.asyncio
    async def test_time_half_hour_zone_formats_minutes(self, frozen_now):
        frozen_now(datetime.datetime(2026, 7, 22, 12, 15, tzinfo=ZoneInfo("Asia/Kolkata")))
        resolver = StaticTextResolver(make_config("{time}", timezone="Asia/Kolkata"))

        text = await resolve_text(resolver)

        assert text == "12:15 IST (UTC+05:30)"

    @pytest.mark.asyncio
    async def test_time_substitutes_all_occurrences(self, frozen_now):
        frozen_now(datetime.datetime(2026, 7, 22, 16, 40, tzinfo=ZoneInfo("America/Phoenix")))
        resolver = StaticTextResolver(
            make_config("{time} and again {time}", timezone="America/Phoenix")
        )

        text = await resolve_text(resolver)

        assert text == "16:40 MST (UTC-07:00) and again 16:40 MST (UTC-07:00)"
        assert "{time}" not in text

    @pytest.mark.asyncio
    async def test_time_without_timezone_uses_host_local(self):
        resolver = StaticTextResolver(make_config("{time}"))

        text = await resolve_text(resolver)

        # Host local — assert shape, not value: HH:MM <abbr> (UTC±hh:mm)
        assert re.fullmatch(r"\d{2}:\d{2} \S+ \(UTC[+-]\d{2}:\d{2}\)", text), text


# ---------------------------------------------------------------------------
# FR3 — {weekday}
# ---------------------------------------------------------------------------


class TestWeekdayPlaceholder:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "dt, expected",
        [
            (datetime.datetime(2026, 7, 20, 9, 0, tzinfo=ZoneInfo("UTC")), "Monday"),
            (datetime.datetime(2026, 7, 22, 9, 0, tzinfo=ZoneInfo("UTC")), "Wednesday"),
            (datetime.datetime(2026, 7, 26, 9, 0, tzinfo=ZoneInfo("UTC")), "Sunday"),
        ],
    )
    async def test_weekday_english_name(self, frozen_now, dt, expected):
        frozen_now(dt)
        resolver = StaticTextResolver(make_config("Today is {weekday}.", timezone="UTC"))

        text = await resolve_text(resolver)

        assert text == f"Today is {expected}."

    @pytest.mark.asyncio
    async def test_weekday_without_timezone_works(self):
        resolver = StaticTextResolver(make_config("{weekday}"))

        text = await resolve_text(resolver)

        assert text == datetime.date.today().strftime("%A")


# ---------------------------------------------------------------------------
# FR2 — {date} with and without timezone
# ---------------------------------------------------------------------------


class TestDateTimezoneInteraction:
    @pytest.mark.asyncio
    async def test_date_uses_configured_zone(self, frozen_now):
        # 00:40 on the 26th in Phoenix, but 07:40 UTC on the 26th — pick an
        # instant where host-local date differs from the zone date by using
        # a zone boundary: 23:40 Phoenix == next-day 06:40 UTC.
        utc_instant = datetime.datetime(2026, 7, 23, 6, 40, tzinfo=ZoneInfo("UTC"))
        frozen_now(utc_instant)

        phoenix = StaticTextResolver(make_config("{date}", timezone="America/Phoenix"))
        utc = StaticTextResolver(make_config("{date}", timezone="UTC"))

        assert await resolve_text(phoenix) == "2026-07-22"
        assert await resolve_text(utc) == "2026-07-23"

    @pytest.mark.asyncio
    async def test_date_without_timezone_unchanged(self, monkeypatch):
        # Same patch seam the pre-clock tests use: date.today.
        fixed = datetime.date(2026, 10, 8)
        monkeypatch.setattr(static_text_module.date, "today", lambda: fixed)
        resolver = StaticTextResolver(make_config("Hoje é {date}."))

        text = await resolve_text(resolver)

        assert text == "Hoje é 2026-10-08."

    @pytest.mark.asyncio
    async def test_all_three_placeholders_together(self, frozen_now):
        frozen_now(
            datetime.datetime(2026, 7, 22, 16, 40, tzinfo=ZoneInfo("America/Phoenix"))
        )
        resolver = StaticTextResolver(
            make_config("Now: {weekday} {date} {time}", timezone="America/Phoenix")
        )

        text = await resolve_text(resolver)

        assert text == "Now: Wednesday 2026-07-22 16:40 MST (UTC-07:00)"

    @pytest.mark.asyncio
    async def test_area_placeholder_still_works_with_clock(self, frozen_now):
        frozen_now(
            datetime.datetime(2026, 7, 22, 16, 40, tzinfo=ZoneInfo("America/Phoenix"))
        )
        resolver = StaticTextResolver(
            make_config("{area} @ {time}", timezone="America/Phoenix"),
            run_context_provider=lambda: {"area": "sr2-spectre"},
        )

        text = await resolve_text(resolver)

        assert text == "sr2-spectre @ 16:40 MST (UTC-07:00)"
