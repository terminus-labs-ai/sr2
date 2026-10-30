"""FR3: English weekday names through the date seam, independent of locale."""

import datetime
from zoneinfo import ZoneInfo

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.resolvers import static_text


class LocalizedDatetime(datetime.datetime):
    def strftime(self, fmt):
        return "jour-localisé" if fmt == "%A" else super().strftime(fmt)

    def astimezone(self, tz=None):
        # The supplied wall clock is the controlled host-local value.
        return self if tz is None else super().astimezone(tz)


@pytest.fixture
def frozen_localized_now(monkeypatch):
    def freeze(instant):
        calls = []

        def now(tz=None):
            calls.append(tz)
            return instant.astimezone(tz)

        monkeypatch.setattr(static_text.date, "now", now)
        return calls

    return freeze


async def resolve_text(timezone=None):
    config = {"text": "Today is {weekday}; again {weekday}."}
    if timezone is not None:
        config["timezone"] = timezone
    resolver = static_text.StaticTextResolver(
        ResolverConfig(type="static_text", config=config)
    )
    result = await resolver.resolve(
        [Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")]
    )
    assert isinstance(result.content[0], TextBlock)
    return result.content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "day, expected",
    [(20, "Monday"), (21, "Tuesday"), (22, "Wednesday"), (23, "Thursday"),
     (24, "Friday"), (25, "Saturday"), (26, "Sunday")],
)
async def test_all_weekdays_are_english(frozen_localized_now, day, expected):
    instant = LocalizedDatetime(2026, 7, day, 9, tzinfo=ZoneInfo("UTC"))
    calls = frozen_localized_now(instant)
    assert instant.strftime("%A") == "jour-localisé"

    text = await resolve_text("UTC")

    assert calls == [ZoneInfo("UTC")]
    assert text == f"Today is {expected}; again {expected}."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zone, local_date, expected",
    [("America/Phoenix", "2026-07-22", "Wednesday"),
     ("Asia/Kolkata", "2026-07-23", "Thursday")],
)
async def test_configured_zone_selects_weekday(
    frozen_localized_now, zone, local_date, expected
):
    instant = LocalizedDatetime(2026, 7, 23, 2, 40, tzinfo=ZoneInfo("UTC"))
    calls = frozen_localized_now(instant)
    supplied = instant.astimezone(ZoneInfo(zone))
    assert supplied.date().isoformat() == local_date
    assert supplied.strftime("%A") == "jour-localisé"

    text = await resolve_text(zone)

    assert calls == [ZoneInfo(zone)]
    assert text == f"Today is {expected}; again {expected}."


@pytest.mark.asyncio
async def test_host_local_weekday_is_english(frozen_localized_now):
    instant = LocalizedDatetime(2026, 7, 26, 0, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    calls = frozen_localized_now(instant)
    assert instant.astimezone().strftime("%A") == "jour-localisé"

    text = await resolve_text()

    assert calls == [None]
    assert text == "Today is Sunday; again Sunday."
