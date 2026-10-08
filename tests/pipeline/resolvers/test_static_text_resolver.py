"""Tests for StaticTextResolver."""

from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.resolvers.static_text import StaticTextResolver


def make_config(text: str) -> ResolverConfig:
    """Build a StaticTextResolver configuration."""
    return ResolverConfig(type="static_text", config={"text": text})


def turn_start_event() -> Event:
    """Build the resolver's default triggering event."""
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


class TestStaticTextResolver:
    @pytest.mark.asyncio
    async def test_interpolates_non_empty_area(self):
        resolver = StaticTextResolver(
            make_config("You are working in {area}."),
            run_context_provider=lambda: {"area": "sr2-spectre"},
        )

        result = await resolver.resolve([turn_start_event()])

        assert isinstance(result, ResolvedContent)
        assert result.content == [TextBlock(text="You are working in sr2-spectre.")]

    @pytest.mark.asyncio
    async def test_emits_placeholder_free_text_verbatim(self):
        resolver = StaticTextResolver(
            make_config("You are Spectre. Be direct."),
            run_context_provider=lambda: {"area": "ignored"},
        )

        result = await resolver.resolve([turn_start_event()])

        assert result.content == [TextBlock(text="You are Spectre. Be direct.")]

    @pytest.mark.asyncio
    async def test_interpolates_date_placeholder_with_local_today(self, monkeypatch):
        fixed = datetime.date(2026, 10, 8)
        monkeypatch.setattr(
            "sr2.pipeline.resolvers.static_text.date.today",
            lambda: fixed,
        )
        resolver = StaticTextResolver(
            make_config("Hoje é {date}."),
            run_context_provider=lambda: {"area": "culinaria-e-dieta"},
        )

        result = await resolver.resolve([turn_start_event()])

        assert result.content == [TextBlock(text="Hoje é 2026-10-08.")]

    @pytest.mark.asyncio
    async def test_date_placeholder_substitutes_all_occurrences(self, monkeypatch):
        fixed = datetime.date(2026, 1, 2)
        monkeypatch.setattr(
            "sr2.pipeline.resolvers.static_text.date.today",
            lambda: fixed,
        )
        resolver = StaticTextResolver(
            make_config("Criado em {date}; atualizado em {date}."),
        )

        result = await resolver.resolve([turn_start_event()])

        assert result.content == [TextBlock(text="Criado em 2026-01-02; atualizado em 2026-01-02.")]

    @pytest.mark.asyncio
    async def test_date_placeholder_works_alongside_area_placeholder(self, monkeypatch):
        fixed = datetime.date(2027, 3, 15)
        monkeypatch.setattr(
            "sr2.pipeline.resolvers.static_text.date.today",
            lambda: fixed,
        )
        resolver = StaticTextResolver(
            make_config("Canal: {area} — data: {date}."),
            run_context_provider=lambda: {"area": "jeep"},
        )

        result = await resolver.resolve([turn_start_event()])

        assert result.content == [TextBlock(text="Canal: jeep — data: 2027-03-15.")]

    @pytest.mark.asyncio
    async def test_date_placeholder_never_emits_raw_braces_when_resolving(self, monkeypatch):
        """Any text containing the placeholder must have it replaced, never emitted raw."""
        fixed = datetime.date(2030, 6, 1)
        monkeypatch.setattr(
            "sr2.pipeline.resolvers.static_text.date.today",
            lambda: fixed,
        )
        resolver = StaticTextResolver(
            make_config("Data atual: {date}."),
            run_context_provider=lambda: {"area": "any"},
        )

        result = await resolver.resolve([turn_start_event()])

        text = result.content[0].text
        assert "{date}" not in text
        assert text == "Data atual: 2030-06-01."

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider",
        [
            lambda: {"area": ""},
            lambda: None,
            lambda: {"mode": "headless"},
            None,
        ],
    )
    async def test_unavailable_area_interpolates_to_empty_string(self, provider):
        resolver = StaticTextResolver(
            make_config("You are working in {area}."), run_context_provider=provider
        )

        result = await resolver.resolve([turn_start_event()])

        assert result.content == [TextBlock(text="You are working in .")]

    @pytest.mark.asyncio
    async def test_reads_text_from_live_config_on_each_resolve(self):
        config = make_config("first {area}")
        resolver = StaticTextResolver(
            config, run_context_provider=lambda: {"area": "area"}
        )

        first = await resolver.resolve([turn_start_event()])
        config.config["text"] = "second {area}"
        second = await resolver.resolve([turn_start_event()])

        assert first.content == [TextBlock(text="first area")]
        assert second.content == [TextBlock(text="second area")]

    @pytest.mark.asyncio
    async def test_build_wires_run_context_provider_from_dependencies(self):
        resolver = StaticTextResolver.build(
            make_config("in {area}"),
            Dependencies(run_context_provider=lambda: {"area": "harbinger"}),
        )

        result = await resolver.resolve([turn_start_event()])

        assert result.content == [TextBlock(text="in harbinger")]

    def test_registers_static_text_alongside_markdown_file(self):
        from importlib.metadata import entry_points

        names = {entry_point.name for entry_point in entry_points(group="sr2.resolvers")}

        assert {"markdown_file", "static_text"} <= names
