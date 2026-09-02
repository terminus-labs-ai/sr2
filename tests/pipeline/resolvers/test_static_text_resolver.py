"""Tests for StaticTextResolver."""

from __future__ import annotations

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
