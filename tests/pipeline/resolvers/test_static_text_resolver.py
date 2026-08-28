"""Tests for StaticTextResolver.

Covers:
  - config.text with {area} interpolates the run context's area
  - text with no placeholders emits verbatim
  - empty / None / absent area interpolates {area} to empty string without
    crashing (the resolver always emits its text — suppression is a
    layer-condition concern)
  - no run context provider at all interpolates {area} to empty string
  - registered in the sr2.resolvers entry-point group alongside markdown_file
"""

from __future__ import annotations

import pytest

from sr2.config.models import ResolverConfig
from sr2.models import TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.resolvers.static_text import StaticTextResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(text: str, **kwargs) -> ResolverConfig:
    """Build a ResolverConfig for StaticTextResolver."""
    return ResolverConfig(type="static_text", config={"text": text}, **kwargs)


def make_turn_start_event() -> Event:
    return Event(name="turn_start", phase=EventPhase.STARTING, source_layer="core")


def make_provider(area: str | None = "sr2-spectre", ctx: dict | None = None):
    """Build a run-context provider returning the given context dict."""
    if ctx is not None:
        return lambda: ctx
    if area is None:
        return lambda: {"mode": "headless"}
    return lambda: {"area": area}


# ---------------------------------------------------------------------------
# 1. {area} interpolation
# ---------------------------------------------------------------------------


class TestStaticTextAreaInterpolation:
    @pytest.mark.asyncio
    async def test_area_placeholder_is_substituted(self):
        """A non-empty area from the provider replaces {area}."""
        resolver = StaticTextResolver(
            make_config("You are working in the {area} area."),
            run_context_provider=make_provider("sr2-spectre"),
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert isinstance(result, ResolvedContent)
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "You are working in the sr2-spectre area."

    @pytest.mark.asyncio
    async def test_area_is_read_per_resolve(self):
        """The provider is consulted on every resolve() call, not cached."""
        calls = []

        def provider():
            calls.append(1)
            return {"area": f"area-{len(calls)}"}

        resolver = StaticTextResolver(
            make_config("area={area}"), run_context_provider=provider
        )
        r1 = await resolver.resolve([make_turn_start_event()])
        r2 = await resolver.resolve([make_turn_start_event()])
        assert r1.content[0].text == "area=area-1"
        assert r2.content[0].text == "area=area-2"

    @pytest.mark.asyncio
    async def test_multiple_placeholders_all_substituted(self):
        """Every {area} occurrence is replaced."""
        resolver = StaticTextResolver(
            make_config("{area} / {area}"), run_context_provider=make_provider("nsr2")
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "nsr2 / nsr2"

    @pytest.mark.asyncio
    async def test_build_wires_provider_from_deps(self):
        """build() pulls run_context_provider out of Dependencies."""
        deps = Dependencies(run_context_provider=make_provider("harbinger"))
        resolver = StaticTextResolver.build(
            make_config("in {area}"), deps
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "in harbinger"

    @pytest.mark.asyncio
    async def test_build_with_none_deps(self):
        """build() with deps=None does not crash; {area} becomes empty."""
        resolver = StaticTextResolver.build(make_config("in {area}"), None)
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "in "


# ---------------------------------------------------------------------------
# 2. No placeholders → verbatim
# ---------------------------------------------------------------------------


class TestStaticTextVerbatim:
    @pytest.mark.asyncio
    async def test_text_without_placeholders_is_verbatim(self):
        """Text with no {area} is emitted unchanged."""
        resolver = StaticTextResolver(
            make_config("You are Spectre. Be direct."),
            run_context_provider=make_provider("sr2-spectre"),
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "You are Spectre. Be direct."

    @pytest.mark.asyncio
    async def test_verbatim_even_when_area_available(self):
        """An available area does not leak into placeholder-free text."""
        resolver = StaticTextResolver(
            make_config("fixed text"), run_context_provider=make_provider("x")
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "fixed text"


# ---------------------------------------------------------------------------
# 3. Empty / None / absent area → empty string, no crash, no branching
# ---------------------------------------------------------------------------


class TestStaticTextEmptyArea:
    @pytest.mark.asyncio
    async def test_empty_string_area_interpolates_to_empty(self):
        """area == '' substitutes to '' — text is still emitted."""
        resolver = StaticTextResolver(
            make_config("You are in the {area} area."),
            run_context_provider=make_provider(""),
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "You are in the  area."

    @pytest.mark.asyncio
    async def test_provider_returning_none_interpolates_to_empty(self):
        """A provider that returns None substitutes to '' without crashing."""
        resolver = StaticTextResolver(
            make_config("in {area}"), run_context_provider=lambda: None
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "in "

    @pytest.mark.asyncio
    async def test_absent_area_key_interpolates_to_empty(self):
        """A context dict without the 'area' key substitutes to ''."""
        resolver = StaticTextResolver(
            make_config("in {area}"),
            run_context_provider=make_provider(ctx={"mode": "headless"}),
        )
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "in "

    @pytest.mark.asyncio
    async def test_no_provider_at_all_interpolates_to_empty(self):
        """No provider (None) substitutes to '' without crashing."""
        resolver = StaticTextResolver(make_config("in {area}"))
        result = await resolver.resolve([make_turn_start_event()])
        assert result.content[0].text == "in "

    @pytest.mark.asyncio
    async def test_empty_area_does_not_suppress_content(self):
        """The resolver never drops the block — suppression is a layer concern."""
        for provider in (
            make_provider(""),
            lambda: None,
            make_provider(ctx={}),
            None,
        ):
            resolver = StaticTextResolver(
                make_config("in {area}"), run_context_provider=provider
            )
            result = await resolver.resolve([make_turn_start_event()])
            assert len(result.content) == 1
            assert result.content[0].text == "in "


# ---------------------------------------------------------------------------
# 4. Entry-point registration
# ---------------------------------------------------------------------------


class TestStaticTextPluginRegistration:
    """static_text is registered as an SR2 resolver entry point."""

    def test_static_text_in_sr2_resolvers_group(self):
        """The 'static_text' entry point is listed in the sr2.resolvers group."""
        from importlib.metadata import entry_points

        eps = entry_points(group="sr2.resolvers")
        names = [ep.name for ep in eps]
        assert "static_text" in names, (
            f"'static_text' not found in sr2.resolvers entry points. Found: {names}"
        )

    def test_registered_alongside_markdown_file(self):
        """static_text and markdown_file coexist in the same group."""
        from importlib.metadata import entry_points

        names = [ep.name for ep in entry_points(group="sr2.resolvers")]
        assert "static_text" in names and "markdown_file" in names, names

    def test_entry_point_loads_resolver_class(self):
        """The entry point loads a class with build() and a resolve() method."""
        from importlib.metadata import entry_points

        eps = entry_points(group="sr2.resolvers")
        ep = next((e for e in eps if e.name == "static_text"), None)
        assert ep is not None, "static_text entry point not found"
        cls = ep.load()
        assert hasattr(cls, "build") and hasattr(cls, "resolve"), (
            f"Loaded class {cls!r} lacks build()/resolve()."
        )

    def test_plugin_registry_can_discover_static_text(self):
        """PluginRegistry for sr2.resolvers can list 'static_text'."""
        from sr2.pipeline.protocols import Resolver
        from sr2.plugins import PluginRegistry

        registry = PluginRegistry(group="sr2.resolvers", protocol=Resolver)
        names = registry.names()
        assert "static_text" in names, (
            f"'static_text' not in registry names. Found: {names}"
        )
