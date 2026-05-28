"""Tests for sr2-10: mutable default argument bug in Layer.__init__.

Bug: `tool_providers: list = []` — all Layer instances constructed without
an explicit tool_providers share the same default list object.

Fix expected: `tool_providers: list | None = None`, assigned in body as
`list(tool_providers) if tool_providers else []`.

These tests are written against the UNFIXED code and must FAIL until the
fix is applied.
"""

from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------


def _make_layer(**kwargs) -> Layer:
    """Construct a Layer with minimal required arguments."""
    defaults = dict(
        name="test",
        target=CompilationTarget.SYSTEM,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
    )
    defaults.update(kwargs)
    return Layer(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolProvidersDefaultIsolation:
    def test_default_is_none_sentinel(self):
        """tool_providers default should be None (sentinel), not a mutable list.

        Inspects the function signature to confirm the default value is None,
        not []. Fails while the mutable default exists.
        """
        import inspect

        sig = inspect.signature(Layer.__init__)
        default = sig.parameters["tool_providers"].default
        assert default is None, (
            f"Expected None sentinel default for tool_providers, got {default!r}. "
            "Mutable default `[]` is present — fix not applied."
        )

    def test_default_parameter_object_is_not_a_list(self):
        """The default for tool_providers must not be a mutable list object.

        With `tool_providers: list = []` the same list object is reused across
        every call that omits the argument. Any mutation of that object — via
        CPython's __defaults__ tuple — propagates to all future callers.
        The fix replaces the default with None (immutable), eliminating the
        shared-object risk entirely.

        This test fails while the mutable default is present.
        """
        import inspect

        sig = inspect.signature(Layer.__init__)
        default = sig.parameters["tool_providers"].default
        assert not isinstance(default, list), (
            f"tool_providers default is a mutable list ({default!r}). "
            "Use `None` as the sentinel and assign `list(tool_providers) if tool_providers else []` "
            "in the body."
        )

    def test_mutating_default_list_does_not_affect_new_instance(self):
        """Mutating __defaults__ sentinel must not infect a subsequently-constructed Layer.

        With `tool_providers: list = []`, the default `[]` lives in
        Layer.__init__.__defaults__. This test locates that default and appends
        to it. Under the buggy code a new Layer then sees a non-empty
        tool_providers. Under the fixed code the default is None (immutable)
        so mutation is impossible.
        """
        import inspect

        sig = inspect.signature(Layer.__init__)
        param = sig.parameters["tool_providers"]

        # Only possible to poison if the default is actually a list.
        if not isinstance(param.default, list):
            # If we reach here the sentinel is already None — test would trivially
            # pass. We still assert the post-fix invariant so CI stays green.
            layer = _make_layer(name="post_fix")
            assert layer.tool_providers == []
            return

        # Buggy path: locate and mutate the shared default list.
        sentinel_list = param.default
        sentinel_list.append("POISON")

        try:
            layer = _make_layer(name="after_poison")
            assert layer.tool_providers == [], (
                f"tool_providers was {layer.tool_providers!r} — "
                "mutable default was shared and poisoned. Fix: use None sentinel."
            )
        finally:
            # Clean up so other tests are not affected.
            sentinel_list.clear()

    def test_instance_tool_providers_is_always_a_list(self):
        """Layer.tool_providers must be a list even when the param is omitted."""
        layer = _make_layer()
        assert isinstance(layer.tool_providers, list)

    def test_instance_tool_providers_is_always_a_list_when_none_passed(self):
        """Layer.tool_providers must be [] when None is passed explicitly (post-fix)."""
        layer = _make_layer(tool_providers=None)
        assert layer.tool_providers == []

    def test_explicit_tool_providers_are_preserved(self):
        """When tool_providers is supplied, the values must appear on the instance."""

        class StubProvider:
            subscriptions = []

        provider = StubProvider()
        layer = _make_layer(tool_providers=[provider])
        assert layer.tool_providers == [provider]

    def test_instance_attribute_is_a_copy_not_the_input(self):
        """layer.tool_providers must be a copy — mutating the input must not affect it."""

        class StubProvider:
            subscriptions = []

        provider_a = StubProvider()
        provider_b = StubProvider()
        original = [provider_a]
        layer = _make_layer(tool_providers=original)

        original.append(provider_b)  # mutate after construction

        assert len(layer.tool_providers) == 1, (
            "layer.tool_providers was not copied — mutation of the input list "
            "propagated into the instance."
        )
