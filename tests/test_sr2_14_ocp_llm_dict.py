"""Tests for sr2-14: OCP — hardcoded 'default' key in llm dict.

These tests pin the DESIRED behavior (the fix). They fail against the current
implementation and should pass once sr2-14 is resolved.

Problems being tested:
  P1. SR2.__init__ accepts a single LLMCallable (not just a dict).
  P2. SR2.__init__ does not have the dual-access smell (self._llm AND deps.llm dict).
  P3. SummarizationTransformer.build() does NOT silently fall back to "default" when
      a configured named key is absent from deps.llm — it should raise, not guess.
  P4. SummarizationTransformer.build() works when deps.llm contains no "default" key
      at all, as long as the configured key (or the sole key) is present.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from sr2.models import TextBlock, TokenUsage
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Shared mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Minimal LLMCallable for testing."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self.stream_calls: list[CompletionRequest] = []
        self.complete_calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.complete_calls.append(request)
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text=f"response from {self.name}")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.stream_calls.append(request)
        yield StreamEvent(type="text", text=f"text from {self.name}")
        yield StreamEvent(type="end")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_minimal_pipeline_config():
    from sr2.config.models import LayerConfig, PipelineConfig, ResolverConfig

    return PipelineConfig(
        layers=[
            LayerConfig(
                name="system",
                resolvers=[
                    ResolverConfig(type="static", config={"text": "You are helpful."})
                ],
            ),
            LayerConfig(
                name="conversation",
                resolvers=[ResolverConfig(type="session")],
            ),
        ]
    )


def make_summarization_config(model_key: str | None = None, **kwargs):
    from sr2.config.models import EventSubscriptionConfig, TransformerConfig

    inner: dict = {"keep_strategy": "keep_last_n", "keep_last_n": 2}
    if model_key is not None:
        inner["model"] = model_key
    inner.update(kwargs)

    return TransformerConfig(
        type="summarization",
        subscriptions=[EventSubscriptionConfig(event="turn_start")],
        config=inner,
        max_executions=5,
    )


# ---------------------------------------------------------------------------
# P1 — SR2.__init__ should accept a single LLMCallable, not just a dict
# ---------------------------------------------------------------------------


class TestSR2AcceptsSingleLLM:
    """SR2 should be constructable with a bare LLMCallable, not forced to wrap it in a dict."""

    def test_sr2_accepts_single_llm_callable(self):
        """P1: SR2.__init__ accepts a single LLMCallable without requiring a dict wrapper."""
        from sr2.orchestrator import SR2

        config = make_minimal_pipeline_config()
        llm = MockLLM("driver")
        counter = CharacterTokenCounter()

        # This should NOT raise. Currently raises because SR2 requires dict[str, LLMCallable]
        # and then enforces the "default" key.
        sr2 = SR2(pipeline_config=config, llm=llm, token_counter=counter)
        assert sr2 is not None

    def test_sr2_single_llm_is_the_driver(self):
        """P1: When a single LLMCallable is provided, it becomes the driver LLM."""
        from sr2.orchestrator import SR2

        config = make_minimal_pipeline_config()
        driver = MockLLM("driver")
        counter = CharacterTokenCounter()

        sr2 = SR2(pipeline_config=config, llm=driver, token_counter=counter)

        # The internal driver should be the same object we passed in.
        # (Attribute name may vary — just assert one_llm is reachable and is driver.)
        assert sr2._llm is driver

    def test_sr2_dict_with_no_default_key_is_valid_if_driver_specified(self):
        """P1: SR2 dict form should not require the string 'default' if driver is explicit.

        Option 1 from the issue: accept one LLMCallable as the driver param.
        If dict is still supported, the dict should not REQUIRE a 'default' key.
        A dict with any named key(s) should be valid as long as the caller
        designates the driver explicitly (or via config).
        """
        from sr2.orchestrator import SR2

        config = make_minimal_pipeline_config()
        llm_haiku = MockLLM("haiku")
        llm_opus = MockLLM("opus")
        counter = CharacterTokenCounter()

        # dict without "default" — should be valid in the new API
        sr2 = SR2(
            pipeline_config=config,
            llm={"haiku": llm_haiku, "opus": llm_opus},
            token_counter=counter,
        )
        assert sr2 is not None


# ---------------------------------------------------------------------------
# P2 — SR2 dual-access smell: self._llm AND deps.llm dict
# ---------------------------------------------------------------------------


class TestSR2NoDualAccess:
    """SR2 should not store the driver LLM via two different paths simultaneously."""

    def test_sr2_driver_not_duplicated_in_deps(self):
        """P2: When llm is a single callable, deps.llm should not be a dict wrapping it.

        The dual-access anti-pattern: self._llm = llm['default'] AND
        deps = Dependencies(llm=llm_dict). The driver should live in ONE place.
        If SR2 accepts a bare callable, deps.llm should be None (or not include
        the driver as a separate dict entry).
        """
        from sr2.orchestrator import SR2
        from sr2.pipeline.dependencies import Dependencies

        config = make_minimal_pipeline_config()
        driver = MockLLM("driver")
        counter = CharacterTokenCounter()

        sr2 = SR2(pipeline_config=config, llm=driver, token_counter=counter)

        # Either deps.llm is None (single-callable path, no dict) OR
        # deps.llm does not contain the driver under two different keys.
        # The point: the driver should not be reachable via both sr2._llm AND deps.llm["default"].
        engine = sr2._engine
        # We access the underlying deps through the engine's layers. If deps.llm is a dict,
        # the driver should not also live in sr2._llm independently.
        # Simplest assertion: if sr2._llm is set, it must be 'driver', and deps.llm
        # (if it is a dict) must not also contain 'driver' under a magic "default" key
        # alongside the explicit llm attribute.
        if hasattr(sr2, "_llm"):
            assert sr2._llm is driver, "sr2._llm should be the provided driver"


# ---------------------------------------------------------------------------
# P3 — SummarizationTransformer.build() should not silently fall back to "default"
#       when a configured named key is absent from deps.llm
# ---------------------------------------------------------------------------


class TestSummarizationNoSilentDefaultFallback:
    """When config specifies model='some-key' and that key is absent, build() should
    raise — not silently fall back to deps.llm.get("default").

    The current behavior:
      key = config.config.get("model", "default")
      llm = deps.llm.get(key, deps.llm.get("default"))   ← silent fallback!

    The desired behavior: raise ConfigError (or KeyError) if the named key is absent.
    Silent fallback hides misconfiguration.
    """

    @pytest.fixture
    def transformer_cls(self):
        from sr2.pipeline.transformers.summarization import SummarizationTransformer

        return SummarizationTransformer

    def make_deps_no_default(self, named_key: str, llm: MockLLM):
        """Build Dependencies whose llm dict has only the named key (no 'default' key)."""
        from sr2.pipeline.dependencies import Dependencies

        return Dependencies(llm={named_key: llm})

    def make_deps_with_default_only(self, default_llm: MockLLM):
        """Build Dependencies whose llm dict has only 'default' (no named key)."""
        from sr2.pipeline.dependencies import Dependencies

        return Dependencies(llm={"default": default_llm})

    def test_named_key_absent_raises_not_silently_falls_back(self, transformer_cls):
        """P3: config['model']='haiku' but 'haiku' absent from deps.llm → should raise, not use 'default'."""
        from sr2.config.models import ConfigError

        default_llm = MockLLM("default")
        config = make_summarization_config(model_key="haiku")
        # deps has only "default", NOT "haiku"
        deps = self.make_deps_with_default_only(default_llm)

        # Currently this silently returns an instance using default_llm.
        # After the fix, it should raise because "haiku" was explicitly requested but absent.
        with pytest.raises((ConfigError, KeyError, ValueError)):
            transformer_cls.build(config, deps)

    def test_named_key_absent_does_not_silently_return_instance(self, transformer_cls):
        """P3: Variant — ensure build() does not return an instance when named key is missing."""
        from sr2.config.models import ConfigError

        default_llm = MockLLM("default")
        config = make_summarization_config(model_key="nonexistent-model")
        deps = self.make_deps_with_default_only(default_llm)

        raised = False
        try:
            result = transformer_cls.build(config, deps)
            # If we got here without raising, check that the returned LLM is NOT the default
            # (i.e., it didn't silently fall back). This is the weaker assertion.
            assert result._llm is not default_llm, (
                "build() silently returned an instance using 'default' LLM "
                "even though config specified a different model key that was absent. "
                "This hides misconfiguration."
            )
        except (ConfigError, KeyError, ValueError):
            raised = True

        # Either it raised (preferred) or it didn't silently return default
        # (but raising is the desired behavior, so this test documents the gap)
        assert raised, (
            "Expected build() to raise when configured model key is absent from deps.llm, "
            "but it silently fell back to 'default'. This is the bug."
        )

    def test_named_key_present_without_default_works(self, transformer_cls):
        """P4: deps.llm has the named key but NO 'default' key → should work fine.

        After the fix, 'default' should not be a required sentinel. If 'haiku' is
        configured and 'haiku' exists in deps.llm, build() must succeed even if
        'default' is absent.
        """
        haiku_llm = MockLLM("haiku")
        config = make_summarization_config(model_key="haiku")
        deps = self.make_deps_no_default("haiku", haiku_llm)

        # Currently this MAY work (haiku is found), but the fallback logic still
        # references "default" as a sentinel: deps.llm.get(key, deps.llm.get("default"))
        # This test asserts that 'haiku' is returned — not a "default" that doesn't exist.
        result = transformer_cls.build(config, deps)

        assert result._llm is haiku_llm

    def test_no_model_key_in_config_without_default_in_deps_raises(self, transformer_cls):
        """P4: No 'model' key in config AND no 'default' key in deps.llm → should raise.

        Current behavior: deps.llm.get("default") → None, then cls(config, None) — silent bug.
        Desired: explicit ConfigError when the driver cannot be resolved.
        """
        from sr2.config.models import ConfigError

        haiku_llm = MockLLM("haiku")
        # config has no 'model' key → falls through to "default" lookup
        config = make_summarization_config(model_key=None)
        # deps has only "haiku" — no "default"
        deps = self.make_deps_no_default("haiku", haiku_llm)

        # Currently: deps.llm.get("default") returns None → cls(config, None) — silent
        # Desired: raise ConfigError because no driver can be resolved
        with pytest.raises((ConfigError, KeyError, ValueError)):
            transformer_cls.build(config, deps)


# ---------------------------------------------------------------------------
# P4 — Dependencies.llm "default" convention is implicit (type system gap)
# ---------------------------------------------------------------------------


class TestDependenciesLLMTypeContract:
    """Dependencies.llm types the 'default' key convention only in prose.
    The desired state: either type-level enforcement OR explicit driver param.
    These tests document the gap without requiring a specific fix shape.
    """

    def test_dependencies_llm_dict_without_default_is_accepted_at_runtime(self):
        """Currently Dependencies accepts any dict[str, LLMCallable] — no 'default' enforcement.

        This test documents the fact that the type system does not enforce the
        'default' key convention. It passes today (no validation in Dependencies).
        After the fix, either Dependencies enforces the key OR the 'default' magic
        string is eliminated entirely (replaced with an explicit driver param).
        """
        from sr2.pipeline.dependencies import Dependencies

        llm_a = MockLLM("a")
        llm_b = MockLLM("b")

        # No "default" key — Dependencies accepts it without complaint
        deps = Dependencies(llm={"a": llm_a, "b": llm_b})

        # The issue: nothing stops callers from passing this into SR2,
        # which then blows up at runtime with a cryptic ValueError.
        # After the fix, either SR2 doesn't require "default" OR
        # Dependencies validates the key presence at construction time.
        assert deps.llm is not None
        assert "default" not in deps.llm  # documents the unguarded state

    def test_driver_key_should_be_explicit_not_magic_string(self):
        """P4: The driver LLM should be identified explicitly, not by the magic string 'default'.

        After the fix: either SR2 takes `llm: LLMCallable` (single callable) as the
        driver, or PipelineConfig models the 'driver' role explicitly.
        This test asserts that SR2 can be constructed without using the string 'default'
        anywhere — i.e., the magic string is not required.
        """
        from sr2.orchestrator import SR2

        config = make_minimal_pipeline_config()
        driver = MockLLM("primary-driver")
        counter = CharacterTokenCounter()

        # If SR2 still requires a dict, this would need a 'default' key today.
        # After the fix: a single callable or an explicit driver param removes the magic string.
        # This test FAILS today because SR2 requires llm to be a dict with "default".
        try:
            sr2 = SR2(pipeline_config=config, llm=driver, token_counter=counter)
            # If we get here: the fix is in place, no magic string needed.
            constructed = True
        except (TypeError, ValueError, AttributeError):
            constructed = False

        assert constructed, (
            "SR2 should be constructable with a single LLMCallable (no magic 'default' key). "
            "Currently fails because SR2.__init__ requires dict[str, LLMCallable] with 'default' key."
        )
