"""Step 4 DI refactor tests for sr2.orchestrator.

Verifies:
  1. _build_resolver(config, deps) signature — deps: Dependencies, no bare llm param
  2. _build_transformer(config, deps) signature — deps: Dependencies, no bare llm param
  3. _build_layer(layer_config, token_counter, deps) — deps: Dependencies, no bare llm param
  4. SR2.__init__ constructs Dependencies(llm=llm) once and threads it
  5. Open/Closed proof — fake component with deps.llm builds via uniform path
  6. Fake resolver that ignores deps builds via uniform path
  7. Unknown resolver type raises ValueError
  8. Unknown transformer type raises ValueError
  9. SR2 end-to-end construction with summarize transformer
  10. _build_transformer source contains no "is SummarizationTransformer" identity check
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sr2.config.models import (
    LayerConfig,
    PipelineConfig,
    ResolverConfig,
    TransformerConfig,
)
from sr2.models import TextBlock, TokenUsage
from sr2.orchestrator import (
    SR2,
    _RESOLVERS,
    _TRANSFORMERS,
    _build_layer,
    _build_resolver,
    _build_transformer,
)
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.protocols.llm import CompletionRequest, CompletionResponse, StreamEvent


# ---------------------------------------------------------------------------
# Mock LLM — satisfies LLMCallable without network
# ---------------------------------------------------------------------------


class MockLLM:
    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text=f"response from {self.name}")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.calls.append(request)
        yield StreamEvent(type="text", text="hi")
        yield StreamEvent(type="end")


# ---------------------------------------------------------------------------
# Fake components for Open/Closed proof
# ---------------------------------------------------------------------------


class FakeDepsAwareTransformer:
    """A fake transformer whose build classmethod consumes deps.llm.

    This class must build correctly via _build_transformer without any
    modification to _build_transformer — the Open/Closed proof.
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.transform_calls: int = 0
        self.execution_count: int = 0

    @classmethod
    def build(cls, config: TransformerConfig, deps: Dependencies) -> "FakeDepsAwareTransformer":
        llm_dict = deps.llm or {}
        model_key = config.config.get("model", "default") if config.config else "default"
        resolved_llm = llm_dict.get(model_key, llm_dict.get("default"))
        return cls(llm=resolved_llm)

    async def transform(self, content: list, events: list) -> list:
        self.transform_calls += 1
        return content


class FakeDepsIgnorantResolver:
    """A fake resolver whose build classmethod ignores deps entirely.

    Registers via _RESOLVER_FACTORIES and must build via _build_resolver
    without any modification to _build_resolver.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.execution_count = 0

    @classmethod
    def build(cls, config: ResolverConfig, deps: Dependencies) -> "FakeDepsIgnorantResolver":
        text = config.config.get("text", "default-text") if config.config else "default-text"
        return cls(text=text)

    async def resolve(self, events: list) -> list:
        self.execution_count += 1
        return [TextBlock(text=self.text)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOKEN_COUNTER = CharacterTokenCounter()


def _minimal_resolver_config(resolver_type: str = "static") -> ResolverConfig:
    return ResolverConfig(type=resolver_type, config={"text": "hello"})


def _minimal_layer_config(
    resolver_type: str = "static",
    transformers: list[TransformerConfig] | None = None,
) -> LayerConfig:
    return LayerConfig(
        name="conversation",
        target="messages",
        resolvers=[_minimal_resolver_config(resolver_type)],
        transformers=transformers,
    )


def _minimal_pipeline_config(with_summarize: bool = False) -> PipelineConfig:
    transformers = [TransformerConfig(type="summarize")] if with_summarize else None
    return PipelineConfig(layers=[_minimal_layer_config(transformers=transformers)])


def _make_deps(llm_name: str = "default") -> Dependencies:
    return Dependencies(llm={"default": MockLLM(llm_name)})


# ---------------------------------------------------------------------------
# 1. _build_resolver signature — accepts deps: Dependencies
# ---------------------------------------------------------------------------


class TestBuildResolverSignature:
    def test_build_resolver_accepts_deps_positional(self):
        """_build_resolver(config, deps) works with positional deps argument."""
        config = _minimal_resolver_config("static")
        deps = _make_deps()

        result = _build_resolver(config, deps)

        assert result is not None

    def test_build_resolver_accepts_deps_keyword(self):
        """_build_resolver(config, deps=deps) works with keyword deps argument."""
        config = _minimal_resolver_config("static")
        deps = _make_deps()

        result = _build_resolver(config, deps=deps)

        assert result is not None

    def test_build_resolver_no_bare_llm_param(self):
        """_build_resolver signature must not accept a bare 'llm' param."""
        sig = inspect.signature(_build_resolver)
        assert "llm" not in sig.parameters, (
            "_build_resolver must not have a bare 'llm' parameter after Step 4 refactor"
        )

    def test_build_resolver_has_deps_param(self):
        """_build_resolver must have a 'deps' parameter of type Dependencies."""
        sig = inspect.signature(_build_resolver)
        assert "deps" in sig.parameters, (
            "_build_resolver must accept 'deps: Dependencies'"
        )


# ---------------------------------------------------------------------------
# 2. _build_transformer signature — accepts deps: Dependencies
# ---------------------------------------------------------------------------


class TestBuildTransformerSignature:
    def test_build_transformer_accepts_deps_positional(self):
        """_build_transformer(config, deps) works with positional deps argument."""
        config = TransformerConfig(type="summarize")
        deps = _make_deps()

        result = _build_transformer(config, deps)

        assert result is not None

    def test_build_transformer_accepts_deps_keyword(self):
        """_build_transformer(config, deps=deps) works with keyword deps argument."""
        config = TransformerConfig(type="summarize")
        deps = _make_deps()

        result = _build_transformer(config, deps=deps)

        assert result is not None

    def test_build_transformer_no_bare_llm_param(self):
        """_build_transformer signature must not have a bare 'llm' param after refactor."""
        sig = inspect.signature(_build_transformer)
        assert "llm" not in sig.parameters, (
            "_build_transformer must not have a bare 'llm' parameter after Step 4 refactor"
        )

    def test_build_transformer_has_deps_param(self):
        """_build_transformer must have a 'deps' parameter."""
        sig = inspect.signature(_build_transformer)
        assert "deps" in sig.parameters, (
            "_build_transformer must accept 'deps: Dependencies'"
        )


# ---------------------------------------------------------------------------
# 3. _build_layer signature — accepts deps: Dependencies (not llm)
# ---------------------------------------------------------------------------


class TestBuildLayerSignature:
    def test_build_layer_accepts_deps_positional(self):
        """_build_layer(layer_config, token_counter, deps) works positionally."""
        layer_config = _minimal_layer_config()
        deps = _make_deps()

        result = _build_layer(layer_config, TOKEN_COUNTER, deps)

        assert result is not None

    def test_build_layer_accepts_deps_keyword(self):
        """_build_layer(..., deps=deps) works with keyword argument."""
        layer_config = _minimal_layer_config()
        deps = _make_deps()

        result = _build_layer(layer_config, TOKEN_COUNTER, deps=deps)

        assert result is not None

    def test_build_layer_no_bare_llm_param(self):
        """_build_layer must not have a separate bare 'llm' param after refactor."""
        sig = inspect.signature(_build_layer)
        assert "llm" not in sig.parameters, (
            "_build_layer must not have a bare 'llm' parameter after Step 4 refactor"
        )

    def test_build_layer_has_deps_param(self):
        """_build_layer must have a 'deps' parameter."""
        sig = inspect.signature(_build_layer)
        assert "deps" in sig.parameters, (
            "_build_layer must accept 'deps: Dependencies'"
        )

    def test_build_layer_threads_deps_to_transformer(self):
        """_build_layer must actually pass deps into _build_transformer, not just accept it.

        Registers a fake deps-aware transformer via patched entry_points, builds a layer
        containing it, and verifies the constructed transformer received the LLM from deps.
        This proves deps is threaded through — not merely accepted and dropped.
        """
        import importlib.metadata
        from unittest.mock import MagicMock, patch

        # Build a fake entry point for the static resolver (needed by the layer)
        from sr2.pipeline.resolvers.static import StaticResolver

        def _ep_side_effect(group: str):
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "static"
                ep.load.return_value = StaticResolver
                dist = MagicMock(); dist.name = "sr2"; ep.dist = dist
                return [ep]
            if group == "sr2.transformers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = "fake_layer_thread"
                ep.load.return_value = FakeDepsAwareTransformer
                dist = MagicMock(); dist.name = "sr2"; ep.dist = dist
                return [ep]
            return []

        with patch("sr2.plugins.registry.entry_points", side_effect=_ep_side_effect):
            # Reset discovery state so patched entry_points is used
            _RESOLVERS._discovered = False
            _RESOLVERS._classes = {}
            _TRANSFORMERS._discovered = False
            _TRANSFORMERS._classes = {}
            try:
                llm_instance = MockLLM("threaded")
                deps = Dependencies(llm={"default": llm_instance})
                layer_config = LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[_minimal_resolver_config("static")],
                    transformers=[TransformerConfig(type="fake_layer_thread")],
                )

                layer = _build_layer(layer_config, TOKEN_COUNTER, deps)

                assert len(layer.transformers) == 1
                transformer = layer.transformers[0]
                assert isinstance(transformer, FakeDepsAwareTransformer)
                assert transformer.llm is llm_instance
            finally:
                _RESOLVERS._discovered = False
                _RESOLVERS._classes = {}
                _TRANSFORMERS._discovered = False
                _TRANSFORMERS._classes = {}


# ---------------------------------------------------------------------------
# 4. SR2.__init__ constructs Dependencies once and threads it
# ---------------------------------------------------------------------------


class TestSR2DepsConstruction:
    def test_sr2_constructs_with_valid_config_and_llm(self):
        """SR2 builds successfully — deps are constructed internally from llm dict."""
        config = _minimal_pipeline_config()
        llm = {"default": MockLLM()}

        instance = SR2(
            pipeline_config=config,
            llm=llm,
            token_counter=TOKEN_COUNTER,
        )

        assert instance is not None

    def test_sr2_constructs_with_summarize_transformer(self):
        """SR2 with a summarize layer builds successfully when deps are threaded correctly."""
        config = _minimal_pipeline_config(with_summarize=True)
        llm = {"default": MockLLM()}

        instance = SR2(
            pipeline_config=config,
            llm=llm,
            token_counter=TOKEN_COUNTER,
        )

        assert instance is not None

    def test_sr2_accepts_dict_without_default_key(self):
        """SR2 accepts a dict without a 'default' key (sr2-14: magic string removed)."""
        config = _minimal_pipeline_config()
        llm = {"other": MockLLM()}

        # A dict without "default" is now valid — first value used as driver.
        instance = SR2(pipeline_config=config, llm=llm, token_counter=TOKEN_COUNTER)
        assert instance is not None


# ---------------------------------------------------------------------------
# 5. Open/Closed proof — fake transformer using deps.llm
# ---------------------------------------------------------------------------


class TestOpenClosedTransformerProof:
    def _make_transformer_ep_side_effect(self, name: str, cls: type):
        """Return an entry_points side_effect that serves *cls* as transformer *name*."""
        import importlib.metadata
        from unittest.mock import MagicMock

        def _side_effect(group: str):
            if group == "sr2.transformers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = name
                ep.load.return_value = cls
                dist = MagicMock(); dist.name = "sr2"; ep.dist = dist
                return [ep]
            return []

        return _side_effect

    def _reset_transformers(self):
        _TRANSFORMERS._discovered = False
        _TRANSFORMERS._classes = {}

    def test_fake_transformer_builds_via_uniform_path(self):
        """A fake transformer registered via entry_points builds without
        modifying _build_transformer. Proves the factory path is uniform (no class
        identity checks needed for new component types).
        """
        from unittest.mock import patch

        side_effect = self._make_transformer_ep_side_effect("fake_deps_aware", FakeDepsAwareTransformer)
        self._reset_transformers()
        try:
            with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
                llm_instance = MockLLM("injected")
                deps = Dependencies(llm={"default": llm_instance})
                config = TransformerConfig(type="fake_deps_aware")

                result = _build_transformer(config, deps)

                assert isinstance(result, FakeDepsAwareTransformer)
                assert result.llm is llm_instance
        finally:
            self._reset_transformers()

    def test_fake_transformer_receives_correct_llm_from_deps(self):
        """The fake transformer's llm attribute is the one from deps, not a copy or None."""
        from unittest.mock import patch

        side_effect = self._make_transformer_ep_side_effect("fake_with_llm", FakeDepsAwareTransformer)
        self._reset_transformers()
        try:
            with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
                llm_a = MockLLM("a")
                llm_b = MockLLM("b")
                deps = Dependencies(llm={"default": llm_a, "alt": llm_b})
                config = TransformerConfig(type="fake_with_llm", config={"model": "alt"})

                result = _build_transformer(config, deps)

                assert isinstance(result, FakeDepsAwareTransformer)
                assert result.llm is llm_b
        finally:
            self._reset_transformers()

    @pytest.mark.asyncio
    async def test_fake_transformer_is_callable_after_build(self):
        """The fake transformer built via the uniform path can execute transform()."""
        from unittest.mock import patch

        side_effect = self._make_transformer_ep_side_effect("fake_callable", FakeDepsAwareTransformer)
        self._reset_transformers()
        try:
            with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
                deps = Dependencies(llm={"default": MockLLM()})
                config = TransformerConfig(type="fake_callable")

                transformer = _build_transformer(config, deps)
                content = [TextBlock(text="test content")]
                result = await transformer.transform(content, [])

                assert transformer.transform_calls == 1
                assert result == content
        finally:
            self._reset_transformers()


# ---------------------------------------------------------------------------
# 6. Open/Closed proof — fake resolver that ignores deps
# ---------------------------------------------------------------------------


class TestOpenClosedResolverProof:
    def _make_resolver_ep_side_effect(self, name: str, cls: type):
        """Return an entry_points side_effect that serves *cls* as resolver *name*."""
        import importlib.metadata
        from unittest.mock import MagicMock

        def _side_effect(group: str):
            if group == "sr2.resolvers":
                ep = MagicMock(spec=importlib.metadata.EntryPoint)
                ep.name = name
                ep.load.return_value = cls
                dist = MagicMock(); dist.name = "sr2"; ep.dist = dist
                return [ep]
            return []

        return _side_effect

    def _reset_resolvers(self):
        _RESOLVERS._discovered = False
        _RESOLVERS._classes = {}

    def test_fake_resolver_builds_via_uniform_path(self):
        """A fake resolver registered via entry_points builds without
        modifying _build_resolver. Proves the factory path is uniform.
        """
        from unittest.mock import patch

        side_effect = self._make_resolver_ep_side_effect("fake_ignorant", FakeDepsIgnorantResolver)
        self._reset_resolvers()
        try:
            with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
                deps = Dependencies(llm={"default": MockLLM()})
                config = ResolverConfig(type="fake_ignorant", config={"text": "hello from fake"})

                result = _build_resolver(config, deps)

                assert isinstance(result, FakeDepsIgnorantResolver)
                assert result.text == "hello from fake"
        finally:
            self._reset_resolvers()

    def test_fake_resolver_that_ignores_deps_builds_cleanly(self):
        """A resolver that ignores deps entirely still builds via the uniform factory path."""
        from unittest.mock import patch

        side_effect = self._make_resolver_ep_side_effect("fake_no_deps", FakeDepsIgnorantResolver)
        self._reset_resolvers()
        try:
            with patch("sr2.plugins.registry.entry_points", side_effect=side_effect):
                # Pass None deps — factory must not break even if deps is empty
                deps = Dependencies(llm=None)
                config = ResolverConfig(type="fake_no_deps", config={"text": "no-deps text"})

                result = _build_resolver(config, deps)

                assert isinstance(result, FakeDepsIgnorantResolver)
        finally:
            self._reset_resolvers()


# ---------------------------------------------------------------------------
# 7. Unknown resolver type raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownResolverType:
    def test_unknown_resolver_raises_value_error(self):
        """_build_resolver with an unregistered type raises PluginNotFoundError."""
        from unittest.mock import patch
        from sr2.plugins.errors import PluginNotFoundError

        config = ResolverConfig(type="totally_nonexistent_resolver_xyz")
        deps = _make_deps()

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            _RESOLVERS._discovered = False
            _RESOLVERS._classes = {}
            try:
                with pytest.raises(PluginNotFoundError):
                    _build_resolver(config, deps)
            finally:
                _RESOLVERS._discovered = False
                _RESOLVERS._classes = {}

    def test_unknown_resolver_error_message_includes_type(self):
        """PluginNotFoundError message includes the unknown type name for debuggability."""
        from unittest.mock import patch
        from sr2.plugins.errors import PluginNotFoundError

        bad_type = "nonexistent_resolver_abc123"
        config = ResolverConfig(type=bad_type)
        deps = _make_deps()

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            _RESOLVERS._discovered = False
            _RESOLVERS._classes = {}
            try:
                with pytest.raises(PluginNotFoundError, match=bad_type):
                    _build_resolver(config, deps)
            finally:
                _RESOLVERS._discovered = False
                _RESOLVERS._classes = {}


# ---------------------------------------------------------------------------
# 8. Unknown transformer type raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownTransformerType:
    def test_unknown_transformer_raises_value_error(self):
        """_build_transformer with an unregistered type raises PluginNotFoundError."""
        from unittest.mock import patch
        from sr2.plugins.errors import PluginNotFoundError

        config = TransformerConfig(type="totally_nonexistent_transformer_xyz")
        deps = _make_deps()

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            _TRANSFORMERS._discovered = False
            _TRANSFORMERS._classes = {}
            try:
                with pytest.raises(PluginNotFoundError):
                    _build_transformer(config, deps)
            finally:
                _TRANSFORMERS._discovered = False
                _TRANSFORMERS._classes = {}

    def test_unknown_transformer_error_message_includes_type(self):
        """PluginNotFoundError message includes the unknown type name for debuggability."""
        from unittest.mock import patch
        from sr2.plugins.errors import PluginNotFoundError

        bad_type = "nonexistent_transformer_abc123"
        config = TransformerConfig(type=bad_type)
        deps = _make_deps()

        with patch("sr2.plugins.registry.entry_points", return_value=[]):
            _TRANSFORMERS._discovered = False
            _TRANSFORMERS._classes = {}
            try:
                with pytest.raises(PluginNotFoundError, match=bad_type):
                    _build_transformer(config, deps)
            finally:
                _TRANSFORMERS._discovered = False
                _TRANSFORMERS._classes = {}


# ---------------------------------------------------------------------------
# 9. SR2 end-to-end with summarize transformer
# ---------------------------------------------------------------------------


class TestSR2EndToEndWithSummarize:
    def test_sr2_constructs_pipeline_with_summarize_layer(self):
        """SR2 end-to-end: config with summarize transformer, multi-key llm dict — no error."""
        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[
                        ResolverConfig(type="static", config={"text": "You are helpful."})
                    ],
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[ResolverConfig(type="session")],
                    transformers=[
                        TransformerConfig(type="summarize", config={"model": "summarization"})
                    ],
                ),
            ]
        )
        llm_dict = {
            "default": MockLLM("default"),
            "summarization": MockLLM("summarization"),
        }

        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
        )

        assert instance is not None

    def test_sr2_all_built_layers_have_correct_transformer_count(self):
        """Each built layer has the correct number of transformers from the config."""
        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "sys"})],
                    transformers=None,
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[ResolverConfig(type="session")],
                    transformers=[TransformerConfig(type="summarize")],
                ),
            ]
        )
        llm_dict = {"default": MockLLM()}

        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
        )

        layers = instance._engine.layers
        assert len(layers) == 2
        # system layer: no transformers
        assert len(layers[0].transformers) == 0
        # conversation layer: one summarize transformer
        assert len(layers[1].transformers) == 1


# ---------------------------------------------------------------------------
# 10. No identity check — source inspection
# ---------------------------------------------------------------------------


class TestNoIdentityCheck:
    def test_build_transformer_source_has_no_is_summarization_transformer(self):
        """_build_transformer must not contain 'is SummarizationTransformer'.

        The identity check was the pre-Step-4 mechanism for injecting LLM into
        the summarize transformer. After Step 4, the factory's build() classmethod
        handles it uniformly. This test fails if the identity check is still present.
        """
        source = inspect.getsource(_build_transformer)
        assert "is SummarizationTransformer" not in source, (
            "_build_transformer still contains 'is SummarizationTransformer' identity check. "
            "Step 4 requires this to be replaced by the uniform factory.build(config, deps) path."
        )

    def test_orchestrator_module_source_has_no_is_summarization_transformer(self):
        """The entire orchestrator module must not contain 'is SummarizationTransformer'."""
        import sr2.orchestrator as orch_module
        module_source = inspect.getsource(orch_module)
        assert "is SummarizationTransformer" not in module_source, (
            "orchestrator.py still contains 'is SummarizationTransformer' after Step 4 refactor."
        )


# ---------------------------------------------------------------------------
# 11. No post-mutation of layer privates after construction (sr2-59)
# ---------------------------------------------------------------------------


class TestNoPlatformPostMutation:
    """Regression tests that define the REQUIRED BEHAVIOR for sr2-59.

    These pass under the current post-mutation implementation and must
    continue to pass after the refactor removes the post-mutation loops.
    They pin the observable invariants the refactor must preserve.
    """

    def test_layer_uses_engine_bus_not_placeholder(self):
        """After PipelineEngine construction, each layer's _event_bus IS the engine's bus.

        The layer must not hold the placeholder EventBus created in _build_layer —
        it must hold the same object as engine._bus.
        """
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.event_bus import EventBus
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.models import CompilationTarget
        from sr2.pipeline.resolvers.static import StaticResolver
        from sr2.config.models import ResolverConfig

        placeholder_bus = EventBus()
        resolver = StaticResolver.build(
            ResolverConfig(type="static", config={"text": "hi"}),
            _make_deps(),
        )
        layer = Layer(
            name="conversation",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            tool_providers=[],
            token_counter=TOKEN_COUNTER,
            event_bus=placeholder_bus,
        )

        engine = PipelineEngine(layers=[layer], token_counter=TOKEN_COUNTER)

        # The layer's bus must be the engine's bus — not the placeholder
        assert layer._event_bus is engine._bus
        assert layer._event_bus is not placeholder_bus

    def test_layer_uses_engine_provenance_store(self):
        """After PipelineEngine construction, each layer's _provenance_store IS the engine's store.

        The engine always creates or adopts a ProvenanceStore; the layer must
        share the same object (identity, not equality).
        """
        from sr2.pipeline.engine import PipelineEngine
        from sr2.pipeline.event_bus import EventBus
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.models import CompilationTarget
        from sr2.pipeline.resolvers.static import StaticResolver
        from sr2.config.models import ResolverConfig

        resolver = StaticResolver.build(
            ResolverConfig(type="static", config={"text": "hi"}),
            _make_deps(),
        )
        layer = Layer(
            name="conversation",
            target=CompilationTarget.MESSAGES,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[resolver],
            transformers=[],
            tool_providers=[],
            token_counter=TOKEN_COUNTER,
            event_bus=EventBus(),
        )

        engine = PipelineEngine(layers=[layer], token_counter=TOKEN_COUNTER)

        assert layer._provenance_store is engine._provenance_store

    def test_engine_built_from_sr2_layers_share_bus(self):
        """All layers in an SR2 instance share the same EventBus object (identity check).

        This is the end-to-end invariant: no matter how many layers exist,
        layer._event_bus is engine._bus for every one of them.
        """
        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "sys"})],
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[ResolverConfig(type="session")],
                ),
            ]
        )
        llm_dict = {"default": MockLLM()}

        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
        )

        engine = instance._engine
        for layer in engine.layers:
            assert layer._event_bus is engine._bus, (
                f"Layer '{layer.name}' has a different EventBus than engine._bus"
            )

    def test_tracer_injected_into_layers(self):
        """If a tracer is passed to SR2, all layers have it set after construction.

        Pins the tracer-injection invariant so the refactor cannot silently
        drop tracer propagation.
        """
        from unittest.mock import MagicMock

        pipeline_config = PipelineConfig(
            layers=[
                LayerConfig(
                    name="system",
                    target="system",
                    resolvers=[ResolverConfig(type="static", config={"text": "sys"})],
                ),
                LayerConfig(
                    name="conversation",
                    target="messages",
                    resolvers=[ResolverConfig(type="session")],
                ),
            ]
        )
        llm_dict = {"default": MockLLM()}
        fake_tracer = MagicMock(name="tracer")

        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
            tracer=fake_tracer,
        )

        engine = instance._engine
        for layer in engine.layers:
            assert layer._tracer is fake_tracer, (
                f"Layer '{layer.name}' does not have the expected tracer"
            )
