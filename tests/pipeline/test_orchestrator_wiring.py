"""Tests for orchestrator wiring of SummarizationTransformer — Step 4.

Acceptance Criteria covered:
  FR8:  "summarize" is registered in _TRANSFORMER_FACTORIES.
  FR9:  _build_transformer accepts (config, llm_dict) and threads the resolved
        LLMCallable to SummarizationTransformer; _build_layer threads llm through.
  AC9:  model key resolution — explicit key hit, explicit key miss (fallback),
        no key (fallback to "default").
  AC12: SR2 constructs without error when config includes a summarize transformer.
  AC13: Unknown transformer type raises ValueError (pre-existing behaviour preserved).

Non-goals:
  - No actual LLM calls are made.
  - Internal implementation details of PipelineEngine / Layer are not asserted.
"""

from __future__ import annotations

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
    _TRANSFORMERS,
    _build_layer,
    _build_transformer,
)
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.transformers.summarization import SummarizationTransformer
from sr2.protocols.llm import CompletionRequest, CompletionResponse


# ---------------------------------------------------------------------------
# Mock LLM — satisfies LLMCallable protocol without real network calls
# ---------------------------------------------------------------------------


class MockLLM:
    """Minimal LLMCallable implementation for wiring tests."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        return CompletionResponse(
            id="mock",
            content=[TextBlock(text=f"summary from {self.name}")],
            stop_reason="end_turn",
            usage=TokenUsage(),
        )

    async def stream(self, request: CompletionRequest):
        # Not exercised by wiring tests
        return
        yield  # make it an async generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize_transformer_config(model: str | None = None) -> TransformerConfig:
    cfg: dict = {}
    if model is not None:
        cfg["model"] = model
    return TransformerConfig(type="summarize", config=cfg)


def _minimal_layer_config(transformers: list[TransformerConfig] | None = None) -> LayerConfig:
    return LayerConfig(
        name="conversation",
        target="messages",
        resolvers=[ResolverConfig(type="static", config={"text": "hello"})],
        transformers=transformers,
    )


def _minimal_pipeline_config(with_summarize: bool = False) -> PipelineConfig:
    transformers = [_summarize_transformer_config()] if with_summarize else None
    return PipelineConfig(
        layers=[_minimal_layer_config(transformers=transformers)],
    )


TOKEN_COUNTER = CharacterTokenCounter()


# ---------------------------------------------------------------------------
# FR8 — Registration
# ---------------------------------------------------------------------------


class TestFR8Registration:
    def test_summarize_key_exists_in_factory_registry(self):
        """'summarize' must be registered in _TRANSFORMERS registry."""
        assert "summarize" in _TRANSFORMERS.names()

    def test_build_transformer_summarize_returns_summarization_transformer(self):
        """`_build_transformer` with type 'summarize' produces a SummarizationTransformer."""
        config = _summarize_transformer_config()
        deps = Dependencies(llm={"default": MockLLM("default")})

        result = _build_transformer(config, deps)

        assert isinstance(result, SummarizationTransformer)


# ---------------------------------------------------------------------------
# AC9 — Model key resolution
# ---------------------------------------------------------------------------


class TestAC9ModelKeyResolution:
    @pytest.mark.asyncio
    async def test_explicit_model_key_hit_uses_correct_llm(self):
        """config["model"] = "custom" and llm has "custom" key → the custom LLM is called."""
        llm_a = MockLLM("default")
        llm_b = MockLLM("custom")
        deps = Dependencies(llm={"default": llm_a, "custom": llm_b})

        config = _summarize_transformer_config(model="custom")
        transformer = _build_transformer(config, deps)

        assert isinstance(transformer, SummarizationTransformer)
        content = [TextBlock(text=f"turn_{i}") for i in range(5)]
        await transformer.transform(content, [])

        assert len(llm_b.calls) == 1
        assert len(llm_a.calls) == 0

    @pytest.mark.asyncio
    async def test_explicit_model_key_miss_raises(self):
        """config["model"] = "nonexistent" and llm has no such key → raises (sr2-14: no silent fallback)."""
        from sr2.config.models import ConfigError

        llm_a = MockLLM("default")
        deps = Dependencies(llm={"default": llm_a})

        config = _summarize_transformer_config(model="nonexistent")

        with pytest.raises((ConfigError, KeyError, ValueError)):
            _build_transformer(config, deps)

    @pytest.mark.asyncio
    async def test_no_model_key_uses_default(self):
        """config has no 'model' key → "default" LLM is called."""
        llm_a = MockLLM("default")
        deps = Dependencies(llm={"default": llm_a})

        config = _summarize_transformer_config(model=None)
        transformer = _build_transformer(config, deps)

        assert isinstance(transformer, SummarizationTransformer)
        content = [TextBlock(text=f"turn_{i}") for i in range(5)]
        await transformer.transform(content, [])

        assert len(llm_a.calls) == 1


# ---------------------------------------------------------------------------
# FR9 — _build_transformer and _build_layer signatures
# ---------------------------------------------------------------------------


class TestFR9Signatures:
    def test_build_transformer_accepts_deps_positional(self):
        """_build_transformer(config, deps) works with positional arguments."""
        config = _summarize_transformer_config()
        deps = Dependencies(llm={"default": MockLLM()})

        # Must not raise
        result = _build_transformer(config, deps)
        assert result is not None

    def test_build_layer_with_summarize_transformer_constructs(self):
        """_build_layer threads deps through so summarize transformers inside layers construct."""
        layer_config = _minimal_layer_config(
            transformers=[_summarize_transformer_config()]
        )
        deps = Dependencies(llm={"default": MockLLM()})

        # Must not raise
        layer = _build_layer(layer_config, TOKEN_COUNTER, deps)
        assert layer is not None

    @pytest.mark.asyncio
    async def test_build_layer_transformer_receives_correct_llm_instance(self):
        """The transformer inside the built layer uses the injected LLM when called."""
        llm_instance = MockLLM("injected")
        deps = Dependencies(llm={"default": llm_instance})
        layer_config = _minimal_layer_config(
            transformers=[_summarize_transformer_config()]
        )

        layer = _build_layer(layer_config, TOKEN_COUNTER, deps)

        assert len(layer.transformers) == 1
        transformer = layer.transformers[0]
        assert isinstance(transformer, SummarizationTransformer)

        content = [TextBlock(text=f"t_{i}") for i in range(5)]
        await transformer.transform(content, [])
        assert len(llm_instance.calls) == 1

    @pytest.mark.asyncio
    async def test_build_layer_model_key_respected_within_layer(self):
        """Model key in transformer config is resolved against the llm dict within _build_layer."""
        llm_default = MockLLM("default")
        llm_summary = MockLLM("summarization")
        deps = Dependencies(llm={"default": llm_default, "summarization": llm_summary})

        layer_config = _minimal_layer_config(
            transformers=[_summarize_transformer_config(model="summarization")]
        )

        layer = _build_layer(layer_config, TOKEN_COUNTER, deps)

        transformer = layer.transformers[0]
        assert isinstance(transformer, SummarizationTransformer)

        content = [TextBlock(text=f"t_{i}") for i in range(5)]
        await transformer.transform(content, [])
        assert len(llm_summary.calls) == 1
        assert len(llm_default.calls) == 0


# ---------------------------------------------------------------------------
# AC12 — SR2 end-to-end construction
# ---------------------------------------------------------------------------


class TestAC12SR2Construction:
    def test_sr2_constructs_with_summarize_transformer(self):
        """SR2 with a config that includes a summarize transformer constructs without error."""
        pipeline_config = _minimal_pipeline_config(with_summarize=True)
        llm_dict = {"default": MockLLM()}

        # Must not raise
        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
        )
        assert instance is not None

    def test_sr2_constructs_without_summarize_transformer(self):
        """SR2 without summarize transformer still constructs correctly (no regression)."""
        pipeline_config = _minimal_pipeline_config(with_summarize=False)
        llm_dict = {"default": MockLLM()}

        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
        )
        assert instance is not None

    def test_sr2_constructs_with_named_model_in_summarize_config(self):
        """SR2 with model='summarization' in transformer config and matching key constructs."""
        pipeline_config = PipelineConfig(
            layers=[
                _minimal_layer_config(
                    transformers=[_summarize_transformer_config(model="summarization")]
                )
            ]
        )
        llm_dict = {"default": MockLLM("default"), "summarization": MockLLM("summarization")}

        # Must not raise
        instance = SR2(
            pipeline_config=pipeline_config,
            llm=llm_dict,
            token_counter=TOKEN_COUNTER,
        )
        assert instance is not None


# ---------------------------------------------------------------------------
# Unknown transformer type — ValueError (existing behaviour preserved)
# ---------------------------------------------------------------------------


class TestUnknownTransformerType:
    def test_unknown_type_raises_value_error(self):
        """_build_transformer with an unregistered type raises PluginNotFoundError."""
        from sr2.pipeline.dependencies import Dependencies
        from sr2.plugins.errors import PluginNotFoundError

        config = TransformerConfig(type="totally_unknown_type")
        deps = Dependencies(llm={"default": MockLLM()})

        with pytest.raises(PluginNotFoundError):
            _build_transformer(config, deps)
