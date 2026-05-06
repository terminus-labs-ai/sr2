"""Scaffolding tests — verify the project structure is sound.

Tests that:
- Public API imports work
- Protocols are runtime-checkable
- Config models validate correctly
- Facade instantiates
- Errors are properly hierarchized
- Dependency injection through facade
"""

import pytest
from typing import get_type_hints

from sr2 import SR2, TurnResult, ToolCall, TokenUsage
from sr2 import PipelineConfig, LayerConfig
from sr2 import CompiledContext, PostProcessResult
from sr2.protocols import (
    ContentProvider,
    ContentReducer,
    MemoryStore,
    MetricExporter,
    ProviderContext,
    ResolvedContent,
    ReducedContent,
    MetricSnapshot,
    LLMClient,
    EmbeddingProvider,
    TokenCounter,
)
from sr2.protocols.llm import Message, CompletionResult
from sr2.tokenization.counting import CharacterCounter, TiktokenCounter, create_token_counter
from sr2.core.errors import (
    SR2Error,
    ConfigError,
    PluginError,
    PluginNotFoundError,
    PluginLicenseError,
    PipelineError,
)


class TestPublicAPI:
    """Verify the public API is importable and coherent."""

    def test_sr2_class_importable(self):
        sr2 = SR2()
        assert sr2 is not None

    def test_turn_result_creation(self):
        result = TurnResult(
            role="assistant",
            content="Hello!",
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )
        assert result.role == "assistant"
        assert result.content == "Hello!"
        assert result.token_usage.total_tokens == 15

    def test_tool_call_creation(self):
        call = ToolCall(tool_name="browser_search", arguments={"query": "test"})
        assert call.tool_name == "browser_search"

    def test_all_exports(self):
        """All items in __all__ are importable."""
        from sr2 import __all__

        for name in __all__:
            # Should be accessible from sr2 module
            obj = __import__("sr2", fromlist=[name])
            assert hasattr(obj, name), f"Missing export: {name}"


class TestProtocols:
    """Verify protocols are runtime-checkable and well-formed."""

    def test_content_provider_is_runtime_checkable(self):
        """ContentProvider should be runtime_checkable."""

        class FakeProvider:
            @property
            def name(self) -> str:
                return "fake"

            async def resolve(self, ctx):
                return ResolvedContent(content="test", tokens=1)

        provider = FakeProvider()
        assert isinstance(provider, ContentProvider)

    def test_content_reducer_is_runtime_checkable(self):

        class FakeReducer:
            @property
            def name(self) -> str:
                return "fake"

            async def reduce(self, content: str, budget: int):
                return ReducedContent(content=content, original_tokens=10, reduced_tokens=5)

        reducer = FakeReducer()
        assert isinstance(reducer, ContentReducer)

    def test_value_objects_frozen(self):
        """Value objects should be immutable."""
        ctx = ProviderContext(session_id="s1", layer_name="l1")
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.session_id = "changed"

    def test_resolved_content_defaults(self):
        content = ResolvedContent(content="hello", tokens=1)
        assert content.metadata == {}

    def test_metric_snapshot_timestamp(self):
        snapshot = MetricSnapshot(turn_id="t1")
        assert snapshot.timestamp is not None


class TestConfigModels:
    """Verify Pydantic config models validate correctly."""

    def test_minimal_pipeline_config(self):
        config = PipelineConfig(layers=[])
        assert config.layers == []
        assert config.total_budget is None

    def test_layer_config_minimal(self):
        layer = LayerConfig(name="test")
        assert layer.name == "test"
        assert layer.priority == 50  # default

    def test_layer_config_with_providers(self):
        from sr2.config.models import ProviderConfig

        layer = LayerConfig(
            name="conversation",
            max_tokens=8000,
            window=10,
            session_history=ProviderConfig(max_tokens=8000),
        )
        assert layer.name == "conversation"
        assert layer.window == 10
        assert layer.session_history.max_tokens == 8000

    def test_layer_config_with_compaction(self):
        from sr2.config.models import CompactionConfig

        layer = LayerConfig(
            name="conversation",
            compaction=CompactionConfig(rules=["schema_and_sample"]),
        )
        assert layer.compaction.rules == ["schema_and_sample"]

    def test_config_validation_rejects_missing_name(self):
        """LayerConfig requires a name."""
        with pytest.raises(Exception):  # ValidationError
            LayerConfig()  # type: ignore[arg-type]


class TestErrorHierarchy:
    """Verify error hierarchy is correct."""

    def test_all_errors_inherit_sr2_error(self):
        assert issubclass(ConfigError, SR2Error)
        assert issubclass(PluginError, SR2Error)
        assert issubclass(PipelineError, SR2Error)

    def test_plugin_subclass_hierarchy(self):
        assert issubclass(PluginNotFoundError, PluginError)
        assert issubclass(PluginLicenseError, PluginError)

    def test_plugin_not_found_message(self):
        err = PluginNotFoundError('Plugin "postgres" not found')
        assert "postgres" in str(err)

    def test_plugin_license_message(self):
        err = PluginLicenseError("Invalid license")
        assert "Invalid license" in str(err)


class TestFacadeInit:
    """Verify SR2 facade instantiates correctly."""

    def test_default_init(self):
        sr2 = SR2()
        assert sr2 is not None

    def test_init_with_pipeline_config(self):
        config = PipelineConfig(layers=[LayerConfig(name="system_prompt")])
        sr2 = SR2(config=config)
        assert sr2 is not None

    def test_init_with_dict(self):
        sr2 = SR2(config={"layers": [{"name": "system_prompt"}]})
        assert sr2 is not None


# ---------------------------------------------------------------------------
# Mock implementations for dependency injection tests
# ---------------------------------------------------------------------------


class _MockLLMClient:
    """Minimal LLMClient implementation for testing."""

    async def complete(
        self,
        messages: list[Message],
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        return CompletionResult(content="mock response")


class _MockEmbeddingProvider:
    """Minimal EmbeddingProvider implementation for testing."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 128 for _ in texts]

    @property
    def dimensions(self) -> int:
        return 128


class TestFacadeDependencyInjection:
    """Verify SR2 facade accepts optional dependency injection params."""

    def test_init_with_llm_client(self):
        """SR2(llm=mock_llm) stores the client without error."""
        llm = _MockLLMClient()
        sr2 = SR2(llm=llm)
        assert sr2._llm is llm

    def test_init_with_token_counter(self):
        """SR2(token_counter=CharacterCounter()) stores the counter."""
        counter = CharacterCounter()
        sr2 = SR2(token_counter=counter)
        assert sr2._token_counter is counter

    def test_init_with_embedding_provider(self):
        """SR2(embedding_provider=mock_embedder) stores the provider."""
        embedder = _MockEmbeddingProvider()
        sr2 = SR2(embedding_provider=embedder)
        assert sr2._embedding_provider is embedder

    def test_init_with_all_deps(self):
        """All three deps at once works."""
        llm = _MockLLMClient()
        counter = CharacterCounter()
        embedder = _MockEmbeddingProvider()
        sr2 = SR2(llm=llm, token_counter=counter, embedding_provider=embedder)
        assert sr2._llm is llm
        assert sr2._token_counter is counter
        assert sr2._embedding_provider is embedder

    def test_init_defaults_to_none(self):
        """SR2() still works — backward compat. LLM and embedder default to None."""
        sr2 = SR2()
        assert sr2._llm is None
        assert sr2._embedding_provider is None

    def test_default_token_counter(self):
        """When no token_counter provided, SR2 creates a default one."""
        sr2 = SR2()
        assert sr2._token_counter is not None
        # Must be either TiktokenCounter or CharacterCounter
        assert isinstance(sr2._token_counter, (TiktokenCounter, CharacterCounter))
