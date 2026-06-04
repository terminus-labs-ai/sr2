"""Tests for sr2-74: Unified compile dispatch through TargetCompiler registry.

Verifies that Layer.compile() routes through the same TargetCompiler registry
used by Engine._compile_request(), eliminating the duplicate _COMPILE_DISPATCH
table. Also verifies that the compile logic (type narrowing) lives in the
registry implementations.
"""

import pytest

from sr2.models import TextBlock
from sr2.pipeline.token_counting import CharacterTokenCounter


class TestRegistryIsSingleSourceOfTruth:
    """The TargetCompiler registry is the sole dispatch mechanism for compilation."""

    def test_layer_compile_uses_registry(self):
        """Layer.compile() delegates to get_compilation_targets(), not a local dict."""
        import sr2.pipeline.layer as layer_mod

        # Verify Layer.compile() references get_compilation_targets
        source = layer_mod.Layer.compile.__code__.co_names
        assert "get_compilation_targets" in source, (
            "Layer.compile() must use the registry, not a local _COMPILE_DISPATCH dict"
        )

    def test_no_compile_dispatch_in_layer_source(self):
        """Layer module must not define _COMPILE_DISPATCH."""
        import sr2.pipeline.layer as layer_mod
        import inspect

        source = inspect.getsource(layer_mod.Layer)
        assert "_COMPILE_DISPATCH" not in source, (
            "Layer must not have a _COMPILE_DISPATCH dict — use the registry"
        )

    def test_no_private_compile_methods_on_layer(self):
        """Layer must not have _compile_system/_compile_messages/_compile_tools methods."""
        from sr2.pipeline.layer import Layer

        for method_name in ("_compile_system", "_compile_messages", "_compile_tools"):
            assert not hasattr(Layer, method_name), (
                f"Layer must not have {method_name} — compile logic lives in TargetCompiler"
            )


class TestTargetCompilerProtocol:
    """TargetCompiler.collect() receives raw content, not pre-compiled output."""

    def test_collect_signature_accepts_raw_content(self):
        """collect() takes content + tool_definitions as separate parameters."""
        from sr2.pipeline.compilation import TargetCompiler

        # The protocol should have collect with 5 parameters (self, content,
        # tool_definitions, system_blocks, messages, tools)
        import inspect
        sig = inspect.signature(TargetCompiler.collect)
        params = list(sig.parameters.keys())
        assert "content" in params, "collect() must accept 'content' parameter"
        assert "tool_definitions" in params, "collect() must accept 'tool_definitions' parameter"

    def test_collect_mutates_correct_accumulator(self):
        """Each collector only mutates its target accumulator."""
        from sr2.models import Message, ToolDefinition
        from sr2.pipeline.compilation import get_compilation_targets
        from sr2.pipeline.models import CompilationTarget

        targets = get_compilation_targets()

        # SYSTEM collector
        system_out: list[TextBlock] = []
        messages_out: list = []
        tools_out: list = []
        targets[CompilationTarget.SYSTEM].collect(
            [TextBlock(text="sys")], [], system_out, messages_out, tools_out
        )
        assert len(system_out) == 1
        assert messages_out == []
        assert tools_out == []

        # MESSAGES collector
        system_out = []
        messages_out = []
        tools_out = []
        targets[CompilationTarget.MESSAGES].collect(
            [Message(role="user", content=[TextBlock(text="hi")])], [],
            system_out, messages_out, tools_out
        )
        assert system_out == []
        assert len(messages_out) == 1
        assert tools_out == []

        # TOOLS collector
        system_out = []
        messages_out = []
        tools_out = []
        targets[CompilationTarget.TOOLS].collect(
            [], [ToolDefinition(name="test", input_schema={})],
            system_out, messages_out, tools_out
        )
        assert system_out == []
        assert messages_out == []
        assert len(tools_out) == 1


class TestLayerCompileContract:
    """Layer.compile() still returns typed output per target."""

    def test_compile_returns_typed_system(self):
        """SYSTEM target compiles to list[TextBlock]."""
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.models import CompilationTarget, ResolvedContent

        layer = Layer(
            name="system",
            target=CompilationTarget.SYSTEM,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        layer.add_content(ResolvedContent(resolver_name="test", source_layer="system", content=[TextBlock(text="test")]))

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert all(isinstance(b, TextBlock) for b in compiled)

    def test_compile_returns_typed_tools(self):
        """TOOLS target compiles to list[ToolDefinition]."""
        from sr2.pipeline.token_counting import CharacterTokenCounter
        from sr2.models import ToolDefinition
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.models import CompilationTarget

        layer = Layer(
            name="tools",
            target=CompilationTarget.TOOLS,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        layer.add_tool_definitions([ToolDefinition(name="search", input_schema={})])

        compiled = layer.compile()
        assert isinstance(compiled, list)
        assert all(isinstance(t, ToolDefinition) for t in compiled)


class TestEngineUsesRegistryDirectly:
    """Engine._compile_request() passes raw content to the registry, skipping layer.compile()."""

    def test_engine_passes_raw_content_to_collector(self):
        """Engine calls collect(layer.get_content(), layer.get_tool_definitions(), ...)"""
        import inspect
        from sr2.pipeline.engine import PipelineEngine

        source = inspect.getsource(PipelineEngine._compile_request)
        # Must NOT call layer.compile() — it passes raw content directly
        assert "layer.compile()" not in source, (
            "Engine must pass raw content to collectors, not call layer.compile()"
        )
        # Must use get_content() and get_tool_definitions()
        assert "get_content()" in source
        assert "get_tool_definitions()" in source


class TestLayerGetToolDefinitions:
    """Layer exposes get_tool_definitions() as a public accessor."""

    def test_get_tool_definitions_public(self):
        """get_tool_definitions() is a public method returning a copy."""
        from sr2.pipeline.token_counting import CharacterTokenCounter
        from sr2.models import ToolDefinition
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.models import CompilationTarget

        layer = Layer(
            name="tools",
            target=CompilationTarget.TOOLS,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        layer.add_tool_definitions([ToolDefinition(name="search", input_schema={})])

        defs = layer.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0].name == "search"

        # Returns a copy, not the internal list
        assert defs is not layer._tool_definitions

    def test_get_tool_definitions_empty(self):
        """Empty layer returns empty list."""
        from sr2.pipeline.token_counting import CharacterTokenCounter
        from sr2.pipeline.compilation import AppendStrategy
        from sr2.pipeline.layer import Layer
        from sr2.pipeline.models import CompilationTarget

        layer = Layer(
            name="tools",
            target=CompilationTarget.TOOLS,
            position=AppendStrategy(),
            token_budget=None,
            resolvers=[],
            transformers=[],
            token_counter=CharacterTokenCounter(),
        )
        assert layer.get_tool_definitions() == []
