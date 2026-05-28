"""Tests for sr2.pipeline.compilation — CompilationTarget, target inference, PositionStrategy.

Covers:
  FR14: CompilationTarget enum and layer-name-based inference with explicit override
  FR15: PositionStrategy protocol with PrefixStrategy and AppendStrategy built-ins
  FR20: Protocol-based — new position strategies addable without modifying engine code
"""

import pytest

from sr2.models import Message, TextBlock


# ---------------------------------------------------------------------------
# 1. CompilationTarget — enum values
# ---------------------------------------------------------------------------


class TestCompilationTarget:
    def test_has_system(self):
        from sr2.pipeline.models import CompilationTarget

        assert CompilationTarget.SYSTEM.value == "system"

    def test_has_messages(self):
        from sr2.pipeline.models import CompilationTarget

        assert CompilationTarget.MESSAGES.value == "messages"

    def test_has_tools(self):
        from sr2.pipeline.models import CompilationTarget

        assert CompilationTarget.TOOLS.value == "tools"

    def test_exactly_three_values(self):
        """No silent additions — exactly SYSTEM, MESSAGES, TOOLS."""
        from sr2.pipeline.models import CompilationTarget

        assert set(CompilationTarget) == {
            CompilationTarget.SYSTEM,
            CompilationTarget.MESSAGES,
            CompilationTarget.TOOLS,
        }


# ---------------------------------------------------------------------------
# 2. infer_compilation_target — removed (sr2-40 OCP cleanup)
# ---------------------------------------------------------------------------


class TestInferCompilationTargetRemoved:
    """FR: infer_compilation_target magic-string heuristic has been deleted.

    Target must be set explicitly on LayerConfig. These tests verify the old
    function no longer exists and that LayerConfig enforces explicit target.
    """

    def test_infer_compilation_target_not_importable(self):
        """infer_compilation_target must no longer exist in sr2.pipeline.models."""
        import sr2.pipeline.models as models_mod

        assert not hasattr(models_mod, "infer_compilation_target")

    def test_infer_compilation_target_not_in_package_init(self):
        """infer_compilation_target must not be re-exported from sr2.pipeline."""
        import sr2.pipeline as pipeline_pkg

        assert not hasattr(pipeline_pkg, "infer_compilation_target")

    def test_layer_config_target_required(self):
        """LayerConfig.target is now a required field — omitting it raises ValidationError."""
        from pydantic import ValidationError
        from sr2.config.models import LayerConfig, ResolverConfig

        with pytest.raises(ValidationError):
            LayerConfig(name="system", resolvers=[ResolverConfig(type="input")])

    def test_layer_config_target_system(self):
        """LayerConfig accepts target='system' explicitly."""
        from sr2.config.models import LayerConfig, ResolverConfig
        from sr2.pipeline.models import CompilationTarget

        cfg = LayerConfig(name="system", target="system", resolvers=[])
        assert CompilationTarget(cfg.target) == CompilationTarget.SYSTEM

    def test_layer_config_target_messages(self):
        """LayerConfig accepts target='messages' explicitly."""
        from sr2.config.models import LayerConfig
        from sr2.pipeline.models import CompilationTarget

        cfg = LayerConfig(name="conversation", target="messages", resolvers=[])
        assert CompilationTarget(cfg.target) == CompilationTarget.MESSAGES

    def test_layer_config_target_tools(self):
        """LayerConfig accepts target='tools' explicitly."""
        from sr2.config.models import LayerConfig
        from sr2.pipeline.models import CompilationTarget

        cfg = LayerConfig(name="tools", target="tools", resolvers=[])
        assert CompilationTarget(cfg.target) == CompilationTarget.TOOLS


# ---------------------------------------------------------------------------
# 4. PositionStrategy protocol — runtime checkable
# ---------------------------------------------------------------------------


class TestPositionStrategyProtocol:
    def test_is_runtime_checkable(self):
        """FR20: Protocol-based — isinstance checks must work."""
        from sr2.pipeline.compilation import PositionStrategy

        # runtime_checkable protocols support isinstance
        assert hasattr(PositionStrategy, "__protocol_attrs__") or hasattr(
            PositionStrategy, "__abstractmethods__"
        ) or isinstance(PositionStrategy, type)

    def test_conforming_class_satisfies_protocol(self):
        """FR20: A class with the right place() method is a valid PositionStrategy."""
        from sr2.pipeline.compilation import PositionStrategy

        class MyStrategy:
            def place(self, existing: list, new: list) -> list:
                return existing + new

        assert isinstance(MyStrategy(), PositionStrategy)

    def test_non_conforming_class_does_not_satisfy(self):
        """A class without place() must not satisfy the protocol."""
        from sr2.pipeline.compilation import PositionStrategy

        class NotAStrategy:
            def arrange(self, existing: list, new: list) -> list:
                return existing + new

        assert not isinstance(NotAStrategy(), PositionStrategy)


# ---------------------------------------------------------------------------
# 5. AppendStrategy — appends new after existing (default)
# ---------------------------------------------------------------------------


class TestAppendStrategy:
    def test_satisfies_position_strategy(self):
        from sr2.pipeline.compilation import AppendStrategy, PositionStrategy

        assert isinstance(AppendStrategy(), PositionStrategy)

    def test_appends_new_after_existing(self):
        from sr2.pipeline.compilation import AppendStrategy

        strategy = AppendStrategy()
        result = strategy.place(["a", "b"], ["c", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_empty_existing_returns_new(self):
        from sr2.pipeline.compilation import AppendStrategy

        strategy = AppendStrategy()
        result = strategy.place([], ["x", "y"])
        assert result == ["x", "y"]

    def test_empty_new_returns_existing(self):
        from sr2.pipeline.compilation import AppendStrategy

        strategy = AppendStrategy()
        result = strategy.place(["a", "b"], [])
        assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# 6. PrefixStrategy — prepends new before existing
# ---------------------------------------------------------------------------


class TestPrefixStrategy:
    def test_satisfies_position_strategy(self):
        from sr2.pipeline.compilation import PositionStrategy, PrefixStrategy

        assert isinstance(PrefixStrategy(), PositionStrategy)

    def test_prepends_new_before_existing(self):
        from sr2.pipeline.compilation import PrefixStrategy

        strategy = PrefixStrategy()
        result = strategy.place(["a", "b"], ["c", "d"])
        assert result == ["c", "d", "a", "b"]

    def test_empty_existing_returns_new(self):
        from sr2.pipeline.compilation import PrefixStrategy

        strategy = PrefixStrategy()
        result = strategy.place([], ["x", "y"])
        assert result == ["x", "y"]

    def test_empty_new_returns_existing(self):
        from sr2.pipeline.compilation import PrefixStrategy

        strategy = PrefixStrategy()
        result = strategy.place(["a", "b"], [])
        assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# 7. Strategies work with various types (generic)
# ---------------------------------------------------------------------------


class TestStrategiesWithVariousTypes:
    """Both strategies are generic — they should work with any list element type."""

    def test_append_with_strings(self):
        from sr2.pipeline.compilation import AppendStrategy

        result = AppendStrategy().place(["hello"], ["world"])
        assert result == ["hello", "world"]

    def test_prefix_with_strings(self):
        from sr2.pipeline.compilation import PrefixStrategy

        result = PrefixStrategy().place(["hello"], ["world"])
        assert result == ["world", "hello"]

    def test_append_with_message_objects(self):
        from sr2.pipeline.compilation import AppendStrategy

        msg1 = Message(role="user", content=[TextBlock(text="hi")])
        msg2 = Message(role="assistant", content=[TextBlock(text="hello")])
        result = AppendStrategy().place([msg1], [msg2])
        assert len(result) == 2
        assert result[0] is msg1
        assert result[1] is msg2

    def test_prefix_with_message_objects(self):
        from sr2.pipeline.compilation import PrefixStrategy

        msg1 = Message(role="user", content=[TextBlock(text="hi")])
        msg2 = Message(role="assistant", content=[TextBlock(text="hello")])
        result = PrefixStrategy().place([msg1], [msg2])
        assert len(result) == 2
        assert result[0] is msg2
        assert result[1] is msg1

    def test_append_with_text_blocks(self):
        from sr2.pipeline.compilation import AppendStrategy

        b1 = TextBlock(text="first")
        b2 = TextBlock(text="second")
        result = AppendStrategy().place([b1], [b2])
        assert len(result) == 2
        assert result[0] is b1
        assert result[1] is b2

    def test_prefix_with_text_blocks(self):
        from sr2.pipeline.compilation import PrefixStrategy

        b1 = TextBlock(text="first")
        b2 = TextBlock(text="second")
        result = PrefixStrategy().place([b1], [b2])
        assert len(result) == 2
        assert result[0] is b2
        assert result[1] is b1
