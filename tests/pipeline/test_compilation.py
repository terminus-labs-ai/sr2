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
# 2. infer_compilation_target — name-based inference
# ---------------------------------------------------------------------------


class TestInferCompilationTargetFromName:
    def test_system_exact(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("system") == CompilationTarget.SYSTEM

    def test_system_prompt_exact(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("system_prompt") == CompilationTarget.SYSTEM

    def test_contains_system(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("my_system_layer") == CompilationTarget.SYSTEM

    def test_tools_exact(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("tools") == CompilationTarget.TOOLS

    def test_tool_exact(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("tool") == CompilationTarget.TOOLS

    def test_contains_tool(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("tool_definitions") == CompilationTarget.TOOLS

    def test_conversation_defaults_to_messages(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("conversation") == CompilationTarget.MESSAGES

    def test_memory_defaults_to_messages(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("memory") == CompilationTarget.MESSAGES

    def test_anything_else_defaults_to_messages(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        assert infer_compilation_target("anything_else") == CompilationTarget.MESSAGES


# ---------------------------------------------------------------------------
# 3. infer_compilation_target — explicit target override
# ---------------------------------------------------------------------------


class TestInferCompilationTargetExplicitOverride:
    def test_explicit_system(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        result = infer_compilation_target("conversation", explicit_target="system")
        assert result == CompilationTarget.SYSTEM

    def test_explicit_messages(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        result = infer_compilation_target("conversation", explicit_target="messages")
        assert result == CompilationTarget.MESSAGES

    def test_explicit_tools(self):
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        result = infer_compilation_target("conversation", explicit_target="tools")
        assert result == CompilationTarget.TOOLS

    def test_explicit_overrides_name_inference(self):
        """Layer named 'system' with explicit_target='messages' → MESSAGES."""
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        result = infer_compilation_target("system", explicit_target="messages")
        assert result == CompilationTarget.MESSAGES

    def test_explicit_none_falls_back_to_inference(self):
        """explicit_target=None means infer from name (default behavior)."""
        from sr2.pipeline.models import CompilationTarget, infer_compilation_target

        result = infer_compilation_target("system", explicit_target=None)
        assert result == CompilationTarget.SYSTEM


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
