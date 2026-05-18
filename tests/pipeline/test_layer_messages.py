"""Tests for Layer message compilation with direct Message support.

After Step 2, Layer._content is widened to list[ContentBlock | Message] and
_compile_messages() passes Message objects through directly instead of
wrapping them. These tests verify:

  1. Layer with Message objects compiles correctly (pass-through)
  2. Multiple Messages with different roles compile in insertion order
  3. Empty layer still compiles to empty list (unchanged)
  4. Raw ContentBlocks (no Messages) still wrap into Message(role="user") (backward compat)
"""

import pytest

from sr2.models import Message, TextBlock, ToolResultBlock, ToolUseBlock
from sr2.pipeline.compilation import AppendStrategy
from sr2.pipeline.event_bus import EventBus
from sr2.pipeline.layer import Layer
from sr2.pipeline.models import CompilationTarget, ResolvedContent
from sr2.pipeline.token_counting import CharacterTokenCounter


# ---------------------------------------------------------------------------
# Helper: build a MESSAGES-target layer with no resolvers/transformers
# ---------------------------------------------------------------------------


def make_messages_layer() -> Layer:
    """Create a minimal MESSAGES-target layer for testing compilation."""
    return Layer(
        name="conversation",
        target=CompilationTarget.MESSAGES,
        position=AppendStrategy(),
        token_budget=None,
        resolvers=[],
        transformers=[],
        token_counter=CharacterTokenCounter(),
        event_bus=EventBus(),
    )


# ---------------------------------------------------------------------------
# 1. Layer with Message objects compiles correctly
# ---------------------------------------------------------------------------


class TestLayerMessagePassthrough:
    def test_single_message_passes_through(self):
        """A Message in layer content should be returned directly by
        _compile_messages(), not wrapped in another Message."""
        layer = make_messages_layer()

        msg = Message(role="user", content=[TextBlock(text="hello")])
        layer.add_content(ResolvedContent(
            resolver_name="input",
            source_layer="conversation",
            content=[msg],
        ))

        compiled = layer.compile()
        assert len(compiled) == 1
        assert isinstance(compiled[0], Message)
        assert compiled[0].role == "user"
        assert compiled[0].content[0].text == "hello"

    def test_message_is_same_object_not_rewrapped(self):
        """The compiled Message should be the original object (or equivalent),
        not wrapped inside another Message."""
        layer = make_messages_layer()

        msg = Message(role="assistant", content=[TextBlock(text="response")])
        layer.add_content(ResolvedContent(
            resolver_name="session",
            source_layer="conversation",
            content=[msg],
        ))

        compiled = layer.compile()
        assert len(compiled) == 1
        # Must be a Message with role="assistant", not wrapped in a user Message
        assert compiled[0].role == "assistant"
        assert compiled[0].content[0].text == "response"


# ---------------------------------------------------------------------------
# 2. Multiple Messages preserve order
# ---------------------------------------------------------------------------


class TestLayerMultipleMessages:
    def test_multiple_messages_compile_in_insertion_order(self):
        """Multiple Messages added via add_content compile in order."""
        layer = make_messages_layer()

        msg1 = Message(role="user", content=[TextBlock(text="question")])
        msg2 = Message(role="assistant", content=[TextBlock(text="answer")])
        msg3 = Message(role="user", content=[TextBlock(text="follow-up")])

        layer.add_content(ResolvedContent(
            resolver_name="session",
            source_layer="conversation",
            content=[msg1, msg2, msg3],
        ))

        compiled = layer.compile()
        assert len(compiled) == 3
        assert compiled[0].role == "user"
        assert compiled[0].content[0].text == "question"
        assert compiled[1].role == "assistant"
        assert compiled[1].content[0].text == "answer"
        assert compiled[2].role == "user"
        assert compiled[2].content[0].text == "follow-up"

    def test_messages_from_multiple_add_content_calls(self):
        """Messages added across separate add_content calls compile in order."""
        layer = make_messages_layer()

        msg1 = Message(role="user", content=[TextBlock(text="first")])
        msg2 = Message(role="assistant", content=[TextBlock(text="second")])

        layer.add_content(ResolvedContent(
            resolver_name="r1",
            source_layer="conversation",
            content=[msg1],
        ))
        layer.add_content(ResolvedContent(
            resolver_name="r2",
            source_layer="conversation",
            content=[msg2],
        ))

        compiled = layer.compile()
        assert len(compiled) == 2
        assert compiled[0].role == "user"
        assert compiled[0].content[0].text == "first"
        assert compiled[1].role == "assistant"
        assert compiled[1].content[0].text == "second"


# ---------------------------------------------------------------------------
# 3. Empty layer compiles to empty list
# ---------------------------------------------------------------------------


class TestLayerEmptyCompilation:
    def test_empty_layer_compiles_to_empty_list(self):
        """Unchanged behavior: empty MESSAGES layer -> empty list."""
        layer = make_messages_layer()
        compiled = layer.compile()
        assert compiled == []


# ---------------------------------------------------------------------------
# 4. Backward compatibility — raw ContentBlocks still wrap
# ---------------------------------------------------------------------------


class TestLayerRawContentBlockBackwardCompat:
    def test_raw_text_blocks_wrapped_in_user_message(self):
        """Legacy path: raw TextBlocks (no Messages) wrap into
        Message(role='user') as before."""
        layer = make_messages_layer()

        layer.add_content(ResolvedContent(
            resolver_name="static",
            source_layer="conversation",
            content=[TextBlock(text="raw text")],
        ))

        compiled = layer.compile()
        assert len(compiled) == 1
        assert isinstance(compiled[0], Message)
        assert compiled[0].role == "user"
        assert compiled[0].content[0].text == "raw text"

    def test_multiple_raw_blocks_wrapped_in_single_message(self):
        """Multiple raw ContentBlocks should wrap into a single user Message
        (preserving current behavior)."""
        layer = make_messages_layer()

        layer.add_content(ResolvedContent(
            resolver_name="static",
            source_layer="conversation",
            content=[
                TextBlock(text="line 1"),
                TextBlock(text="line 2"),
            ],
        ))

        compiled = layer.compile()
        assert len(compiled) == 1
        assert isinstance(compiled[0], Message)
        assert compiled[0].role == "user"
        assert len(compiled[0].content) == 2

    def test_mixed_raw_blocks_and_messages(self):
        """When content has both raw ContentBlocks and Messages,
        raw blocks are grouped into user Messages and Messages pass through."""
        layer = make_messages_layer()

        layer.add_content(ResolvedContent(
            resolver_name="static",
            source_layer="conversation",
            content=[
                TextBlock(text="preamble"),
                Message(role="user", content=[TextBlock(text="question")]),
                TextBlock(text="epilogue"),
            ],
        ))

        compiled = layer.compile()
        assert len(compiled) == 3
        # raw block -> wrapped in Message(role="user")
        assert compiled[0].role == "user"
        assert compiled[0].content[0].text == "preamble"
        # Message passes through
        assert compiled[1].role == "user"
        assert compiled[1].content[0].text == "question"
        # raw block -> wrapped in Message(role="user")
        assert compiled[2].role == "user"
        assert compiled[2].content[0].text == "epilogue"
