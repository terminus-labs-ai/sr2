"""Tests for deterministic context serialization."""

from sr2.pipeline.serializer import ContextSerializer
from sr2.resolvers.registry import ResolvedContent


def _rc(content: str, key: str = "k") -> ResolvedContent:
    return ResolvedContent(key=key, content=content, tokens=len(content) // 4)


class TestContextSerializer:
    def test_same_items_identical_output(self):
        """Same items produce identical output on repeated calls."""
        s = ContextSerializer()
        items = [_rc("hello world"), _rc("second item")]
        assert s.serialize_layer(items) == s.serialize_layer(items)

    def test_strips_trailing_whitespace(self):
        """Trailing whitespace is stripped from each item."""
        s = ContextSerializer()
        items = [_rc("hello   "), _rc("world\t\t")]
        result = s.serialize_layer(items)
        assert result == "hello\nworld"

    def test_skips_empties(self):
        """Empty and whitespace-only items are skipped."""
        s = ContextSerializer()
        items = [_rc("content"), _rc(""), _rc("   "), _rc("more")]
        result = s.serialize_layer(items)
        assert result == "content\nmore"

    def test_serialize_context_joins_layers(self):
        """Layers are joined with double newline."""
        s = ContextSerializer()
        layers = {"core": "system prompt", "memory": "user likes python"}
        result = s.serialize_context(layers)
        assert result == "system prompt\n\nuser likes python"

    def test_serialize_context_skips_empty_layers(self):
        """Empty layer strings are skipped."""
        s = ContextSerializer()
        layers = {"core": "prompt", "empty": "", "data": "facts"}
        result = s.serialize_context(layers)
        assert result == "prompt\n\nfacts"

    def test_hash_content_deterministic(self):
        """hash_content is deterministic for identical input."""
        s = ContextSerializer()
        assert s.hash_content("hello") == s.hash_content("hello")

    def test_hash_content_different_for_different_input(self):
        """hash_content differs for different content."""
        s = ContextSerializer()
        assert s.hash_content("hello") != s.hash_content("world")

    def test_hash_content_length(self):
        """hash_content returns 16 hex chars."""
        s = ContextSerializer()
        h = s.hash_content("test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
