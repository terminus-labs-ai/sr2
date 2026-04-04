"""Deterministic context serialization for KV-cache prefix stability."""

import hashlib


class ContextSerializer:
    """Ensures byte-identical output for unchanged content.

    Normalizes whitespace and ordering so that semantically identical
    context produces the same string across invocations.
    """

    def serialize_layer(self, items: list) -> str:
        """Serialize a list of ResolvedContent items to a deterministic string.

        Strips trailing whitespace from each item, skips empties,
        and joins with newline.
        """
        parts = []
        for item in items:
            text = item.content.rstrip()
            if text:
                parts.append(text)
        return "\n".join(parts)

    def serialize_context(self, layers: dict[str, str]) -> str:
        """Join serialized layer sections with double newline.

        Args:
            layers: dict mapping layer name to already-serialized layer string.
        """
        sections = []
        for layer_str in layers.values():
            stripped = layer_str.rstrip()
            if stripped:
                sections.append(stripped)
        return "\n\n".join(sections)

    def hash_content(self, content: str) -> str:
        """SHA-256 hash truncated to 16 hex chars."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
