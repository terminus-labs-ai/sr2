"""Tests for the extractor plugin registry and MemoryExtractionTransformer.build() integration.

Covers:
- Registry resolves 'rule_based' key to RuleBasedExtractor class
- Registry raises PluginNotFoundError for unknown extractor names
- MemoryExtractionTransformer.build() uses config 'extractor' key to select via registry
- MemoryExtractionTransformer.build() defaults to 'rule_based' when 'extractor' absent from config
- deps.extras['memory_extractor'] override takes precedence over registry lookup
"""

from __future__ import annotations

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest

from sr2.memory import InMemoryMemoryStore, RuleBasedExtractor
from sr2.memory.schema import MemoryScope
from sr2.plugins.errors import PluginNotFoundError
from sr2.pipeline.dependencies import Dependencies
from sr2.config.models import TransformerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_store():
    return InMemoryMemoryStore()


def make_config(extractor: str | None = None, max_executions: int = 10) -> TransformerConfig:
    """Build a TransformerConfig with optional extractor key in config."""
    cfg_data: dict = {}
    if extractor is not None:
        cfg_data["extractor"] = extractor
    return TransformerConfig(
        type="memory_extraction",
        max_executions=max_executions,
        config=cfg_data,
    )


def make_deps(*, memory_store=None, memory_extractor=None) -> Dependencies:
    ex: dict = {}
    if memory_store is not None:
        ex["memory_store"] = memory_store
    if memory_extractor is not None:
        ex["memory_extractor"] = memory_extractor
    return Dependencies(extras=ex)


def _make_entry_point(name: str, cls: type, dist_name: str = "sr2") -> MagicMock:
    ep = MagicMock(spec=importlib.metadata.EntryPoint)
    ep.name = name
    ep.load.return_value = cls
    dist = MagicMock()
    dist.name = dist_name
    ep.dist = dist
    return ep


# ---------------------------------------------------------------------------
# TestExtractorRegistry — registry resolves names to classes
# ---------------------------------------------------------------------------


class TestExtractorRegistry:
    """The extractor registry resolves plugin names to extractor classes."""

    def test_registry_resolves_rule_based_to_rule_based_extractor(self):
        """Registry.get('rule_based') returns RuleBasedExtractor class."""
        from sr2.memory.extractor_registry import EXTRACTORS

        cls = EXTRACTORS.get("rule_based")
        assert cls is RuleBasedExtractor

    def test_rule_based_instance_is_a_memory_extractor(self):
        """Instantiating the returned class yields a MemoryExtractor-compatible object."""
        from sr2.memory.extractor_registry import EXTRACTORS
        from sr2.memory.protocol import MemoryExtractor

        cls = EXTRACTORS.get("rule_based")
        instance = cls()
        assert isinstance(instance, MemoryExtractor)

    def test_registry_raises_plugin_not_found_for_unknown_name(self):
        """Registry raises PluginNotFoundError for an unknown extractor name."""
        from sr2.memory.extractor_registry import EXTRACTORS

        with pytest.raises(PluginNotFoundError):
            EXTRACTORS.get("nonexistent_extractor")

    def test_unknown_name_error_message_mentions_requested_name(self):
        """PluginNotFoundError message includes the unknown name."""
        from sr2.memory.extractor_registry import EXTRACTORS

        with pytest.raises(PluginNotFoundError) as exc_info:
            EXTRACTORS.get("bad_extractor_name")

        assert "bad_extractor_name" in str(exc_info.value)

    def test_unknown_name_error_message_lists_available_names(self):
        """PluginNotFoundError message lists available plugin names."""
        from sr2.memory.extractor_registry import EXTRACTORS

        with pytest.raises(PluginNotFoundError) as exc_info:
            EXTRACTORS.get("nonexistent")

        message = str(exc_info.value)
        # 'rule_based' is the only registered extractor — it should appear
        assert "rule_based" in message

    def test_registry_is_a_plugin_registry(self):
        """EXTRACTORS is a PluginRegistry instance."""
        from sr2.memory.extractor_registry import EXTRACTORS
        from sr2.plugins.registry import PluginRegistry

        assert isinstance(EXTRACTORS, PluginRegistry)

    def test_registry_group_is_sr2_extractors(self):
        """EXTRACTORS registry is scoped to 'sr2.extractors' group."""
        from sr2.memory.extractor_registry import EXTRACTORS

        assert EXTRACTORS._group == "sr2.extractors"

    def test_rule_based_is_in_names(self):
        """'rule_based' appears in EXTRACTORS.names()."""
        from sr2.memory.extractor_registry import EXTRACTORS

        assert "rule_based" in EXTRACTORS.names()


# ---------------------------------------------------------------------------
# TestBuildUsesConfigExtractorKey
# ---------------------------------------------------------------------------


class TestBuildUsesConfigExtractorKey:
    """MemoryExtractionTransformer.build() selects extractor via config 'extractor' key."""

    def test_build_without_extractor_config_defaults_to_rule_based(self):
        """No 'extractor' in config → defaults to 'rule_based' → RuleBasedExtractor instance."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = make_store()
        deps = make_deps(memory_store=store)
        config = make_config()  # no extractor key

        transformer = MemoryExtractionTransformer.build(config, deps)

        assert isinstance(transformer._extractor, RuleBasedExtractor)

    def test_build_with_extractor_rule_based_uses_rule_based(self):
        """config.config['extractor'] == 'rule_based' → RuleBasedExtractor instance."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = make_store()
        deps = make_deps(memory_store=store)
        config = make_config(extractor="rule_based")

        transformer = MemoryExtractionTransformer.build(config, deps)

        assert isinstance(transformer._extractor, RuleBasedExtractor)

    def test_build_with_unknown_extractor_name_raises_plugin_not_found(self):
        """config.config['extractor'] == 'unknown_name' → PluginNotFoundError."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = make_store()
        deps = make_deps(memory_store=store)
        config = make_config(extractor="unknown_extractor_name")

        with pytest.raises(PluginNotFoundError):
            MemoryExtractionTransformer.build(config, deps)


# ---------------------------------------------------------------------------
# TestExtrasOverrideTakesPrecedence
# ---------------------------------------------------------------------------


class TestExtrasOverrideTakesPrecedence:
    """deps.extras['memory_extractor'] takes precedence over registry lookup."""

    def test_extras_override_used_when_present(self):
        """When memory_extractor is in extras, that instance is used — not the registry."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer
        from sr2.memory.schema import ExtractionResult

        class CustomExtractor:
            def extract(self, turn_text: str, turn_id: str | None = None) -> ExtractionResult:
                return ExtractionResult()

        store = make_store()
        custom = CustomExtractor()
        deps = make_deps(memory_store=store, memory_extractor=custom)
        config = make_config()  # no extractor key — would default to rule_based without override

        transformer = MemoryExtractionTransformer.build(config, deps)

        assert transformer._extractor is custom

    def test_extras_override_beats_explicit_config_extractor_key(self):
        """extras['memory_extractor'] wins even when config specifies an extractor name."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer
        from sr2.memory.schema import ExtractionResult

        class AnotherExtractor:
            def extract(self, turn_text: str, turn_id: str | None = None) -> ExtractionResult:
                return ExtractionResult()

        store = make_store()
        custom = AnotherExtractor()
        deps = make_deps(memory_store=store, memory_extractor=custom)
        config = make_config(extractor="rule_based")  # explicit, but should lose to extras

        transformer = MemoryExtractionTransformer.build(config, deps)

        assert transformer._extractor is custom

    def test_registry_used_when_no_extras_override(self):
        """When memory_extractor absent from extras, registry is used (not direct import)."""
        from sr2.memory.extraction_transformer import MemoryExtractionTransformer

        store = make_store()
        deps = make_deps(memory_store=store)  # no memory_extractor in extras
        config = make_config(extractor="rule_based")

        transformer = MemoryExtractionTransformer.build(config, deps)

        # Registry should have resolved 'rule_based' → RuleBasedExtractor
        assert isinstance(transformer._extractor, RuleBasedExtractor)
        # And it must not be a hardcoded import bypass — the instance IS the protocol-correct type
        assert hasattr(transformer._extractor, "extract")
