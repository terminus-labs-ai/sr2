"""Extractor plugin registry for the sr2.extractors entry-point group.

Provides a lazy-discovery registry for MemoryExtractor implementations.
Plugins register themselves via the 'sr2.extractors' entry-point group in
their package's pyproject.toml.

Usage::

    from sr2.memory.extractor_registry import EXTRACTORS

    cls = EXTRACTORS.get("rule_based")   # → RuleBasedExtractor
    instance = cls()
"""

from __future__ import annotations

from sr2.memory.protocol import MemoryExtractor
from sr2.plugins.registry import PluginRegistry

# Singleton registry for the sr2.extractors group.
# Uses object as the protocol argument — MemoryExtractor defines instance
# attributes, so class-level isinstance checks yield false negatives.
# Correctness is enforced at build time when the extractor is instantiated.
EXTRACTORS: PluginRegistry = PluginRegistry("sr2.extractors", object)
