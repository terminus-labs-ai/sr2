"""Ensure BridgeMemoryConfig does not duplicate SR2's RetrievalConfig fields.

SR2's RetrievalConfig (in PipelineConfig) is the single source of truth for
retrieval semantics (strategy, top_k, max_tokens). BridgeMemoryConfig must
only contain infrastructure concerns: enabled, backend, db_path, database_url.

The four fields removed in audit fix-06:
  - retrieval_top_k         (dead code; diverged from SR2 default)
  - retrieval_max_tokens    (dead code; diverged from SR2 default)
  - retrieval_strategy      (dead code; diverged from SR2 default)
  - max_memories_per_turn   (dead code; owned by MemoryExtractor)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sr2.config.models import PipelineConfig, RetrievalConfig
from sr2_bridge.config import BridgeConfig, BridgeMemoryConfig


# ---------------------------------------------------------------------------
# Dead field deletion — these fields must NOT exist after the fix
# ---------------------------------------------------------------------------

_DELETED_FIELDS = {
    "retrieval_top_k",
    "retrieval_max_tokens",
    "retrieval_strategy",
    "max_memories_per_turn",
}


class TestDeletedFields:
    """BridgeMemoryConfig must not define the four deleted fields."""

    def test_retrieval_top_k_not_in_model_fields(self):
        """retrieval_top_k was dead code and must be gone."""
        assert "retrieval_top_k" not in BridgeMemoryConfig.model_fields, (
            "retrieval_top_k still exists in BridgeMemoryConfig. "
            "Retrieval top_k belongs in PipelineConfig.retrieval.top_k."
        )

    def test_retrieval_max_tokens_not_in_model_fields(self):
        """retrieval_max_tokens was dead code and must be gone."""
        assert "retrieval_max_tokens" not in BridgeMemoryConfig.model_fields, (
            "retrieval_max_tokens still exists in BridgeMemoryConfig. "
            "Retrieval max_tokens belongs in PipelineConfig.retrieval.max_tokens."
        )

    def test_retrieval_strategy_not_in_model_fields(self):
        """retrieval_strategy was dead code (and used wrong default 'keyword' vs SR2's 'hybrid')."""
        assert "retrieval_strategy" not in BridgeMemoryConfig.model_fields, (
            "retrieval_strategy still exists in BridgeMemoryConfig. "
            "Retrieval strategy belongs in PipelineConfig.retrieval.strategy."
        )

    def test_max_memories_per_turn_not_in_model_fields(self):
        """max_memories_per_turn was dead code and must be gone."""
        assert "max_memories_per_turn" not in BridgeMemoryConfig.model_fields, (
            "max_memories_per_turn still exists in BridgeMemoryConfig. "
            "This is owned by MemoryExtractor, not bridge config."
        )

    def test_no_deleted_fields_exist(self):
        """Comprehensive check — none of the four deleted fields must be present."""
        present = _DELETED_FIELDS & set(BridgeMemoryConfig.model_fields.keys())
        assert not present, (
            f"BridgeMemoryConfig still defines dead retrieval fields: {present}. "
            f"These belong in PipelineConfig.retrieval (or MemoryExtractor)."
        )


# ---------------------------------------------------------------------------
# Legitimate fields must remain
# ---------------------------------------------------------------------------

class TestLegitimateFieldsRetained:
    """The four infrastructure fields must not be accidentally deleted."""

    def test_enabled_retained(self):
        assert "enabled" in BridgeMemoryConfig.model_fields

    def test_backend_retained(self):
        assert "backend" in BridgeMemoryConfig.model_fields

    def test_db_path_retained(self):
        assert "db_path" in BridgeMemoryConfig.model_fields

    def test_database_url_retained(self):
        assert "database_url" in BridgeMemoryConfig.model_fields

    def test_only_infrastructure_fields_present(self):
        """BridgeMemoryConfig must contain only these four fields (no extras)."""
        expected = {"enabled", "backend", "db_path", "database_url"}
        actual = set(BridgeMemoryConfig.model_fields.keys())
        unexpected = actual - expected
        assert not unexpected, (
            f"BridgeMemoryConfig has unexpected extra fields: {unexpected}. "
            f"Only infrastructure fields are allowed: {expected}."
        )

    def test_defaults_still_work(self):
        """Constructing with no args must succeed after field removal."""
        cfg = BridgeMemoryConfig()
        assert cfg.enabled is False
        assert cfg.backend == "sqlite"
        assert cfg.db_path == "sr2_bridge_memory.db"
        assert cfg.database_url is None


# ---------------------------------------------------------------------------
# Enforcement validator — model_validator must reject the deleted field names
# ---------------------------------------------------------------------------

class TestEnforcementValidator:
    """model_validator(mode='before') must reject any attempt to pass the deleted fields."""

    def test_rejects_retrieval_top_k(self):
        """Passing retrieval_top_k must raise ValidationError with informative message."""
        with pytest.raises(ValidationError) as exc_info:
            BridgeMemoryConfig(retrieval_top_k=5)
        error_text = str(exc_info.value)
        # Message should mention where the config actually belongs
        assert "retrieval" in error_text.lower() or "pipeline" in error_text.lower(), (
            f"ValidationError message should mention the correct config location. Got: {error_text}"
        )

    def test_rejects_retrieval_max_tokens(self):
        """Passing retrieval_max_tokens must raise ValidationError."""
        with pytest.raises(ValidationError):
            BridgeMemoryConfig(retrieval_max_tokens=1000)

    def test_rejects_retrieval_strategy(self):
        """Passing retrieval_strategy must raise ValidationError."""
        with pytest.raises(ValidationError):
            BridgeMemoryConfig(retrieval_strategy="keyword")

    def test_rejects_max_memories_per_turn(self):
        """Passing max_memories_per_turn must raise ValidationError."""
        with pytest.raises(ValidationError):
            BridgeMemoryConfig(max_memories_per_turn=3)

    def test_rejects_unknown_retrieval_prefixed_field(self):
        """Any field starting with 'retrieval_' must be rejected as an architecture violation."""
        with pytest.raises(ValidationError):
            BridgeMemoryConfig(**{"retrieval_something_new": 42})

    def test_valid_config_not_rejected(self):
        """Validator must not fire on legitimate infra fields."""
        cfg = BridgeMemoryConfig(enabled=True, backend="sqlite", db_path="/tmp/test.db")
        assert cfg.enabled is True


# ---------------------------------------------------------------------------
# Architectural guard — BridgeConfig tree must not shadow RetrievalConfig fields
# ---------------------------------------------------------------------------

def _collect_all_field_names(model_cls, visited=None) -> set[str]:
    """Recursively collect all field names from a Pydantic model tree.

    Uses field_info.annotation (not model_cls.__annotations__) because the
    latter returns the un-evaluated string or bare class without model_fields
    in the Pydantic v2 repr.  field_info.annotation is the resolved type.
    """
    if visited is None:
        visited = set()
    fields = set()
    for name, field_info in model_cls.model_fields.items():
        fields.add(name)
        annotation = field_info.annotation  # resolved type, not the raw annotation
        if annotation is not None and hasattr(annotation, "model_fields"):
            cls_id = id(annotation)
            if cls_id not in visited:
                visited.add(cls_id)
                fields.update(_collect_all_field_names(annotation, visited))
    return fields


_RETRIEVAL_SEMANTIC_FIELDS = {
    "retrieval_top_k",
    "retrieval_max_tokens",
    "retrieval_strategy",
    "max_memories_per_turn",
}


def test_bridge_config_has_no_retrieval_semantic_fields():
    """BridgeConfig tree must not define any retrieval-semantic fields."""
    bridge_fields = _collect_all_field_names(BridgeConfig)
    violations = _RETRIEVAL_SEMANTIC_FIELDS & bridge_fields
    assert not violations, (
        f"BridgeConfig redefines SR2 retrieval fields: {violations}. "
        f"Retrieval semantics belong in PipelineConfig.retrieval, not bridge config."
    )


def test_sr2_retrieval_config_owns_canonical_defaults():
    """SR2's RetrievalConfig must define the canonical defaults for retrieval."""
    cfg = RetrievalConfig()
    assert cfg.top_k == 10
    assert cfg.max_tokens == 4000
    assert cfg.strategy == "hybrid"


# ---------------------------------------------------------------------------
# Wiring: proxy_optimize() must pass RetrievalConfig values to retriever
# ---------------------------------------------------------------------------

class TestProxyOptimizeRetrieverWiring:
    """proxy_optimize() must pass top_k and max_tokens from RetrievalConfig to retrieve().

    All tests build PipelineConfig with the desired RetrievalConfig values BEFORE
    constructing BridgeEngine — mutating config after construction has no effect
    because SR2 reads retrieval config at __init__ time.
    """

    @pytest.mark.asyncio
    async def test_proxy_optimize_passes_top_k_from_retrieval_config(self):
        """retrieve() must be called with top_k from PipelineConfig.retrieval.top_k."""
        from sr2.compaction.engine import ConversationTurn
        from sr2_bridge.engine import BridgeEngine

        # Build config with non-default top_k BEFORE constructing the engine.
        pipeline_cfg = PipelineConfig(retrieval=RetrievalConfig(top_k=7))
        engine = BridgeEngine(pipeline_cfg)

        captured_kwargs = {}

        async def capturing_retrieve(query, top_k=None, max_tokens=None):
            captured_kwargs["top_k"] = top_k
            captured_kwargs["max_tokens"] = max_tokens
            return []

        engine._sr2._retriever.retrieve = capturing_retrieve

        turn = ConversationTurn(
            turn_number=1,
            role="user",
            content="hello",
        )

        await engine._sr2.proxy_optimize(
            new_turns=[turn],
            session_id="test-session",
            retrieval_query="test query",
        )

        assert captured_kwargs.get("top_k") == 7, (
            f"proxy_optimize() passed top_k={captured_kwargs.get('top_k')!r} to retrieve(), "
            f"expected 7 (from RetrievalConfig.top_k). "
            f"This means RetrievalConfig.top_k is still being ignored in proxy mode."
        )

    @pytest.mark.asyncio
    async def test_proxy_optimize_passes_max_tokens_from_retrieval_config(self):
        """retrieve() must be called with max_tokens from PipelineConfig.retrieval.max_tokens."""
        from sr2.compaction.engine import ConversationTurn
        from sr2_bridge.engine import BridgeEngine

        # Build config with non-default max_tokens BEFORE constructing the engine.
        pipeline_cfg = PipelineConfig(retrieval=RetrievalConfig(max_tokens=1234))
        engine = BridgeEngine(pipeline_cfg)

        captured_kwargs = {}

        async def capturing_retrieve(query, top_k=None, max_tokens=None):
            captured_kwargs["top_k"] = top_k
            captured_kwargs["max_tokens"] = max_tokens
            return []

        engine._sr2._retriever.retrieve = capturing_retrieve

        turn = ConversationTurn(
            turn_number=1,
            role="user",
            content="hello",
        )

        await engine._sr2.proxy_optimize(
            new_turns=[turn],
            session_id="test-session",
            retrieval_query="test query",
        )

        assert captured_kwargs.get("max_tokens") == 1234, (
            f"proxy_optimize() passed max_tokens={captured_kwargs.get('max_tokens')!r} to retrieve(), "
            f"expected 1234 (from RetrievalConfig.max_tokens). "
            f"This means RetrievalConfig.max_tokens is still being ignored in proxy mode."
        )

    @pytest.mark.asyncio
    async def test_proxy_optimize_retriever_uses_retrieval_config_strategy(self):
        """HybridRetriever must be constructed with strategy from RetrievalConfig.

        This tests the retriever construction wiring (Step 2a of the fix):
        SR2.__init__ must wire RetrievalConfig.strategy into the retriever.
        """
        from sr2_bridge.engine import BridgeEngine

        # Build config with non-default strategy BEFORE constructing the engine.
        pipeline_cfg = PipelineConfig(retrieval=RetrievalConfig(strategy="keyword"))
        engine = BridgeEngine(pipeline_cfg)

        # After fix, _retriever._strategy should reflect RetrievalConfig.strategy
        assert engine._sr2._retriever._strategy == "keyword", (
            f"HybridRetriever._strategy={engine._sr2._retriever._strategy!r}, "
            f"expected 'keyword' from RetrievalConfig.strategy. "
            f"The retriever is still using a hardcoded default instead of the config value."
        )

    @pytest.mark.asyncio
    async def test_proxy_optimize_retriever_top_k_wired_from_config(self):
        """HybridRetriever._top_k must reflect RetrievalConfig.top_k after construction.

        This tests the retriever construction wiring (Step 2a of the fix):
        SR2.__init__ must wire RetrievalConfig.top_k into the retriever.
        """
        from sr2_bridge.engine import BridgeEngine

        pipeline_cfg = PipelineConfig(retrieval=RetrievalConfig(top_k=3))
        engine = BridgeEngine(pipeline_cfg)

        assert engine._sr2._retriever._top_k == 3, (
            f"HybridRetriever._top_k={engine._sr2._retriever._top_k!r}, "
            f"expected 3 from RetrievalConfig.top_k. "
            f"The retriever is still using a hardcoded default."
        )
