"""CompactionTransformer: SR2 Transformer plugin for rule-based compaction."""

from __future__ import annotations

from sr2.compaction.engine import CompactionEngine
from sr2.compaction.rules import result_summary, schema_and_sample
from sr2.config.models import TransformerConfig
from sr2.models import ContentBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventSubscription
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions


class CompactionTransformer:
    """Rule-based compaction transformer registered as an SR2 plugin.

    Applies schema_and_sample and result_summary rules to all content
    blocks in the pipeline layer.
    """

    name: str = "compaction"

    def __init__(self, config: TransformerConfig) -> None:
        self._config = config
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0
        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, []
        )

        cfg = config.config or {}
        max_tokens: int = cfg.get("max_result_tokens", 500)

        self._engine = CompactionEngine(
            rules=[
                schema_and_sample,
                lambda b: result_summary(b, max_tokens=max_tokens),
            ]
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: TransformerConfig, deps: Dependencies) -> "CompactionTransformer":
        """Construct from a TransformerConfig and a Dependencies container."""
        return cls(config)

    # ------------------------------------------------------------------
    # Transformer protocol
    # ------------------------------------------------------------------

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult:
        """Apply compaction rules to *content* and return a TransformationResult."""
        compacted_content = self._engine.apply_to_blocks(content)
        any_change = compacted_content is not content

        return TransformationResult(
            transformer_name=self.name,
            source_layer="compaction",
            content=compacted_content if any_change else None,
            events=[],
            entries=[],
        )
