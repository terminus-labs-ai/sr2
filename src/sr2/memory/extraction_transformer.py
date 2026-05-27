"""MemoryExtractionTransformer: extracts and persists memories from assistant responses.

Subscribes to assistant_response.COMPLETED events. On each firing, extracts
durable facts from the response text via a MemoryExtractor and saves each
resulting Memory to the injected MemoryStore.

This transformer is purely side-effectful — it does not modify the pipeline
layer content. transform() always returns content=None (pass-through).
"""

from __future__ import annotations

from sr2.config.models import ConfigError, TransformerConfig
from sr2.memory.extractor_registry import EXTRACTORS
from sr2.memory.protocol import MemoryExtractor, MemoryStore
from sr2.models import ContentBlock, TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import TransformationResult


class MemoryExtractionTransformer:
    """Extract memories from completed assistant responses and persist them.

    Wired via deps.extras:
    - ``memory_store`` (required): a MemoryStore implementation — raises
      ConfigError at build time if absent.
    - ``memory_extractor`` (optional): a MemoryExtractor implementation.
      Defaults to RuleBasedExtractor() when absent (forward-compatible).
    """

    name: str = "memory_extraction"

    def __init__(
        self,
        config: TransformerConfig,
        store: MemoryStore,
        extractor: MemoryExtractor,
    ) -> None:
        self._config = config
        self._store = store
        self._extractor = extractor
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0

        self.subscriptions: list[EventSubscription] = [
            EventSubscription(
                event_name="assistant_response",
                phase=EventPhase.COMPLETED,
            )
        ]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls, config: TransformerConfig, deps: Dependencies
    ) -> "MemoryExtractionTransformer":
        """Construct from a TransformerConfig and a Dependencies container."""
        if "memory_store" not in deps.extras:
            raise ConfigError(
                "transformer 'memory_extraction' requires memory_store in deps.extras "
                "but none was provided"
            )

        store: MemoryStore = deps.extras["memory_store"]

        # extras override takes precedence over registry lookup
        if "memory_extractor" in deps.extras:
            extractor: MemoryExtractor = deps.extras["memory_extractor"]
        else:
            extractor_name: str = config.config.get("extractor", "rule_based")
            extractor = EXTRACTORS.get(extractor_name)()

        return cls(config, store, extractor)

    # ------------------------------------------------------------------
    # Transformer protocol
    # ------------------------------------------------------------------

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult:
        """Extract memories from assistant_response.COMPLETED events and save them.

        Always returns content=None — this transformer does not modify the layer.
        """
        for event in events:
            if (
                event.name == "assistant_response"
                and event.phase == EventPhase.COMPLETED
            ):
                text = self._extract_text(event)
                if text:
                    result = self._extractor.extract(text)
                    for memory in result.memories:
                        self._store.save(memory)

        return TransformationResult(
            transformer_name=self.name,
            source_layer=self.name,
            content=None,
            events=[],
            entries=[],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(event: Event) -> str:
        """Concatenate text from all TextBlocks in event.data."""
        if not event.data:
            return ""
        parts: list[str] = []
        for item in event.data:
            if isinstance(item, TextBlock):
                parts.append(item.text)
        return " ".join(parts)
