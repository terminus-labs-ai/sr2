"""SummarizationTransformer: compresses older content via LLM summarization."""

from __future__ import annotations

from datetime import datetime, timezone

import ulid

from sr2.config.models import ConfigError, TransformerConfig
from sr2.models import ContentBlock, Message, TextBlock
from sr2.pipeline.dependencies import Dependencies
from sr2.pipeline.events import Event, EventPhase, EventSubscription
from sr2.pipeline.models import TransformationResult
from sr2.pipeline.provenance import Entry, EntryOrigin
from sr2.pipeline.protocols import TokenCounter
from sr2.pipeline.token_counting import CharacterTokenCounter
from sr2.pipeline.utils import PHASE_MAP, build_subscriptions
from sr2.protocols.llm import CompletionRequest, LLMCallable

_DEFAULT_SUMMARIZATION_PROMPT = "Summarize the following conversation turns concisely."


class SummarizationTransformer:
    """Summarizes older content blocks via an injected LLM callable.

    Supports two keep strategies:
    - keep_last_n: keep the N most-recent blocks, summarize the rest.
    - keep_within_tokens: keep the most-recent blocks that fit within a
      token budget (newest-to-oldest), summarize everything else.
    """

    name: str = "summarization"

    def __init__(
        self,
        config: TransformerConfig,
        llm: LLMCallable,
        token_counter: TokenCounter | None = None,
        session_id: str = "",
    ) -> None:
        self._config = config
        self._llm = llm
        self._token_counter: TokenCounter = token_counter or CharacterTokenCounter()
        self._session_id = session_id
        self.max_executions: int = config.max_executions
        self.execution_count: int = 0

        self.subscriptions: list[EventSubscription] = build_subscriptions(
            config.subscriptions, PHASE_MAP, []
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, config: TransformerConfig, deps: Dependencies) -> "SummarizationTransformer":
        """Construct from a TransformerConfig and a Dependencies container."""
        if deps.llm is None:
            raise ConfigError("transformer 'summarize' requires an LLM but none was provided")
        cfg = config.config or {}
        key = cfg.get("model") if cfg.get("model") else None

        if key is not None:
            # Explicit model key: must be present — no silent fallback.
            if key not in deps.llm:
                raise ConfigError(
                    f"transformer 'summarization' configured model={key!r} "
                    f"but that key is absent from deps.llm (available: {list(deps.llm.keys())!r})"
                )
            llm = deps.llm[key]
        else:
            # No model key in config: use "default" if present, else raise.
            if "default" not in deps.llm:
                raise ConfigError(
                    "transformer 'summarization' has no 'model' config key and "
                    f"deps.llm has no 'default' entry (available: {list(deps.llm.keys())!r})"
                )
            llm = deps.llm["default"]
        return cls(config, llm, session_id=deps.session_id)

    # ------------------------------------------------------------------
    # Transformer protocol
    # ------------------------------------------------------------------

    async def transform(
        self, content: list[ContentBlock], events: list[Event]
    ) -> TransformationResult:
        cfg = self._config.config
        strategy = cfg.get("keep_strategy", "keep_last_n")

        if strategy == "keep_within_tokens":
            to_summarize, to_keep = self._split_by_tokens(content, cfg)
        else:
            # keep_last_n (default)
            to_summarize, to_keep = self._split_by_count(content, cfg)

        # No-op: nothing to summarize
        if not to_summarize:
            return TransformationResult(
                transformer_name=self.name,
                source_layer="summarization",
                content=None,
                events=[],
                entries=[],
            )

        # Call LLM to summarize
        summary_block = await self._summarize(to_summarize)

        # Build provenance entries — one per summarized item
        entries = self._make_entries(to_summarize)

        # Emit event
        event = Event(
            name="summarization_complete",
            phase=EventPhase.COMPLETED,
            source_layer="summarization",
            data=[summary_block],
        )

        new_content: list[ContentBlock] = [summary_block] + list(to_keep)

        return TransformationResult(
            transformer_name=self.name,
            source_layer="summarization",
            content=new_content,
            events=[event],
            entries=entries,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_by_count(
        self, content: list[ContentBlock], cfg: dict
    ) -> tuple[list[ContentBlock], list[ContentBlock]]:
        """Split content into (to_summarize, to_keep) by keep_last_n."""
        keep_n = cfg.get("keep_last_n", 3)
        if len(content) <= keep_n:
            return [], content
        return list(content[:-keep_n]), list(content[-keep_n:])

    def _split_by_tokens(
        self, content: list[ContentBlock], cfg: dict
    ) -> tuple[list[ContentBlock], list[ContentBlock]]:
        """Split content into (to_summarize, to_keep) by keep_within_tokens.

        Walk newest-to-oldest, accumulate tokens, keep items while cumulative
        count <= budget. Everything not kept is summarized.
        """
        budget = cfg.get("keep_tokens", 0)
        kept: list[ContentBlock] = []
        cumulative = 0
        for block in reversed(content):
            block_tokens = self._token_counter.count([block])
            if cumulative + block_tokens <= budget:
                kept.append(block)
                cumulative += block_tokens
            else:
                break

        # kept is newest-to-oldest; reverse to restore original order
        kept = list(reversed(kept))
        n_kept = len(kept)
        to_summarize = list(content[: len(content) - n_kept])
        return to_summarize, kept

    def _block_to_text(self, block: ContentBlock) -> str:
        """Extract a text representation from any ContentBlock type."""
        if isinstance(block, TextBlock):
            return block.text
        # For non-text blocks, emit a typed placeholder rather than str(block).
        block_type = type(block).__name__
        name_attr = getattr(block, "name", None)
        if name_attr is not None:
            return f"[{block_type}: {name_attr}]"
        return f"[{block_type}]"

    async def _summarize(self, blocks: list[ContentBlock]) -> TextBlock:
        """Build a CompletionRequest and call the LLM, returning the summary TextBlock."""
        turns_text = "\n".join(self._block_to_text(b) for b in blocks)
        cfg = self._config.config or {}
        prompt = cfg.get("summarization_prompt") or _DEFAULT_SUMMARIZATION_PROMPT
        request = CompletionRequest(
            system=[TextBlock(text=prompt)],
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text=turns_text)],
                )
            ],
        )
        response = await self._llm.complete(request)
        # Extract first TextBlock from response
        for block in response.content:
            if isinstance(block, TextBlock):
                return block
        return TextBlock(text="")

    def _make_entries(self, blocks: list[ContentBlock]) -> list[Entry]:
        """Create one provenance Entry per summarized block."""
        now = datetime.now(timezone.utc)
        entries = []
        for block in blocks:
            entry_id = str(ulid.ULID())
            source_id = str(ulid.ULID())
            entry = Entry(
                id=entry_id,
                content=block,
                sources=(source_id,),
                origin=EntryOrigin(kind="transformer", name=self.name),
                layer="summarization",
                session_id=self._session_id,
                created_at=now,
            )
            entries.append(entry)
        return entries
