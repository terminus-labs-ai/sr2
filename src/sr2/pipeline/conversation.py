"""Three-zone conversation manager: summarized -> compacted -> raw."""

from dataclasses import dataclass, field

from sr2.compaction.engine import (
    CompactionEngine,
    CompactionResult,
    ConversationTurn,
)
from sr2.summarization.engine import SummarizationEngine, SummarizationResult


@dataclass
class ConversationZones:
    """The three zones of conversation history."""

    summarized: list[str] = field(default_factory=list)
    compacted: list[ConversationTurn] = field(default_factory=list)
    raw: list[ConversationTurn] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        summarized_tokens = sum(len(s) // 4 for s in self.summarized)
        compacted_tokens = sum(len(t.content) // 4 for t in self.compacted)
        raw_tokens = sum(len(t.content) // 4 for t in self.raw)
        return summarized_tokens + compacted_tokens + raw_tokens


class ConversationManager:
    """Manages three-zone conversation history, per session."""

    def __init__(
        self,
        compaction_engine: CompactionEngine,
        summarization_engine: SummarizationEngine | None = None,
        raw_window: int = 5,
        compacted_max_tokens: int = 6000,
    ):
        self._compaction = compaction_engine
        self._summarization = summarization_engine
        self._raw_window = raw_window
        self._compacted_max = compacted_max_tokens
        self._zones_by_session: dict[str, ConversationZones] = {}

    def _get_zones(self, session_id: str) -> ConversationZones:
        """Get or create zones for a session."""
        if session_id not in self._zones_by_session:
            self._zones_by_session[session_id] = ConversationZones()
        return self._zones_by_session[session_id]

    def zones(self, session_id: str = "default") -> ConversationZones:
        """Get zones for a session."""
        return self._get_zones(session_id)

    def add_turn(self, turn: ConversationTurn, session_id: str = "default") -> None:
        """Add a new turn to the raw zone."""
        self._get_zones(session_id).raw.append(turn)

    def run_compaction(self, session_id: str = "default") -> CompactionResult | None:
        """Run compaction on raw zone, moving compacted turns to compacted zone.

        1. If raw zone > raw_window, compact the overflow
        2. Move compacted turns to compacted zone
        3. Keep only last raw_window turns in raw zone
        """
        zones = self._get_zones(session_id)
        all_turns = zones.compacted + zones.raw
        if len(all_turns) <= self._raw_window:
            return None

        result = self._compaction.compact(all_turns)

        if len(result.turns) > self._raw_window:
            zones.compacted = result.turns[: -self._raw_window]
            zones.raw = result.turns[-self._raw_window :]
        else:
            zones.compacted = []
            zones.raw = result.turns

        return result

    async def run_summarization(self, session_id: str = "default") -> SummarizationResult | None:
        """Check if summarization should trigger, and run it if needed.

        1. Calculate compacted zone tokens
        2. Check if summarization should trigger
        3. If yes: summarize the compacted zone
        4. Move summary to summarized zone
        5. Clear the compacted zone
        """
        if not self._summarization:
            return None

        zones = self._get_zones(session_id)
        compacted_tokens = sum(len(t.content) // 4 for t in zones.compacted)
        if not self._summarization.should_trigger(compacted_tokens, self._compacted_max):
            return None

        if not zones.compacted:
            return None

        turns_text = "\n".join(
            f"{t.role}: {t.content}" for t in zones.compacted
        )
        first_turn = zones.compacted[0].turn_number
        last_turn = zones.compacted[-1].turn_number
        turn_range = f"{first_turn}-{last_turn}"

        result = await self._summarization.summarize(
            turns_text=turns_text,
            turn_range=turn_range,
            original_tokens=compacted_tokens,
        )

        if hasattr(result.summary, "summary_of_turns"):
            from sr2.resolvers.summarization_resolver import SummarizationResolver

            summary_text = SummarizationResolver.format_structured_summary(
                result.summary
            )
        else:
            summary_text = str(result.summary)

        zones.summarized.append(summary_text)
        zones.compacted = []

        return result

    def get_all_turns(self, session_id: str = "default") -> list[ConversationTurn]:
        """Get all turns across zones (for full history access)."""
        zones = self._get_zones(session_id)
        return zones.compacted + zones.raw

    def destroy_session(self, session_id: str) -> None:
        """Clean up zones for a destroyed session."""
        self._zones_by_session.pop(session_id, None)
