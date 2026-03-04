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
        # Track zone transition counts per session for metrics
        self._zone_transitions: dict[str, dict[str, int]] = {}

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
        raw_count_before = len(zones.raw)
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

        # Track zone transitions (messages that moved from raw to compacted)
        transitioned = max(0, raw_count_before - len(zones.raw))
        if transitioned > 0:
            transitions = self._zone_transitions.setdefault(session_id, {})
            transitions["raw_to_compacted"] = transitions.get("raw_to_compacted", 0) + transitioned

        return result

    async def run_summarization(self, session_id: str = "default") -> SummarizationResult | None:
        """Check if summarization should trigger, and run it if needed.

        1. Calculate compacted zone tokens
        2. Check if summarization should trigger
        3. If yes: summarize the compacted zone (excluding preserve_recent_turns)
        4. Move summary to summarized zone
        5. Keep preserved turns in the compacted zone
        """
        if not self._summarization:
            return None

        zones = self._get_zones(session_id)
        compacted_tokens = sum(len(t.content) // 4 for t in zones.compacted)
        if not self._summarization.should_trigger(compacted_tokens, self._compacted_max):
            return None

        if not zones.compacted:
            return None

        # Exclude preserve_recent_turns from summarization
        preserve_n = self._summarization.preserve_recent_turns
        if preserve_n >= len(zones.compacted):
            return None
        to_summarize = zones.compacted[:-preserve_n] if preserve_n > 0 else zones.compacted
        to_preserve = zones.compacted[-preserve_n:] if preserve_n > 0 else []

        turns_text = "\n".join(f"{t.role}: {t.content}" for t in to_summarize)
        first_turn = to_summarize[0].turn_number
        last_turn = to_summarize[-1].turn_number
        turn_range = f"{first_turn}-{last_turn}"
        summarized_tokens = sum(len(t.content) // 4 for t in to_summarize)

        result = await self._summarization.summarize(
            turns_text=turns_text,
            turn_range=turn_range,
            original_tokens=summarized_tokens,
        )

        if hasattr(result.summary, "summary_of_turns"):
            from sr2.resolvers.summarization_resolver import SummarizationResolver

            summary_text = SummarizationResolver.format_structured_summary(result.summary)
        else:
            summary_text = str(result.summary)

        # Track zone transitions (compacted messages summarized)
        compacted_count = len(to_summarize)

        zones.summarized.append(summary_text)
        zones.compacted = to_preserve

        if compacted_count > 0:
            transitions = self._zone_transitions.setdefault(session_id, {})
            transitions["compacted_to_summarized"] = (
                transitions.get("compacted_to_summarized", 0) + compacted_count
            )

        return result

    def get_all_turns(self, session_id: str = "default") -> list[ConversationTurn]:
        """Get all turns across zones (for full history access)."""
        zones = self._get_zones(session_id)
        return zones.compacted + zones.raw

    @property
    def raw_window(self) -> int:
        """The configured raw window size."""
        return self._raw_window

    def get_raw_window_utilization(self, session_id: str = "default") -> float:
        """Get the raw window utilization ratio (0.0 to 1.0)."""
        zones = self._get_zones(session_id)
        if self._raw_window <= 0:
            return 0.0
        return min(1.0, len(zones.raw) / self._raw_window)

    def get_zone_transitions(self, session_id: str = "default") -> dict[str, int]:
        """Get cumulative zone transition counts for a session."""
        return dict(self._zone_transitions.get(session_id, {}))

    def destroy_session(self, session_id: str) -> None:
        """Clean up zones for a destroyed session."""
        self._zones_by_session.pop(session_id, None)
        self._zone_transitions.pop(session_id, None)
