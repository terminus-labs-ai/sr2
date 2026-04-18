"""Three-zone conversation manager: summarized -> compacted -> raw."""

import logging
from dataclasses import dataclass, field

from sr2.compaction.engine import (
    CompactionEngine,
    CompactionResult,
    ConversationTurn,
)
from sr2.summarization.engine import StructuredSummary, SummarizationEngine, SummarizationResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationZones:
    """The three zones of conversation history, plus compaction-immune session notes."""

    summarized: list[str] = field(default_factory=list)
    compacted: list[ConversationTurn] = field(default_factory=list)
    raw: list[ConversationTurn] = field(default_factory=list)
    session_notes: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        summarized_tokens = sum(len(s) // 4 for s in self.summarized)
        compacted_tokens = sum(len(t.content) // 4 for t in self.compacted)
        raw_tokens = sum(len(t.content) // 4 for t in self.raw)
        notes_tokens = sum(len(n) // 4 for n in self.session_notes)
        return summarized_tokens + compacted_tokens + raw_tokens + notes_tokens


class ConversationManager:
    """Manages three-zone conversation history, per session."""

    def __init__(
        self,
        compaction_engine: CompactionEngine,
        summarization_engine: SummarizationEngine | None = None,
        raw_window: int = 5,
        compacted_max_tokens: int = 6000,
        trace_collector=None,
    ):
        self._compaction = compaction_engine
        self._summarization = summarization_engine
        self._raw_window = raw_window
        self._compacted_max = compacted_max_tokens
        self._trace = trace_collector
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
        logger.debug(
            "Added turn %d to raw zone (role=%s, content_type=%s, %d tokens, session=%s)",
            turn.turn_number,
            turn.role,
            turn.content_type,
            len(turn.content) // 4,
            session_id,
        )

    def run_compaction(
        self,
        session_id: str = "default",
        model_hint: str | None = None,
        prefix_budget: int | None = None,
        token_budget: int | None = None,
        current_tokens: int | None = None,
    ) -> CompactionResult | None:
        """Run compaction on raw zone, moving compacted turns to compacted zone.

        1. If raw zone > raw_window, compact the overflow
        2. Move compacted turns to compacted zone
        3. Keep only last raw_window turns in raw zone
        """
        if not self._compaction._config.enabled:
            logger.debug("run_compaction: skipped (compaction disabled)")
            return None

        zones = self._get_zones(session_id)
        raw_count_before = len(zones.raw)
        all_turns = zones.compacted + zones.raw
        logger.debug(
            "run_compaction: session=%s, raw=%d, compacted=%d, total=%d, raw_window=%d",
            session_id,
            len(zones.raw),
            len(zones.compacted),
            len(all_turns),
            self._raw_window,
        )
        if len(all_turns) <= self._raw_window:
            logger.debug(
                "run_compaction: skipped (total turns %d <= raw_window %d)",
                len(all_turns),
                self._raw_window,
            )
            return None

        result = self._compaction.compact(
            all_turns,
            model_hint=model_hint,
            prefix_budget=prefix_budget,
            token_budget=token_budget,
            current_tokens=current_tokens,
        )

        if len(result.turns) > self._raw_window:
            zones.compacted = result.turns[: -self._raw_window]
            zones.raw = result.turns[-self._raw_window :]
        else:
            zones.compacted = []
            zones.raw = result.turns

        logger.debug(
            "run_compaction: result — compacted_zone=%d turns, raw_zone=%d turns, "
            "turns_compacted=%d, tokens %d->%d",
            len(zones.compacted),
            len(zones.raw),
            result.turns_compacted,
            result.original_tokens,
            result.compacted_tokens,
        )

        # Track zone transitions (messages that moved from raw to compacted)
        transitioned = max(0, raw_count_before - len(zones.raw))
        if transitioned > 0:
            transitions = self._zone_transitions.setdefault(session_id, {})
            transitions["raw_to_compacted"] = transitions.get("raw_to_compacted", 0) + transitioned

        return result

    async def run_summarization(
        self, session_id: str = "default", force: bool = False,
    ) -> SummarizationResult | None:
        """Check if summarization should trigger, and run it if needed.

        1. Calculate compacted zone tokens
        2. Check if summarization should trigger (or force=True)
        3. If yes: summarize the compacted zone (excluding preserve_recent_turns)
        4. Move summary to summarized zone
        5. Keep preserved turns in the compacted zone

        Args:
            force: Skip the threshold check and summarize if there are
                compacted turns. Used when the turn count itself is the
                problem (e.g. hundreds of short messages that individually
                fit in budget but overwhelm the LLM as individual messages).
        """
        if not self._summarization:
            return None

        zones = self._get_zones(session_id)
        compacted_tokens = sum(len(t.content) // 4 for t in zones.compacted)
        threshold_tokens = self._summarization._config.threshold * self._compacted_max
        logger.debug(
            "run_summarization: session=%s, compacted_tokens=%d, threshold=%d (%.0f%% of %d), force=%s",
            session_id,
            compacted_tokens,
            int(threshold_tokens),
            self._summarization._config.threshold * 100,
            self._compacted_max,
            force,
        )
        if not force and not self._summarization.should_trigger(compacted_tokens, self._compacted_max):
            logger.debug("run_summarization: not triggered (below threshold)")
            return None

        if not zones.compacted:
            return None

        # Exclude preserve_recent_turns from summarization
        preserve_n = self._summarization.preserve_recent_turns
        if preserve_n >= len(zones.compacted):
            logger.warning(
                "Summarization triggered but skipped: preserve_recent_turns (%d) >= compacted turns (%d)",
                preserve_n,
                len(zones.compacted),
            )
            return None
        to_summarize = zones.compacted[:-preserve_n] if preserve_n > 0 else zones.compacted
        to_preserve = zones.compacted[-preserve_n:] if preserve_n > 0 else []

        turns_text = "\n".join(f"{t.role}: {t.content}" for t in to_summarize)
        first_turn = to_summarize[0].turn_number
        last_turn = to_summarize[-1].turn_number
        turn_range = f"{first_turn}-{last_turn}"
        summarized_tokens = sum(len(t.content) // 4 for t in to_summarize)

        # Cap input to summarization LLM to avoid context window overflow.
        # The LLM prompt includes system instructions (~500 tokens) + turns_text.
        # Cap turns_text to compacted_max_tokens to stay within typical model limits.
        max_input_chars = self._compacted_max * 4  # ~1 token per 4 chars
        if len(turns_text) > max_input_chars:
            logger.warning(
                "run_summarization: truncating input from %d to %d chars "
                "(compacted_max_tokens=%d) to fit summarization model",
                len(turns_text),
                max_input_chars,
                self._compacted_max,
            )
            turns_text = turns_text[:max_input_chars]

        result = await self._summarization.summarize(
            turns_text=turns_text,
            turn_range=turn_range,
            original_tokens=summarized_tokens,
        )

        if isinstance(result.summary, StructuredSummary):
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

    def seed_from_history(self, history: list[dict], session_id: str = "default") -> int:
        """Populate zones from raw session history dicts.

        Idempotent — no-op if zones already have raw or compacted content.
        Returns the count of turns seeded.
        """
        zones = self._get_zones(session_id)
        if zones.raw or zones.compacted:
            return 0

        for i, msg in enumerate(history):
            role = msg["role"]
            content = msg.get("content") or ""
            content_type = "tool_output" if role in ("tool", "tool_result") else None

            # Build metadata from tool_calls, tool_call_id, and existing metadata
            meta: dict | None = None
            existing = msg.get("metadata")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if existing or tool_calls or tool_call_id:
                meta = dict(existing) if existing else {}
                if tool_calls:
                    meta["tool_calls"] = tool_calls
                if tool_call_id:
                    meta["tool_call_id"] = tool_call_id

            zones.raw.append(
                ConversationTurn(
                    turn_number=i,
                    role=role,
                    content=content,
                    content_type=content_type,
                    metadata=meta,
                )
            )

        count = len(history)
        if count:
            logger.debug(
                "Seeded %d turns from history (session=%s)", count, session_id
            )
        return count

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

    # --- Session Notes API ---

    def add_session_note(self, note: str, session_id: str = "default") -> None:
        """Append a note to the compaction-immune session notes."""
        zones = self._get_zones(session_id)
        zones.session_notes.append(note)
        logger.debug(
            "Added session note (session=%s, total=%d)", session_id, len(zones.session_notes)
        )

    def replace_session_notes(self, notes: list[str], session_id: str = "default") -> None:
        """Replace all session notes."""
        zones = self._get_zones(session_id)
        zones.session_notes = list(notes)
        logger.debug("Replaced session notes (session=%s, total=%d)", session_id, len(notes))

    def clear_session_notes(self, session_id: str = "default") -> None:
        """Clear all session notes."""
        zones = self._get_zones(session_id)
        zones.session_notes.clear()
        logger.debug("Cleared session notes (session=%s)", session_id)

    def get_session_notes(self, session_id: str = "default") -> list[str]:
        """Read current session notes."""
        return list(self._get_zones(session_id).session_notes)

    # --- Seeding from history ---

    def seed_from_history(self, history: list[dict], session_id: str = "default") -> int:
        """Populate zones from raw session history dicts.

        Idempotent — no-op if zones already have raw or compacted content.
        Returns count of turns seeded.
        """
        zones = self._get_zones(session_id)
        if zones.raw or zones.compacted:
            return 0

        for i, msg in enumerate(history):
            role = msg["role"]
            content = msg.get("content") or ""
            content_type = "tool_output" if role in ("tool", "tool_result") else None

            # Build metadata from tool_calls, tool_call_id, and existing metadata
            meta: dict | None = None
            existing_meta = msg.get("metadata")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if existing_meta or tool_calls or tool_call_id:
                meta = {}
                if existing_meta:
                    meta.update(existing_meta)
                if tool_calls:
                    meta["tool_calls"] = tool_calls
                if tool_call_id:
                    meta["tool_call_id"] = tool_call_id

            zones.raw.append(
                ConversationTurn(
                    turn_number=i,
                    role=role,
                    content=content,
                    content_type=content_type,
                    metadata=meta,
                )
            )

        count = len(history)
        if count:
            logger.debug(
                "Seeded %d turns from history (session=%s)", count, session_id
            )
        return count

    # --- Persistence ---

    def restore_zones(self, session_id: str, zones: ConversationZones) -> None:
        """Restore previously persisted zones into the manager."""
        self._zones_by_session[session_id] = zones

    def destroy_session(self, session_id: str) -> None:
        """Clean up zones for a destroyed session."""
        self._zones_by_session.pop(session_id, None)
        self._zone_transitions.pop(session_id, None)
