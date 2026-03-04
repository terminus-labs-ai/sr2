"""Compaction engine that applies rules to conversation turns."""

import logging
from dataclasses import dataclass

from sr2.compaction.rules import CompactionInput, get_rule
from sr2.config.models import CompactionConfig

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in conversation history."""

    turn_number: int
    role: str  # "user", "assistant", "tool_result"
    content: str
    content_type: str | None = None  # "tool_output", "file_content", etc.
    metadata: dict | None = None
    compacted: bool = False


@dataclass
class CompactionResult:
    """Result of running compaction on conversation history."""

    turns: list[ConversationTurn]
    original_tokens: int
    compacted_tokens: int
    turns_compacted: int


class CompactionEngine:
    """Applies compaction rules to conversation history."""

    def __init__(self, config: CompactionConfig):
        self._config = config
        self._rule_map = self._build_rule_map()

    def _build_rule_map(self) -> dict[str, tuple]:
        """Map content_type -> (rule_config, rule_instance)."""
        result = {}
        for rule_config in self._config.rules:
            rule = get_rule(rule_config.strategy)
            result[rule_config.type] = (rule_config, rule)
        return result

    def compact(self, turns: list[ConversationTurn]) -> CompactionResult:
        """Apply compaction to turns outside the raw window.

        Rules:
        1. Never compact user messages (role == "user")
        2. Never compact turns in the raw window (last N turns)
        3. Never compact already-compacted turns
        4. Skip content below min_content_size tokens
        5. Match turn's content_type to rule, apply if found
        """
        raw_window = self._config.raw_window
        min_size = self._config.min_content_size

        if len(turns) <= raw_window:
            total = sum(len(t.content) // 4 for t in turns)
            return CompactionResult(
                turns=turns,
                original_tokens=total,
                compacted_tokens=total,
                turns_compacted=0,
            )

        compactable = turns[:-raw_window]
        protected = turns[-raw_window:]

        original_tokens = 0
        compacted_tokens = 0
        turns_compacted = 0

        for turn in compactable:
            est_tokens = len(turn.content) // 4
            original_tokens += est_tokens

            if turn.role == "user":
                compacted_tokens += est_tokens
                continue
            if turn.compacted:
                compacted_tokens += est_tokens
                continue
            if est_tokens < min_size:
                compacted_tokens += est_tokens
                continue
            if turn.content_type not in self._rule_map:
                compacted_tokens += est_tokens
                continue

            rule_config, rule = self._rule_map[turn.content_type]
            inp = CompactionInput(
                content=turn.content,
                content_type=turn.content_type,
                tokens=est_tokens,
                metadata=turn.metadata,
            )
            output = rule.compact(inp, rule_config.model_dump())

            if output.was_compacted:
                compacted_content = output.content
                if output.recovery_hint:
                    compacted_content += f"\n  Recovery: {output.recovery_hint}"
                new_turn = ConversationTurn(
                    turn_number=turn.turn_number,
                    role=turn.role,
                    content=compacted_content,
                    content_type=turn.content_type,
                    metadata=turn.metadata,
                    compacted=True,
                )
                compactable[compactable.index(turn)] = new_turn
                compacted_tokens += output.tokens
                turns_compacted += 1
            else:
                compacted_tokens += est_tokens

        for turn in protected:
            est = len(turn.content) // 4
            original_tokens += est
            compacted_tokens += est

        all_turns = compactable + protected
        logger.info(
            f"Compaction completed - went from {original_tokens} to {compacted_tokens}, compacted {turns_compacted}/{len(all_turns)} turns"
        )

        return CompactionResult(
            turns=all_turns,
            original_tokens=original_tokens,
            compacted_tokens=compacted_tokens,
            turns_compacted=turns_compacted,
        )
