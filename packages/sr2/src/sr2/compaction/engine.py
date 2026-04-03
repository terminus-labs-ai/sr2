"""Compaction engine that applies rules to conversation turns."""

import logging
from dataclasses import dataclass

from sr2.compaction.cost_gate import CompactionCostGate
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
        self._cost_gate: CompactionCostGate | None = None
        if config.cost_gate.enabled:
            self._cost_gate = CompactionCostGate(config.cost_gate)

    @property
    def raw_window(self) -> int:
        """Number of recent turns kept in full detail."""
        return self._config.raw_window

    def _build_rule_map(self) -> dict[str, tuple]:
        """Map content_type -> (rule_config, rule_instance)."""
        result = {}
        for rule_config in self._config.rules:
            rule = get_rule(rule_config.strategy)
            result[rule_config.type] = (rule_config, rule)
        return result

    def compact(
        self,
        turns: list[ConversationTurn],
        model_hint: str | None = None,
    ) -> CompactionResult:
        """Apply compaction to turns outside the raw window.

        Rules:
        1. Never compact user messages (role == "user")
        2. Never compact turns in the raw window (last N turns)
        3. Never compact already-compacted turns
        4. Skip content below min_content_size tokens
        5. Match turn's content_type to rule, apply if found
        6. If cost gate is enabled, only compact if net-positive
        """
        raw_window = self._config.raw_window
        min_size = self._config.min_content_size

        if len(turns) <= raw_window:
            total = sum(len(t.content) // 4 for t in turns)
            logger.debug(
                "Compaction skipped: %d turns <= raw_window %d (%d tokens)",
                len(turns), raw_window, total,
            )
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
                logger.debug(
                    "Turn %d: skipped (user message, %d tokens)", turn.turn_number, est_tokens,
                )
                compacted_tokens += est_tokens
                continue
            if turn.compacted:
                logger.debug(
                    "Turn %d: skipped (already compacted, %d tokens)", turn.turn_number, est_tokens,
                )
                compacted_tokens += est_tokens
                continue
            if est_tokens < min_size:
                logger.debug(
                    "Turn %d: skipped (below min_content_size: %d < %d tokens)",
                    turn.turn_number, est_tokens, min_size,
                )
                compacted_tokens += est_tokens
                continue
            if turn.content_type not in self._rule_map:
                logger.warning(
                    "No compaction rule for content_type %r on turn %d (%d tokens), passing through uncompacted",
                    turn.content_type, turn.turn_number, est_tokens,
                )
                compacted_tokens += est_tokens
                continue

            rule_config, rule = self._rule_map[turn.content_type]
            logger.debug(
                "Turn %d: matched content_type=%r -> strategy=%r (%d tokens before)",
                turn.turn_number, turn.content_type, rule_config.strategy, est_tokens,
            )

            # Cost gate: estimate savings vs cache invalidation cost
            if self._cost_gate is not None:
                # Estimate compacted size as ~25% of original (heuristic)
                estimated_compacted = max(est_tokens // 4, 10)
                # Tokens after this turn = sum of all remaining compactable + protected
                idx = compactable.index(turn)
                tokens_after = sum(
                    len(t.content) // 4 for t in compactable[idx + 1 :]
                ) + sum(len(t.content) // 4 for t in protected)

                decision = self._cost_gate.should_compact(
                    turn_index=turn.turn_number,
                    turn_tokens=est_tokens,
                    estimated_compacted_tokens=estimated_compacted,
                    total_tokens_after_turn=tokens_after,
                    model_hint=model_hint,
                )
                if not decision.allowed:
                    logger.info(
                        "Turn %d: cost gate blocked compaction — %s",
                        turn.turn_number,
                        decision.reason,
                    )
                    compacted_tokens += est_tokens
                    continue

            inp = CompactionInput(
                content=turn.content,
                content_type=turn.content_type,
                tokens=est_tokens,
                metadata=turn.metadata,
            )
            output = rule.compact(inp, rule_config.model_dump())

            if output.was_compacted:
                logger.debug(
                    "Turn %d: compacted %d -> %d tokens (strategy=%r, recovery_hint=%r)",
                    turn.turn_number, est_tokens, output.tokens,
                    rule_config.strategy, output.recovery_hint,
                )
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
                logger.debug(
                    "Turn %d: strategy=%r ran but content unchanged (%d tokens)",
                    turn.turn_number, rule_config.strategy, est_tokens,
                )
                compacted_tokens += est_tokens

        for turn in protected:
            est = len(turn.content) // 4
            original_tokens += est
            compacted_tokens += est
            logger.debug(
                "Turn %d: protected (in raw_window, %d tokens)", turn.turn_number, est,
            )

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
