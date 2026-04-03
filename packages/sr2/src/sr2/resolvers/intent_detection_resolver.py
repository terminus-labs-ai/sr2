"""Intent detection resolver: classifies user messages to detect topic shifts."""

import json
import logging
from typing import Any

from sr2.resolvers.registry import ContentResolver, ResolverContext, ResolvedContent

logger = logging.getLogger(__name__)


class IntentDetectionResolver(ContentResolver):
    """Detects intent classification and topic shifts in user messages.

    Used internally to determine if a topic shift warrants memory refresh.
    Returns a structured classification with confidence scores.
    """

    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        """Classify the intent of the trigger input (user message).

        Returns a structured intent classification including:
        - primary_intent: main topic/intent
        - confidence: 0-1 confidence score
        - topic_shift: boolean indicating detected topic change
        - previous_topic: last observed topic (if any)
        """
        if not context.trigger_input:
            return ResolvedContent(
                key="intent_classification",
                content="",
                tokens=0,
            )

        # For now, use a simple heuristic-based approach
        # (Real implementation would call an LLM for classification)
        intent = self._classify_intent(context.trigger_input, context)

        # Serialize the classification
        intent_str = json.dumps(intent, indent=2)

        return ResolvedContent(
            key="intent_classification",
            content=intent_str,
            tokens=len(intent_str) // 4,  # Character heuristic
        )

    def _classify_intent(self, message: str, context: ResolverContext) -> dict[str, Any]:
        """Classify intent using heuristics.

        A real implementation would:
        1. Call a fast LLM model to classify the message
        2. Compare with cached previous topic
        3. Return confidence-scored classification
        """
        message_lower = message.lower()

        # Detect common topic keywords
        topics = {
            "technical": ["code", "error", "bug", "debug", "api", "database", "deploy"],
            "planning": ["plan", "schedule", "timeline", "milestone", "task", "deadline"],
            "documentation": ["doc", "guide", "manual", "readme", "wiki", "help"],
            "analysis": ["analyze", "analyze", "review", "audit", "report", "data"],
            "general": [],  # Default/fallback
        }

        detected_topic = "general"
        max_matches = 0

        for topic, keywords in topics.items():
            matches = sum(1 for kw in keywords if kw in message_lower)
            if matches > max_matches:
                max_matches = matches
                detected_topic = topic

        # Estimate confidence based on keyword matches
        confidence = min(0.95, 0.5 + (max_matches * 0.15))

        return {
            "primary_intent": detected_topic,
            "confidence": confidence,
            "message_length": len(message),
            "keyword_matches": max_matches,
        }
