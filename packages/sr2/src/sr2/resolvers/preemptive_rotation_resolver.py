"""Pre-emptive context rotation resolver.

Detects when token budget is nearing limit and proactively triggers context rotation
before hitting hard limits. This prevents cache invalidation from emergency truncation.
"""

import logging
from typing import Any

from sr2.resolvers.registry import ContentResolver, ResolverContext, ResolvedContent

logger = logging.getLogger(__name__)


class PreemptiveRotationResolver(ContentResolver):
    """Detects token budget pressure and triggers proactive rotation.

    Monitors the ratio of current tokens to budget. When approaching the threshold
    (pre_rot_threshold, default 0.75), this resolver:
    1. Flags that rotation should be initiated
    2. Optionally compacts or summarizes to free space
    3. Allows graceful cleanup before hard truncation
    """

    async def resolve(
        self,
        config: dict[str, Any],
        context: ResolverContext,
    ) -> ResolvedContent | None:
        """Check if pre-emptive rotation is needed.

        Returns a signal indicating:
        - rotation_needed: boolean
        - current_ratio: current tokens / budget ratio
        - action: recommended action (compact, summarize, rotate)
        """
        # This resolver doesn't generate content itself.
        # Instead, it's consulted by the pipeline to decide on rotation strategy.
        # Return empty content — the signal is in the metadata.
        return ResolvedContent(
            key="preemptive_rotation_signal",
            content="",
            tokens=0,
            metadata={
                "rotation_needed": False,
                "current_ratio": 0.0,
                "action": "none",
            },
        )

    @staticmethod
    def should_rotate(current_tokens: int, token_budget: int, threshold: float = 0.75) -> bool:
        """Check if current token usage exceeds rotation threshold.

        Args:
            current_tokens: Current token count
            token_budget: Total token budget
            threshold: Fraction of budget (0-1) that triggers rotation

        Returns:
            True if rotation should be initiated
        """
        if token_budget <= 0:
            return False
        ratio = current_tokens / token_budget
        return ratio >= threshold

    @staticmethod
    def get_rotation_status(
        current_tokens: int, token_budget: int, threshold: float = 0.75
    ) -> dict[str, Any]:
        """Get detailed rotation status information."""
        if token_budget <= 0:
            ratio = 0.0
        else:
            ratio = current_tokens / token_budget

        should_rotate = ratio >= threshold

        return {
            "current_tokens": current_tokens,
            "token_budget": token_budget,
            "ratio": ratio,
            "threshold": threshold,
            "should_rotate": should_rotate,
            "tokens_until_rotation": max(
                0, int(token_budget * threshold) - current_tokens
            ),
        }
