"""sr2-94 / AC4-r6 — backward compat: continuity-disabled retains full replay.

Kept in its own module (apart from test_working_set.py, which reds on the
not-yet-existing WorkingSetResolver import) so this guard PASSES today and
must keep passing after the implementation lands: it is the backward-compat
anchor of the cage, never masked by the deliberate red.

Row: AC4-r6 — SessionResolver (legacy) seeded with 100 messages, no
continuity config -> resolve() returns all 100 seeded messages (full replay,
unchanged behavior).
"""

from __future__ import annotations

from sr2.config.models import ResolverConfig
from sr2.models import Message, TextBlock
from sr2.pipeline.events import Event, EventPhase
from sr2.pipeline.models import ResolvedContent
from sr2.pipeline.resolvers.session import SessionResolver


def _message(text: str, role: str) -> Message:
    return Message(role=role, content=[TextBlock(text=text)])  # type: ignore[arg-type]


class TestAC4R6LegacySessionResolverUnchanged:
    async def test_seeded_session_resolver_returns_all_messages(self):
        """AC4-r6: legacy SessionResolver seeded with 100 messages, no
        continuity config -> resolve() returns all 100 seeded messages
        (full replay, unchanged behavior)."""
        seeded = [
            _message(
                f"seed {i} conversation history",
                "user" if i % 2 == 0 else "assistant",
            )
            for i in range(100)
        ]
        resolver = SessionResolver(ResolverConfig(type="session"))
        resolver.seed(seeded)

        result = await resolver.resolve(
            [
                Event(
                    name="user_input",
                    phase=EventPhase.STARTING,
                    source_layer="core",
                    data=[TextBlock(text="r6 turn input that is not yet history")],
                )
            ]
        )

        assert isinstance(result, ResolvedContent)
        assert len(result.content) == 100, (
            "full replay must return every seeded message; got "
            f"{len(result.content)} of 100"
        )
        for i, item in enumerate(result.content):
            assert isinstance(item, Message)
            text = "".join(
                b.text for b in item.content if isinstance(b, TextBlock)
            )
            assert text == f"seed {i} conversation history"
