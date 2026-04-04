"""Tests for SessionNotesResolver."""

import pytest

from sr2.resolvers.registry import ResolverContext
from sr2.resolvers.session_notes_resolver import SessionNotesResolver


def _context(notes: list[str] | None = None) -> ResolverContext:
    agent_config = {}
    if notes is not None:
        agent_config["session_notes"] = notes
    return ResolverContext(
        agent_config=agent_config,
        trigger_input="",
        session_id="test",
        interface_type="user_message",
    )


class TestSessionNotesResolver:
    @pytest.mark.asyncio
    async def test_empty_notes_returns_empty(self):
        resolver = SessionNotesResolver()
        result = await resolver.resolve("session_notes", {}, _context([]))
        assert result.content == ""
        assert result.tokens == 0

    @pytest.mark.asyncio
    async def test_missing_notes_returns_empty(self):
        resolver = SessionNotesResolver()
        result = await resolver.resolve("session_notes", {}, _context())
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_notes_formatted_as_xml(self):
        resolver = SessionNotesResolver()
        result = await resolver.resolve(
            "session_notes", {}, _context(["fix auth bug", "use JWT tokens"])
        )
        assert "<session_notes>" in result.content
        assert "</session_notes>" in result.content
        assert "- fix auth bug" in result.content
        assert "- use JWT tokens" in result.content

    @pytest.mark.asyncio
    async def test_notes_order_preserved(self):
        resolver = SessionNotesResolver()
        result = await resolver.resolve(
            "session_notes", {}, _context(["first", "second", "third"])
        )
        lines = result.content.strip().split("\n")
        # Skip wrapper tags
        note_lines = [line for line in lines if line.startswith("- ")]
        assert note_lines == ["- first", "- second", "- third"]

    @pytest.mark.asyncio
    async def test_max_tokens_drops_oldest(self):
        resolver = SessionNotesResolver()
        # Each note ~25 chars = ~6 tokens. With max_tokens=30, only newest notes fit.
        notes = [f"note number {i} with some padding text" for i in range(20)]
        result = await resolver.resolve(
            "session_notes", {"max_tokens": 30}, _context(notes)
        )
        # Should have fewer notes than input
        note_lines = [line for line in result.content.split("\n") if line.startswith("- ")]
        assert len(note_lines) < 20
        # The kept notes should be the newest ones
        assert note_lines[-1].endswith("19 with some padding text")

    @pytest.mark.asyncio
    async def test_tokens_counted(self):
        resolver = SessionNotesResolver()
        result = await resolver.resolve(
            "session_notes", {}, _context(["a note"])
        )
        assert result.tokens > 0

    @pytest.mark.asyncio
    async def test_max_tokens_from_config(self):
        resolver = SessionNotesResolver()
        result = await resolver.resolve(
            "session_notes", {"max_tokens": 5000}, _context(["note"])
        )
        assert "<session_notes>" in result.content
