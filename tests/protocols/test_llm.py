"""Tests for the LLMClient protocol, Message, and CompletionResult.

Verifies:
- Message dataclass is frozen/immutable
- CompletionResult dataclass is frozen with correct defaults
- LLMClient protocol is runtime_checkable
- Non-conforming classes are rejected by isinstance
- A mock LLMClient can be called and returns CompletionResult
"""

import asyncio

import pytest

from sr2.protocols import Message, CompletionResult, LLMClient


class TestMessage:
    """Verify Message value object."""

    def test_message_creation(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_is_frozen(self):
        msg = Message(role="user", content="hello")
        with pytest.raises(Exception):  # FrozenInstanceError
            msg.role = "changed"

    def test_message_is_frozen_content(self):
        msg = Message(role="assistant", content="hi")
        with pytest.raises(Exception):  # FrozenInstanceError
            msg.content = "changed"


class TestCompletionResult:
    """Verify CompletionResult value object."""

    def test_completion_result_creation(self):
        result = CompletionResult(content="answer")
        assert result.content == "answer"

    def test_completion_result_is_frozen(self):
        result = CompletionResult(content="answer")
        with pytest.raises(Exception):  # FrozenInstanceError
            result.content = "changed"

    def test_completion_result_usage_default(self):
        result = CompletionResult(content="answer")
        assert result.usage == {}

    def test_completion_result_model_default(self):
        result = CompletionResult(content="answer")
        assert result.model == ""

    def test_completion_result_with_usage(self):
        result = CompletionResult(
            content="answer",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="gpt-4",
        )
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.model == "gpt-4"


class TestLLMClientProtocol:
    """Verify LLMClient protocol is runtime-checkable."""

    def test_conforming_class_satisfies_isinstance(self):
        """A class with the correct complete signature satisfies isinstance."""

        class FakeClient:
            async def complete(
                self,
                messages: list[Message],
                model: str | None = None,
                temperature: float = 0.0,
                max_tokens: int | None = None,
            ) -> CompletionResult:
                return CompletionResult(content="ok")

        client = FakeClient()
        assert isinstance(client, LLMClient)

    def test_missing_method_fails_isinstance(self):
        """A class without complete does NOT satisfy isinstance."""

        class NotAClient:
            pass

        obj = NotAClient()
        assert not isinstance(obj, LLMClient)

    def test_mock_client_returns_completion_result(self):
        """A mock LLMClient can be called and returns CompletionResult."""

        class MockClient:
            async def complete(
                self,
                messages: list[Message],
                model: str | None = None,
                temperature: float = 0.0,
                max_tokens: int | None = None,
            ) -> CompletionResult:
                return CompletionResult(
                    content="mock response",
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                    model=model or "test-model",
                )

        async def _run():
            client = MockClient()
            msgs = [Message(role="user", content="hello")]
            result = await client.complete(msgs, model="test-model")
            assert result.content == "mock response"
            assert result.usage == {"prompt_tokens": 5, "completion_tokens": 3}
            assert result.model == "test-model"

        asyncio.run(_run())
