"""Tests for intent detection resolver."""

import pytest

from sr2.resolvers.intent_detection_resolver import IntentDetectionResolver
from sr2.resolvers.registry import ResolverContext


@pytest.fixture
def intent_resolver() -> IntentDetectionResolver:
    """Create an intent detection resolver."""
    return IntentDetectionResolver()


@pytest.mark.asyncio
async def test_intent_detection_technical(intent_resolver: IntentDetectionResolver) -> None:
    """Test detection of technical intent."""
    context = ResolverContext(
        agent_config={},
        trigger_input="There's a bug in the API error handling",
    )

    result = await intent_resolver.resolve({}, context)
    assert result is not None
    assert result.key == "intent_classification"
    assert "technical" in result.content
    assert result.tokens > 0


@pytest.mark.asyncio
async def test_intent_detection_planning(intent_resolver: IntentDetectionResolver) -> None:
    """Test detection of planning intent."""
    context = ResolverContext(
        agent_config={},
        trigger_input="We need to plan the next sprint timeline and milestones",
    )

    result = await intent_resolver.resolve({}, context)
    assert result is not None
    assert result.key == "intent_classification"
    assert "planning" in result.content


@pytest.mark.asyncio
async def test_intent_detection_documentation(intent_resolver: IntentDetectionResolver) -> None:
    """Test detection of documentation intent."""
    context = ResolverContext(
        agent_config={},
        trigger_input="Can you help me write a README for the documentation?",
    )

    result = await intent_resolver.resolve({}, context)
    assert result is not None
    assert "documentation" in result.content


@pytest.mark.asyncio
async def test_intent_detection_empty_input(intent_resolver: IntentDetectionResolver) -> None:
    """Test handling of empty input."""
    context = ResolverContext(agent_config={}, trigger_input=None)

    result = await intent_resolver.resolve({}, context)
    assert result is not None
    assert result.content == ""
    assert result.tokens == 0


@pytest.mark.asyncio
async def test_intent_detection_general_fallback(intent_resolver: IntentDetectionResolver) -> None:
    """Test fallback to general intent."""
    context = ResolverContext(
        agent_config={},
        trigger_input="Just saying hello",
    )

    result = await intent_resolver.resolve({}, context)
    assert result is not None
    assert "general" in result.content
