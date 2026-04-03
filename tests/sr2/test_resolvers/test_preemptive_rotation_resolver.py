"""Tests for pre-emptive context rotation resolver."""

import pytest

from sr2.resolvers.preemptive_rotation_resolver import PreemptiveRotationResolver
from sr2.resolvers.registry import ResolverContext


@pytest.fixture
def rotation_resolver() -> PreemptiveRotationResolver:
    """Create a pre-emptive rotation resolver."""
    return PreemptiveRotationResolver()


@pytest.mark.asyncio
async def test_rotation_resolve_returns_empty(rotation_resolver: PreemptiveRotationResolver) -> None:
    """Test that resolver returns empty content (signal is in metadata)."""
    context = ResolverContext(agent_config={}, trigger_input="test")

    result = await rotation_resolver.resolve({}, context)
    assert result is not None
    assert result.key == "preemptive_rotation_signal"
    assert result.content == ""
    assert result.tokens == 0


def test_should_rotate_below_threshold() -> None:
    """Test that rotation is not needed below threshold."""
    assert PreemptiveRotationResolver.should_rotate(4000, 8000, threshold=0.75) is False


def test_should_rotate_at_threshold() -> None:
    """Test that rotation is needed at threshold."""
    assert PreemptiveRotationResolver.should_rotate(6000, 8000, threshold=0.75) is True


def test_should_rotate_above_threshold() -> None:
    """Test that rotation is needed above threshold."""
    assert PreemptiveRotationResolver.should_rotate(7500, 8000, threshold=0.75) is True


def test_should_rotate_well_below_threshold() -> None:
    """Test that rotation is not needed well below threshold."""
    assert PreemptiveRotationResolver.should_rotate(1000, 8000, threshold=0.75) is False


def test_should_rotate_zero_budget() -> None:
    """Test edge case: zero budget."""
    assert PreemptiveRotationResolver.should_rotate(100, 0, threshold=0.75) is False


def test_get_rotation_status_below_threshold() -> None:
    """Test rotation status below threshold."""
    status = PreemptiveRotationResolver.get_rotation_status(4000, 8000, threshold=0.75)
    assert status["should_rotate"] is False
    assert status["ratio"] == 0.5
    assert status["tokens_until_rotation"] == 2000


def test_get_rotation_status_at_threshold() -> None:
    """Test rotation status at threshold."""
    status = PreemptiveRotationResolver.get_rotation_status(6000, 8000, threshold=0.75)
    assert status["should_rotate"] is True
    assert status["ratio"] == 0.75
    assert status["tokens_until_rotation"] == 0


def test_get_rotation_status_above_threshold() -> None:
    """Test rotation status above threshold."""
    status = PreemptiveRotationResolver.get_rotation_status(7500, 8000, threshold=0.75)
    assert status["should_rotate"] is True
    assert status["ratio"] == 0.9375
    assert status["tokens_until_rotation"] == 0


def test_get_rotation_status_custom_threshold() -> None:
    """Test rotation status with custom threshold."""
    status = PreemptiveRotationResolver.get_rotation_status(5000, 10000, threshold=0.5)
    assert status["should_rotate"] is True
    assert status["ratio"] == 0.5
    assert status["tokens_until_rotation"] == 0


def test_get_rotation_status_zero_budget() -> None:
    """Test rotation status with zero budget."""
    status = PreemptiveRotationResolver.get_rotation_status(1000, 0, threshold=0.75)
    assert status["ratio"] == 0.0
    assert status["should_rotate"] is False
