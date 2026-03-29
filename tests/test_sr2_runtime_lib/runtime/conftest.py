"""Shared fixtures for sr2.runtime tests."""

from __future__ import annotations

import pytest

from sr2.runtime.config import AgentConfig, ModelConfig, PersonaConfig


@pytest.fixture
def sample_agent_config():
    """Valid AgentConfig for testing."""
    return AgentConfig(
        name="test-agent",
        description="A test agent",
        model=ModelConfig(
            provider="ollama",
            name="test-model",
            base_url="http://localhost:11434",
        ),
        persona=PersonaConfig(
            system_prompt="You are a helpful test agent.",
        ),
    )


@pytest.fixture
def sample_config_dict():
    """Valid config dict for testing."""
    return {
        "name": "test-agent",
        "description": "A test agent",
        "model": {
            "provider": "ollama",
            "name": "test-model",
            "base_url": "http://localhost:11434",
        },
        "persona": {
            "system_prompt": "You are a helpful test agent.",
        },
    }
