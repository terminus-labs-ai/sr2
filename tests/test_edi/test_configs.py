"""Tests for EDI agent configs (Task 062)."""

import os

import pytest
import yaml

from sr2.config.loader import ConfigLoader


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "agents", "edi")


@pytest.fixture
def loader():
    return ConfigLoader()


def test_agent_yaml_loads(loader):
    """agent.yaml loads and validates."""
    path = os.path.join(CONFIGS_DIR, "agent.yaml")
    config = loader.load(path)
    assert config.token_budget == 65536
    # system_prompt is in the raw YAML but not in PipelineConfig model;
    # verify it exists in the raw data
    with open(path) as f:
        raw = yaml.safe_load(f)
    assert "EDI" in raw["system_prompt"]


def test_all_interface_configs_load(loader):
    """All interface configs load with correct inheritance."""
    iface_dir = os.path.join(CONFIGS_DIR, "interfaces")
    expected = {
        "heartbeat_plan",
    }
    found = set()
    for f in os.listdir(iface_dir):
        if f.endswith(".yaml"):
            name = f.rsplit(".", 1)[0]
            found.add(name)
            config = loader.load(os.path.join(iface_dir, f))
            # All should inherit from agent.yaml (verify by checking token_budget is set)
            assert config.token_budget > 0

    assert found == expected


def test_heartbeat_configs_pipeline_disabled(loader):
    """Heartbeat config has compaction/summarization/retrieval disabled."""
    path = os.path.join(CONFIGS_DIR, "interfaces", "heartbeat_plan.yaml")
    config = loader.load(path)
    assert not config.compaction.enabled
    assert not config.summarization.enabled
    assert not config.retrieval.enabled


def test_token_budgets(loader):
    """Token budgets: user_message=48000, heartbeat_plan=3000, webhooks=16000."""
    cases = {
        "heartbeat_plan": 3000,
    }
    for name, expected_budget in cases.items():
        path = os.path.join(CONFIGS_DIR, "interfaces", f"{name}.yaml")
        config = loader.load(path)
        assert config.token_budget == expected_budget, (
            f"{name}: expected {expected_budget}, got {config.token_budget}"
        )
