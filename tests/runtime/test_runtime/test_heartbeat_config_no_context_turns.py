"""Tests for HeartbeatConfig after Fix 11: max_context_turns removal.

Verifies that:
1. HeartbeatConfig no longer has a max_context_turns field.
2. YAML configs with max_context_turns are silently dropped (backward compat via extra="ignore").
3. HeartbeatConfig retains only its operational fields.
4. The Agent wires ScheduleHeartbeatTool.max_context_turns from SR2's pipeline config.
"""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml

from sr2_runtime.config import HeartbeatConfig


# ---------------------------------------------------------------------------
# Tests: field absence
# ---------------------------------------------------------------------------


class TestHeartbeatConfigFieldAbsence:
    """max_context_turns must not exist on HeartbeatConfig."""

    def test_max_context_turns_not_in_model_fields(self):
        """max_context_turns must not appear in HeartbeatConfig.model_fields."""
        assert "max_context_turns" not in HeartbeatConfig.model_fields, (
            "HeartbeatConfig.max_context_turns was found in model_fields — "
            "it must be removed per Fix 11. Context window is now owned by "
            "SR2's CompactionConfig.raw_window."
        )

    def test_default_instance_has_no_max_context_turns_attribute(self):
        """A default HeartbeatConfig instance must not have a max_context_turns attribute."""
        cfg = HeartbeatConfig()
        assert not hasattr(cfg, "max_context_turns"), (
            "HeartbeatConfig instance has max_context_turns attribute — "
            "it must be removed."
        )

    def test_max_context_turns_not_in_model_json_schema(self):
        """max_context_turns must not appear in the JSON schema."""
        schema = HeartbeatConfig.model_json_schema()
        props = schema.get("properties", {})
        assert "max_context_turns" not in props, (
            "max_context_turns is still present in HeartbeatConfig JSON schema."
        )


# ---------------------------------------------------------------------------
# Tests: retained operational fields
# ---------------------------------------------------------------------------


class TestHeartbeatConfigRetainedFields:
    """HeartbeatConfig retains its scheduling/capacity fields."""

    def test_enabled_field_present(self):
        assert "enabled" in HeartbeatConfig.model_fields

    def test_poll_interval_seconds_field_present(self):
        assert "poll_interval_seconds" in HeartbeatConfig.model_fields

    def test_max_pending_per_agent_field_present(self):
        assert "max_pending_per_agent" in HeartbeatConfig.model_fields

    def test_default_values(self):
        cfg = HeartbeatConfig()
        assert cfg.enabled is False
        assert cfg.poll_interval_seconds == 30
        assert cfg.max_pending_per_agent == 100

    def test_exactly_three_operational_fields(self):
        """HeartbeatConfig must have exactly enabled, poll_interval_seconds,
        max_pending_per_agent — no context management fields."""
        expected_fields = {"enabled", "poll_interval_seconds", "max_pending_per_agent"}
        actual_fields = set(HeartbeatConfig.model_fields.keys())
        assert actual_fields == expected_fields, (
            f"HeartbeatConfig has unexpected fields. "
            f"Extra: {actual_fields - expected_fields}. "
            f"Missing: {expected_fields - actual_fields}."
        )


# ---------------------------------------------------------------------------
# Tests: backward compat — silently drop max_context_turns from YAML
# ---------------------------------------------------------------------------


class TestHeartbeatConfigBackwardCompat:
    """YAML with max_context_turns must not raise — extra='ignore' drops it silently."""

    def test_max_context_turns_in_yaml_is_silently_dropped(self):
        """HeartbeatConfig with max_context_turns from YAML does not raise."""
        raw = {
            "enabled": True,
            "poll_interval_seconds": 60,
            "max_context_turns": 10,   # legacy field — must be silently dropped
            "max_pending_per_agent": 50,
        }
        # Must not raise ValidationError
        cfg = HeartbeatConfig.model_validate(raw)
        assert cfg.enabled is True
        assert cfg.poll_interval_seconds == 60
        assert cfg.max_pending_per_agent == 50
        assert not hasattr(cfg, "max_context_turns")

    def test_max_context_turns_only_is_silently_dropped(self):
        """YAML with only max_context_turns (everything else default) does not raise."""
        raw = {"max_context_turns": 20}
        cfg = HeartbeatConfig.model_validate(raw)
        # Defaults apply
        assert cfg.enabled is False
        assert cfg.poll_interval_seconds == 30
        assert cfg.max_pending_per_agent == 100

    def test_extra_ignore_mode_is_set(self):
        """HeartbeatConfig must have extra='ignore' model config."""
        extra_setting = HeartbeatConfig.model_config.get("extra")
        assert extra_setting == "ignore", (
            f"HeartbeatConfig.model_config['extra'] is {extra_setting!r}, expected 'ignore'. "
            "This is required for backward compat with configs that still have max_context_turns."
        )

    def test_yaml_roundtrip_with_legacy_field(self):
        """Simulates loading a real YAML config that still has max_context_turns."""
        yaml_str = """
enabled: false
poll_interval_seconds: 30
max_context_turns: 10
max_pending_per_agent: 100
"""
        raw = yaml.safe_load(yaml_str)
        cfg = HeartbeatConfig.model_validate(raw)
        assert cfg.enabled is False
        assert cfg.poll_interval_seconds == 30
        assert cfg.max_pending_per_agent == 100
        assert not hasattr(cfg, "max_context_turns")

    def test_arbitrary_extra_fields_are_also_silently_dropped(self):
        """Confirm extra='ignore' applies to any unknown field, not just max_context_turns."""
        raw = {
            "enabled": True,
            "some_future_field": "value",
            "another_unknown": 99,
        }
        cfg = HeartbeatConfig.model_validate(raw)
        assert cfg.enabled is True
        assert not hasattr(cfg, "some_future_field")
        assert not hasattr(cfg, "another_unknown")


# ---------------------------------------------------------------------------
# Tests: agent wiring — max_context_turns sourced from SR2 pipeline config
# ---------------------------------------------------------------------------


def _make_agent_config_dir(raw_window: int) -> str:
    """Create a minimal config_dir for Agent with heartbeat enabled and given raw_window."""
    tmpdir = tempfile.mkdtemp()
    iface_dir = os.path.join(tmpdir, "interfaces")
    os.makedirs(iface_dir)

    with open(os.path.join(tmpdir, "agent.yaml"), "w") as f:
        f.write(f"""
pipeline:
  token_budget: 8000
  compaction:
    enabled: false
    raw_window: {raw_window}
    rules: []
  summarization:
    enabled: false
  retrieval:
    enabled: false
  intent_detection:
    enabled: false
  layers:
    - name: core
      cache_policy: immutable
      contents:
        - key: system_prompt
          source: config
runtime:
  llm:
    model:
      name: "test-model"
    fast_model:
      name: "test-fast-model"
  heartbeat:
    enabled: true
    poll_interval_seconds: 30
    max_pending_per_agent: 100
""")
    return tmpdir


class TestAgentHeartbeatWiring:
    """Agent must source ScheduleHeartbeatTool.max_context_turns from SR2 pipeline config."""

    @pytest.mark.asyncio
    async def test_heartbeat_uses_sr2_raw_window(self):
        """ScheduleHeartbeatTool receives max_context_turns from SR2's compaction.raw_window.

        This verifies the Fix 11 wiring: the agent calls sr2.get_raw_window("heartbeat")
        and passes the result to ScheduleHeartbeatTool — not the removed
        HeartbeatConfig.max_context_turns.
        """
        from sr2_runtime.agent import Agent, AgentConfig

        raw_window = 7  # Distinctive value — not 5 (default) or 10 (old HeartbeatConfig default)
        config_dir = _make_agent_config_dir(raw_window)

        captured_max_context_turns = []

        def capture_schedule_tool(*args, **kwargs):
            captured_max_context_turns.append(kwargs.get("max_context_turns"))
            mock_tool = MagicMock()
            mock_tool.schema.return_value = {}
            return mock_tool

        with patch("sr2_runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")
            mock_llm.set_llm_client = MagicMock()

            agent = Agent(config=AgentConfig(name="test-wiring", config_dir=config_dir))
            agent._loop._llm = mock_llm

            with (
                patch.object(agent, "_ensure_ollama_models", new=AsyncMock()),
                patch("sr2_runtime.heartbeat.ScheduleHeartbeatTool", side_effect=capture_schedule_tool),
                patch("sr2_runtime.heartbeat.CancelHeartbeatTool", return_value=MagicMock()),
                patch("sr2_runtime.heartbeat.HeartbeatScanner") as MockScanner,
                patch("sr2_runtime.heartbeat.InMemoryHeartbeatStore", return_value=MagicMock()),
                patch.object(agent._mcp_manager, "set_llm_client"),
                patch.object(agent._mcp_manager, "connect_all", new=AsyncMock(return_value={})),
                patch.object(agent, "_load_runtime_plugins", new=AsyncMock(return_value=[])),
            ):
                MockScanner.return_value.start = AsyncMock()
                await agent.start()

        assert len(captured_max_context_turns) == 1, (
            "ScheduleHeartbeatTool was not constructed during agent.start()"
        )
        assert captured_max_context_turns[0] == raw_window, (
            f"Expected max_context_turns={raw_window} (from SR2 pipeline config), "
            f"got {captured_max_context_turns[0]}. "
            "The agent must source this value from sr2.get_raw_window('heartbeat'), "
            "not from the removed HeartbeatConfig.max_context_turns."
        )

    @pytest.mark.asyncio
    async def test_heartbeat_uses_default_raw_window_when_no_interface_override(self):
        """When no heartbeat interface config exists, uses base pipeline raw_window."""
        from sr2_runtime.agent import Agent, AgentConfig

        # Default raw_window from CompactionConfig is 5
        config_dir = _make_agent_config_dir(raw_window=5)
        captured = []

        def capture_tool(*args, **kwargs):
            captured.append(kwargs.get("max_context_turns"))
            mock_tool = MagicMock()
            mock_tool.schema.return_value = {}
            return mock_tool

        with patch("sr2_runtime.agent.LLMClient") as MockLLM:
            mock_llm = MockLLM.return_value
            mock_llm.embed = AsyncMock(return_value=[0.0] * 10)
            mock_llm.fast_complete = AsyncMock(return_value="extracted")

            agent = Agent(config=AgentConfig(name="test-default", config_dir=config_dir))
            agent._loop._llm = mock_llm

            with (
                patch.object(agent, "_ensure_ollama_models", new=AsyncMock()),
                patch("sr2_runtime.heartbeat.ScheduleHeartbeatTool", side_effect=capture_tool),
                patch("sr2_runtime.heartbeat.CancelHeartbeatTool", return_value=MagicMock()),
                patch("sr2_runtime.heartbeat.HeartbeatScanner") as MockScanner,
                patch("sr2_runtime.heartbeat.InMemoryHeartbeatStore", return_value=MagicMock()),
                patch.object(agent._mcp_manager, "set_llm_client"),
                patch.object(agent._mcp_manager, "connect_all", new=AsyncMock(return_value={})),
                patch.object(agent, "_load_runtime_plugins", new=AsyncMock(return_value=[])),
            ):
                MockScanner.return_value.start = AsyncMock()
                await agent.start()

        assert captured[0] == 5, (
            f"Expected max_context_turns=5 (base pipeline default), got {captured[0]}"
        )
