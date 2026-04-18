"""Tests for Toolbox + ToolExecutor + config tools integration.

Verifies the wiring that Agent.__init__() must perform:
- Toolbox registered as a handler in ToolExecutor
- Config tools registered in the Toolbox
- Full-tier schemas returned separately
- End-to-end dispatch through executor -> toolbox -> config tool
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from sr2_runtime.tool_executor import ToolExecutor
from sr2_runtime.toolbox import Toolbox, ToolboxEntry
from sr2_runtime.tools.config_tools import (
    EditConfigTool,
    InspectSchemaTool,
    ReadConfigTool,
    RollbackConfigTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_PIPELINE_YAML = {
    "token_budget": 4096,
    "layers": [
        {
            "name": "core",
            "cache_policy": "immutable",
            "contents": [
                {"key": "system", "source": "static_template", "template": "You are a test agent."}
            ],
        }
    ],
}


def _write_pipeline_yaml(config_dir: Path, filename: str = "chat.yaml") -> Path:
    """Write a minimal valid pipeline YAML and return its path."""
    path = config_dir / filename
    path.write_text(yaml.dump(_MINIMAL_PIPELINE_YAML, default_flow_style=False))
    return path


def _make_sr2_mock() -> MagicMock:
    """Create a mock SR2 facade with reload_interface."""
    sr2 = MagicMock()
    sr2.reload_interface = AsyncMock()
    return sr2


def _wire_config_tools(config_dir: Path, sr2: MagicMock | None = None) -> Toolbox:
    """Create a Toolbox with all 4 config tools registered (mirrors Agent wiring)."""
    if sr2 is None:
        sr2 = _make_sr2_mock()

    toolbox = Toolbox()
    toolbox.register(ToolboxEntry(
        name="read_config",
        one_liner="Read a YAML config file.",
        handler=ReadConfigTool(config_dir),
    ))
    toolbox.register(ToolboxEntry(
        name="edit_config",
        one_liner="Edit a YAML config file with validation and hot-reload.",
        handler=EditConfigTool(config_dir, sr2),
    ))
    toolbox.register(ToolboxEntry(
        name="inspect_schema",
        one_liner="Return the JSON schema for a config section.",
        handler=InspectSchemaTool(),
    ))
    toolbox.register(ToolboxEntry(
        name="rollback_config",
        one_liner="Rollback a config file to a previous version.",
        handler=RollbackConfigTool(config_dir, sr2),
    ))
    return toolbox


def _wire_executor(toolbox: Toolbox) -> ToolExecutor:
    """Register toolbox in a ToolExecutor (mirrors Agent wiring)."""
    executor = ToolExecutor()
    executor.register(Toolbox.TOOL_NAME, toolbox)
    return executor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolboxAgentIntegration:
    """Tests that Toolbox integrates correctly with ToolExecutor and config tools."""

    # --- Registration ---

    def test_toolbox_registered_in_executor(self, tmp_path: Path):
        """After wiring, ToolExecutor has 'toolbox' registered."""
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)
        assert executor.has("toolbox") is True

    def test_config_tools_registered_in_toolbox(self, tmp_path: Path):
        """All 4 config tools are accessible in the toolbox."""
        toolbox = _wire_config_tools(tmp_path)
        assert toolbox.has("read_config")
        assert toolbox.has("edit_config")
        assert toolbox.has("inspect_schema")
        assert toolbox.has("rollback_config")

    # --- Schema ---

    def test_toolbox_schema_has_correct_name(self, tmp_path: Path):
        """Toolbox tool_definition has name='toolbox'."""
        toolbox = _wire_config_tools(tmp_path)
        assert toolbox.tool_definition["name"] == "toolbox"

    def test_toolbox_schema_is_compact(self, tmp_path: Path):
        """Toolbox schema stays under 100 tokens (~400 chars)."""
        toolbox = _wire_config_tools(tmp_path)
        schema_json = json.dumps(toolbox.tool_definition)
        token_estimate = len(schema_json) / 4
        assert token_estimate < 100, f"Schema ~{token_estimate:.0f} tokens, limit 100"

    def test_full_tier_schemas_separate_from_toolbox(self, tmp_path: Path):
        """Full-tier tool schemas come from get_full_tier_schemas(), not list."""
        toolbox = _wire_config_tools(tmp_path)

        # Add a full-tier tool
        full_handler = MagicMock()
        full_handler.tool_definition = {
            "name": "save_memory",
            "description": "Save a memory",
            "parameters": {"type": "object", "properties": {}},
        }
        toolbox.register(ToolboxEntry(
            name="save_memory",
            one_liner="Save a memory",
            handler=full_handler,
            tier="full",
        ))

        # Full-tier schemas include it
        full_schemas = toolbox.get_full_tier_schemas()
        full_names = [s["name"] for s in full_schemas]
        assert "save_memory" in full_names

        # Config tools are NOT in full-tier
        assert "read_config" not in full_names

    # --- Dispatch through executor ---

    @pytest.mark.asyncio
    async def test_executor_dispatches_toolbox_list(self, tmp_path: Path):
        """Executor.execute('toolbox', {action: 'list'}) returns config tool listing."""
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        result = await executor.execute("toolbox", {"action": "list"})
        assert "read_config" in result
        assert "edit_config" in result
        assert "inspect_schema" in result
        assert "rollback_config" in result

    @pytest.mark.asyncio
    async def test_executor_dispatches_toolbox_describe(self, tmp_path: Path):
        """Executor -> toolbox describe returns the inner tool's schema."""
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        result = await executor.execute(
            "toolbox", {"action": "describe", "tool": "read_config"}
        )
        schema = json.loads(result)
        assert schema["name"] == "read_config"
        assert "file" in json.dumps(schema["parameters"])

    @pytest.mark.asyncio
    async def test_executor_dispatches_toolbox_use_read_config(self, tmp_path: Path):
        """Executor -> toolbox use -> read_config reads a YAML file."""
        _write_pipeline_yaml(tmp_path, "chat.yaml")
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        result = await executor.execute("toolbox", {
            "action": "use",
            "tool": "read_config",
            "arguments": {"file": "chat.yaml"},
        })
        assert "token_budget" in result
        assert "4096" in result

    @pytest.mark.asyncio
    async def test_executor_dispatches_toolbox_use_read_config_path(self, tmp_path: Path):
        """Executor -> toolbox use -> read_config with dot-path navigates into YAML."""
        _write_pipeline_yaml(tmp_path, "chat.yaml")
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        result = await executor.execute("toolbox", {
            "action": "use",
            "tool": "read_config",
            "arguments": {"file": "chat.yaml", "path": "token_budget"},
        })
        assert "4096" in result

    @pytest.mark.asyncio
    async def test_executor_dispatches_toolbox_use_inspect_schema(self, tmp_path: Path):
        """Executor -> toolbox use -> inspect_schema returns JSON schema."""
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        result = await executor.execute("toolbox", {
            "action": "use",
            "tool": "inspect_schema",
            "arguments": {"section": "compaction"},
        })
        schema = json.loads(result)
        assert "properties" in schema
        assert schema["title"] == "CompactionConfig"

    # --- End-to-end edit ---

    @pytest.mark.asyncio
    async def test_end_to_end_edit_via_toolbox(self, tmp_path: Path):
        """Full flow: executor -> toolbox -> edit_config modifies file on disk."""
        _write_pipeline_yaml(tmp_path, "chat.yaml")
        sr2 = _make_sr2_mock()
        toolbox = _wire_config_tools(tmp_path, sr2)
        executor = _wire_executor(toolbox)

        result = await executor.execute("toolbox", {
            "action": "use",
            "tool": "edit_config",
            "arguments": {
                "file": "chat.yaml",
                "edits": [{"path": "token_budget", "value": 8192}],
            },
        })
        assert "OK" in result

        # Verify file changed on disk
        updated = yaml.safe_load((tmp_path / "chat.yaml").read_text())
        assert updated["token_budget"] == 8192

        # Verify reload was called
        sr2.reload_interface.assert_called_once_with("chat")

    @pytest.mark.asyncio
    async def test_end_to_end_edit_creates_backup(self, tmp_path: Path):
        """Edit through toolbox creates a backup in .config_history/."""
        _write_pipeline_yaml(tmp_path, "chat.yaml")
        sr2 = _make_sr2_mock()
        toolbox = _wire_config_tools(tmp_path, sr2)
        executor = _wire_executor(toolbox)

        await executor.execute("toolbox", {
            "action": "use",
            "tool": "edit_config",
            "arguments": {
                "file": "chat.yaml",
                "edits": [{"path": "token_budget", "value": 8192}],
            },
        })

        backup_dir = tmp_path / ".config_history" / "chat"
        assert backup_dir.exists()
        backups = list(backup_dir.iterdir())
        assert len(backups) == 1

    @pytest.mark.asyncio
    async def test_end_to_end_rollback_via_toolbox(self, tmp_path: Path):
        """Full flow: edit then rollback restores original content."""
        _write_pipeline_yaml(tmp_path, "chat.yaml")
        sr2 = _make_sr2_mock()
        toolbox = _wire_config_tools(tmp_path, sr2)
        executor = _wire_executor(toolbox)

        # Edit
        await executor.execute("toolbox", {
            "action": "use",
            "tool": "edit_config",
            "arguments": {
                "file": "chat.yaml",
                "edits": [{"path": "token_budget", "value": 8192}],
            },
        })

        # Rollback
        result = await executor.execute("toolbox", {
            "action": "use",
            "tool": "rollback_config",
            "arguments": {"file": "chat.yaml", "version": 1},
        })
        assert "OK" in result

        # Verify original content restored
        restored = yaml.safe_load((tmp_path / "chat.yaml").read_text())
        assert restored["token_budget"] == 4096

    # --- Call tracking ---

    @pytest.mark.asyncio
    async def test_executor_tracks_toolbox_calls(self, tmp_path: Path):
        """ToolExecutor call count increments for toolbox dispatches."""
        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        await executor.execute("toolbox", {"action": "list"})
        await executor.execute("toolbox", {"action": "list"})

        assert executor.get_call_count("toolbox") == 2

    # --- Subdirectory config files ---

    @pytest.mark.asyncio
    async def test_read_config_in_subdirectory(self, tmp_path: Path):
        """Toolbox can read config files in subdirectories of config_dir."""
        interfaces_dir = tmp_path / "interfaces"
        interfaces_dir.mkdir()
        _write_pipeline_yaml(tmp_path / "interfaces", "chat.yaml")

        toolbox = _wire_config_tools(tmp_path)
        executor = _wire_executor(toolbox)

        result = await executor.execute("toolbox", {
            "action": "use",
            "tool": "read_config",
            "arguments": {"file": "interfaces/chat.yaml"},
        })
        assert "token_budget" in result
