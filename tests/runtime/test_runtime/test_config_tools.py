"""Tests for self-modification config tools."""

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from sr2_runtime.tools.config_tools import (
    EditConfigTool,
    InspectSchemaTool,
    ReadConfigTool,
    RollbackConfigTool,
    resolve_and_guard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CONFIG = {
    "token_budget": 16000,
    "summarization": {
        "enabled": True,
        "trigger": "token_threshold",
    },
}


@pytest.fixture()
def config_dir(tmp_path: Path) -> Path:
    """Create an interfaces directory and return the root config dir."""
    (tmp_path / "interfaces").mkdir()
    return tmp_path


@pytest.fixture()
def sample_config(config_dir: Path) -> Path:
    """Write a valid PipelineConfig YAML to interfaces/chat.yaml."""
    path = config_dir / "interfaces" / "chat.yaml"
    path.write_text(yaml.dump(SAMPLE_CONFIG))
    return path


@pytest.fixture()
def mock_sr2() -> MagicMock:
    """Mock SR2 facade with a no-op reload_interface."""
    sr2 = MagicMock()
    sr2.reload_interface = AsyncMock(return_value=None)
    return sr2


# ---------------------------------------------------------------------------
# TestResolveAndGuard
# ---------------------------------------------------------------------------


class TestResolveAndGuard:
    """Tests for the resolve_and_guard helper."""

    def test_resolves_relative_path(self, config_dir: Path, sample_config: Path):
        """Valid relative path returns the resolved absolute Path."""
        result = resolve_and_guard(config_dir, "interfaces/chat.yaml")
        assert result == sample_config
        assert result.is_absolute()

    def test_blocks_traversal(self, config_dir: Path):
        """Directory traversal outside config_dir raises ValueError."""
        with pytest.raises(ValueError, match="[Tt]raversal|[Oo]utside"):
            resolve_and_guard(config_dir, "../../etc/passwd")

    def test_nonexistent_file_raises(self, config_dir: Path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            resolve_and_guard(config_dir, "interfaces/nope.yaml")


# ---------------------------------------------------------------------------
# TestReadConfigTool
# ---------------------------------------------------------------------------


class TestReadConfigTool:
    """Tests for ReadConfigTool."""

    def test_has_tool_definition(self, config_dir: Path):
        """tool_definition returns a schema with name='read_config'."""
        tool = ReadConfigTool(config_dir)
        defn = tool.tool_definition
        assert defn["name"] == "read_config"

    @pytest.mark.asyncio
    async def test_reads_full_file(self, config_dir: Path, sample_config: Path):
        """Reading a file with no path returns the full YAML content."""
        tool = ReadConfigTool(config_dir)
        result = await tool.execute(file="interfaces/chat.yaml")
        parsed = yaml.safe_load(result)
        assert parsed["token_budget"] == 16000

    @pytest.mark.asyncio
    async def test_reads_subsection_with_dot_path(
        self, config_dir: Path, sample_config: Path
    ):
        """Dot-notation path navigates into nested YAML."""
        tool = ReadConfigTool(config_dir)
        result = await tool.execute(file="interfaces/chat.yaml", path="summarization")
        parsed = yaml.safe_load(result)
        assert parsed["enabled"] is True
        assert parsed["trigger"] == "token_threshold"

    @pytest.mark.asyncio
    async def test_invalid_dot_path_returns_error(
        self, config_dir: Path, sample_config: Path
    ):
        """Non-existent dot path returns an error string (no exception)."""
        tool = ReadConfigTool(config_dir)
        result = await tool.execute(
            file="interfaces/chat.yaml", path="pipeline.nonexistent.deep"
        )
        assert "error" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self, config_dir: Path):
        """Missing file returns an error string (no exception)."""
        tool = ReadConfigTool(config_dir)
        result = await tool.execute(file="interfaces/missing.yaml")
        assert "error" in result.lower() or "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_returns_error(self, config_dir: Path):
        """Traversal attempt returns an error string (no exception)."""
        tool = ReadConfigTool(config_dir)
        result = await tool.execute(file="../../etc/passwd")
        assert "error" in result.lower() or "traversal" in result.lower()


# ---------------------------------------------------------------------------
# TestEditConfigTool
# ---------------------------------------------------------------------------


class TestEditConfigTool:
    """Tests for EditConfigTool."""

    def test_has_tool_definition(self, config_dir: Path, mock_sr2: MagicMock):
        """tool_definition returns a schema with name='edit_config'."""
        tool = EditConfigTool(config_dir, mock_sr2)
        defn = tool.tool_definition
        assert defn["name"] == "edit_config"

    @pytest.mark.asyncio
    async def test_edits_single_field(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Editing token_budget persists the change to disk."""
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": 32000}],
        )
        assert "error" not in result.lower()

        data = yaml.safe_load(sample_config.read_text())
        assert data["token_budget"] == 32000

    @pytest.mark.asyncio
    async def test_edits_nested_path(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Editing a nested dot-path like 'summarization.enabled' works."""
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "summarization.enabled", "value": False}],
        )
        assert "error" not in result.lower()

        data = yaml.safe_load(sample_config.read_text())
        assert data["summarization"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_edit_delete_key(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Delete operation removes the key from the YAML."""
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "summarization", "delete": True}],
        )
        assert "error" not in result.lower()

        data = yaml.safe_load(sample_config.read_text())
        assert "summarization" not in data

    @pytest.mark.asyncio
    async def test_protected_path_rejected(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Editing 'system_prompt' returns an error, file unchanged."""
        original = sample_config.read_text()
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "system_prompt", "value": "hacked"}],
        )
        assert "error" in result.lower() or "protected" in result.lower()
        assert sample_config.read_text() == original

    @pytest.mark.asyncio
    async def test_protected_path_extends_rejected(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Editing 'extends' returns an error, file unchanged."""
        original = sample_config.read_text()
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "extends", "value": "evil"}],
        )
        assert "error" in result.lower() or "protected" in result.lower()
        assert sample_config.read_text() == original

    @pytest.mark.asyncio
    async def test_invalid_value_rejected_by_pydantic(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Invalid value that fails PipelineConfig validation returns error, no write."""
        original = sample_config.read_text()
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": "garbage"}],
        )
        assert "error" in result.lower() or "valid" in result.lower()
        assert sample_config.read_text() == original

    @pytest.mark.asyncio
    async def test_creates_backup(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """After a successful edit, a backup file exists in .config_history/."""
        tool = EditConfigTool(config_dir, mock_sr2)
        await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": 8000}],
        )

        backup_dir = config_dir / ".config_history" / "chat"
        assert backup_dir.exists()
        backups = list(backup_dir.iterdir())
        assert len(backups) >= 1

    @pytest.mark.asyncio
    async def test_prunes_old_backups(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """After many edits, only MAX_VERSIONS (5) backups remain."""
        tool = EditConfigTool(config_dir, mock_sr2)
        for i in range(7):
            await tool.execute(
                file="interfaces/chat.yaml",
                edits=[{"path": "token_budget", "value": 10000 + i * 1000}],
            )
            # Small delay so timestamps differ if needed
            time.sleep(0.01)

        backup_dir = config_dir / ".config_history" / "chat"
        backups = list(backup_dir.iterdir())
        assert len(backups) == EditConfigTool.MAX_VERSIONS

    @pytest.mark.asyncio
    async def test_reverts_on_reload_failure(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """If sr2.reload_interface raises, the original file is restored."""
        original = sample_config.read_text()
        mock_sr2.reload_interface = AsyncMock(
            side_effect=RuntimeError("reload boom")
        )

        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": 64000}],
        )
        assert "error" in result.lower()
        assert sample_config.read_text() == original

    @pytest.mark.asyncio
    async def test_agent_yaml_rejected(
        self, config_dir: Path, mock_sr2: MagicMock
    ):
        """Editing agent.yaml is not supported in Phase 1."""
        agent_yaml = config_dir / "agent.yaml"
        agent_yaml.write_text(yaml.dump({"name": "test-agent"}))

        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="agent.yaml",
            edits=[{"path": "name", "value": "evil"}],
        )
        assert "error" in result.lower() or "not supported" in result.lower()


# ---------------------------------------------------------------------------
# TestInspectSchemaTool
# ---------------------------------------------------------------------------


class TestInspectSchemaTool:
    """Tests for InspectSchemaTool."""

    def test_has_tool_definition(self):
        """tool_definition returns a schema with name='inspect_schema'."""
        tool = InspectSchemaTool()
        defn = tool.tool_definition
        assert defn["name"] == "inspect_schema"

    @pytest.mark.asyncio
    async def test_returns_pipeline_schema(self):
        """section='pipeline' returns valid JSON with PipelineConfig fields."""
        tool = InspectSchemaTool()
        result = await tool.execute(section="pipeline")
        schema = json.loads(result)
        assert "properties" in schema
        assert "token_budget" in schema["properties"]

    @pytest.mark.asyncio
    async def test_returns_compaction_schema(self):
        """section='compaction' returns valid JSON with CompactionConfig fields."""
        tool = InspectSchemaTool()
        result = await tool.execute(section="compaction")
        schema = json.loads(result)
        assert "properties" in schema
        assert "enabled" in schema["properties"]

    @pytest.mark.asyncio
    async def test_unknown_section_returns_error(self):
        """Unknown section returns an error listing available sections."""
        tool = InspectSchemaTool()
        result = await tool.execute(section="nonsense")
        lower = result.lower()
        assert "error" in lower or "unknown" in lower
        # Should list available sections
        assert "pipeline" in lower


# ---------------------------------------------------------------------------
# TestRollbackConfigTool
# ---------------------------------------------------------------------------


class TestRollbackConfigTool:
    """Tests for RollbackConfigTool."""

    def test_has_tool_definition(self, config_dir: Path, mock_sr2: MagicMock):
        """tool_definition returns a schema with name='rollback_config'."""
        tool = RollbackConfigTool(config_dir, mock_sr2)
        defn = tool.tool_definition
        assert defn["name"] == "rollback_config"

    @pytest.mark.asyncio
    async def test_rollback_restores_previous(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """After editing, rollback restores the original content."""
        original_data = yaml.safe_load(sample_config.read_text())

        # Make an edit to create a backup
        edit_tool = EditConfigTool(config_dir, mock_sr2)
        await edit_tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": 99000}],
        )
        # Verify the edit took effect
        assert yaml.safe_load(sample_config.read_text())["token_budget"] == 99000

        # Rollback
        rollback_tool = RollbackConfigTool(config_dir, mock_sr2)
        result = await rollback_tool.execute(file="interfaces/chat.yaml", version=1)
        assert "error" not in result.lower()

        restored = yaml.safe_load(sample_config.read_text())
        assert restored["token_budget"] == original_data["token_budget"]

    @pytest.mark.asyncio
    async def test_rollback_is_reversible(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """After rollback, the pre-rollback state was backed up too."""
        edit_tool = EditConfigTool(config_dir, mock_sr2)
        await edit_tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": 50000}],
        )

        rollback_tool = RollbackConfigTool(config_dir, mock_sr2)
        await rollback_tool.execute(file="interfaces/chat.yaml", version=1)

        # A new backup should have been created during rollback
        backup_dir = config_dir / ".config_history" / "chat"
        backups = sorted(backup_dir.iterdir())
        # At least 2: original edit backup + pre-rollback backup
        assert len(backups) >= 2

    @pytest.mark.asyncio
    async def test_no_backups_returns_error(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Rollback with no backups returns an error string."""
        tool = RollbackConfigTool(config_dir, mock_sr2)
        result = await tool.execute(file="interfaces/chat.yaml", version=1)
        assert "error" in result.lower() or "no backup" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_version_returns_error(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """version=99 when only a couple backups exist returns error."""
        # Create one backup
        edit_tool = EditConfigTool(config_dir, mock_sr2)
        await edit_tool.execute(
            file="interfaces/chat.yaml",
            edits=[{"path": "token_budget", "value": 8000}],
        )

        tool = RollbackConfigTool(config_dir, mock_sr2)
        result = await tool.execute(file="interfaces/chat.yaml", version=99)
        assert "error" in result.lower() or "version" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_backup_content_returns_error(
        self, config_dir: Path, sample_config: Path, mock_sr2: MagicMock
    ):
        """Backup YAML that fails PipelineConfig validation is not restored."""
        # Manually create a bad backup
        backup_dir = config_dir / ".config_history" / "chat"
        backup_dir.mkdir(parents=True, exist_ok=True)
        bad_backup = backup_dir / "chat.yaml.20260101T000000"
        bad_backup.write_text(yaml.dump({"token_budget": "not_a_number"}))

        original = sample_config.read_text()

        tool = RollbackConfigTool(config_dir, mock_sr2)
        result = await tool.execute(file="interfaces/chat.yaml", version=1)
        assert "error" in result.lower() or "valid" in result.lower()
        # File should be unchanged
        assert sample_config.read_text() == original


# ---------------------------------------------------------------------------
# TestRealWorldConfigStructure
# ---------------------------------------------------------------------------

# Interface configs in production use extends: + pipeline: wrapper
REAL_INTERFACE_CONFIG = {
    "extends": "agent",
    "pipeline": {
        "token_budget": 8192,
        "summarization": {
            "enabled": False,
        },
        "retrieval": {
            "enabled": False,
        },
    },
}


class TestRealWorldConfigStructure:
    """Tests for configs that use extends: + pipeline: wrapper (real-world format)."""

    @pytest.fixture()
    def real_config(self, config_dir: Path) -> Path:
        path = config_dir / "interfaces" / "proactive.yaml"
        path.write_text(yaml.dump(REAL_INTERFACE_CONFIG))
        return path

    @pytest.mark.asyncio
    async def test_edit_inside_pipeline_wrapper(
        self, config_dir: Path, real_config: Path, mock_sr2: MagicMock
    ):
        """Editing pipeline.token_budget works with the pipeline: wrapper."""
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/proactive.yaml",
            edits=[{"path": "pipeline.token_budget", "value": 16000}],
        )
        assert "error" not in result.lower()
        data = yaml.safe_load(real_config.read_text())
        assert data["pipeline"]["token_budget"] == 16000

    @pytest.mark.asyncio
    async def test_edit_nested_inside_pipeline_wrapper(
        self, config_dir: Path, real_config: Path, mock_sr2: MagicMock
    ):
        """Editing pipeline.summarization.enabled works with nested paths."""
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/proactive.yaml",
            edits=[{"path": "pipeline.summarization.enabled", "value": True}],
        )
        assert "error" not in result.lower()
        data = yaml.safe_load(real_config.read_text())
        assert data["pipeline"]["summarization"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_edit_adds_new_key_inside_pipeline(
        self, config_dir: Path, real_config: Path, mock_sr2: MagicMock
    ):
        """Adding a new key (e.g. preserve_recent_turns) that wasn't in the file."""
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/proactive.yaml",
            edits=[
                {"path": "pipeline.summarization.preserve_recent_turns", "value": 8}
            ],
        )
        assert "error" not in result.lower()
        data = yaml.safe_load(real_config.read_text())
        assert data["pipeline"]["summarization"]["preserve_recent_turns"] == 8

    @pytest.mark.asyncio
    async def test_extends_key_preserved_after_edit(
        self, config_dir: Path, real_config: Path, mock_sr2: MagicMock
    ):
        """The extends: key is preserved after editing other fields."""
        tool = EditConfigTool(config_dir, mock_sr2)
        await tool.execute(
            file="interfaces/proactive.yaml",
            edits=[{"path": "pipeline.token_budget", "value": 4096}],
        )
        data = yaml.safe_load(real_config.read_text())
        assert data["extends"] == "agent"

    @pytest.mark.asyncio
    async def test_invalid_value_in_pipeline_wrapper_rejected(
        self, config_dir: Path, real_config: Path, mock_sr2: MagicMock
    ):
        """Invalid value inside pipeline: wrapper is caught by validation."""
        original = real_config.read_text()
        tool = EditConfigTool(config_dir, mock_sr2)
        result = await tool.execute(
            file="interfaces/proactive.yaml",
            edits=[{"path": "pipeline.token_budget", "value": "garbage"}],
        )
        assert "error" in result.lower()
        assert real_config.read_text() == original
