"""Self-modification config tools for SR2 agents.

Provides read, edit, rollback, and schema inspection tools that let agents
modify their own pipeline configuration at runtime.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML

from sr2.config.loader import ConfigLoader
from sr2.config.models import (
    CompactionConfig,
    PipelineConfig,
    RetrievalConfig,
    SummarizationConfig,
)
from sr2_runtime.config import AgentYAMLConfig


def resolve_and_guard(config_dir: Path, file: str) -> Path:
    """Resolve *file* relative to *config_dir*, blocking directory traversal.

    Raises:
        ValueError: If the resolved path escapes *config_dir*.
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = (config_dir / file).resolve()
    try:
        resolved.relative_to(config_dir.resolve())
    except ValueError:
        raise ValueError(
            f"Path traversal detected: '{file}' resolves outside config directory"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BACKUP_DIR_NAME = ".config_history"


def _backup_dir_for(config_dir: Path, file: str) -> Path:
    stem = Path(file).stem
    return config_dir / _BACKUP_DIR_NAME / stem


def _create_backup(
    config_dir: Path, file: str, content: str, max_versions: int
) -> None:
    """Save *content* as a timestamped backup and prune old versions."""
    bdir = _backup_dir_for(config_dir, file)
    bdir.mkdir(parents=True, exist_ok=True)
    filename = Path(file).name
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    (bdir / f"{filename}.{ts}").write_text(content)
    _prune_backups(bdir, filename, max_versions)


def _prune_backups(bdir: Path, filename: str, max_versions: int) -> None:
    backups = sorted(bdir.glob(f"{filename}.*"))
    while len(backups) > max_versions:
        backups.pop(0).unlink()


def _navigate_dot_path(data: dict, path: str) -> Any:
    """Walk *data* following a dot-separated *path*."""
    current = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Key not found: '{key}' in path '{path}'")
        current = current[key]
    return current


def _set_dot_path(data: dict, path: str, value: Any) -> None:
    """Set a value in *data* at the given dot-separated *path*."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _delete_dot_path(data: dict, path: str) -> None:
    """Delete a key from *data* at the given dot-separated *path*."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current[key]
    del current[keys[-1]]


def _ruamel_yaml() -> YAML:
    """Create a ruamel.yaml instance that preserves formatting."""
    ry = YAML()
    ry.preserve_quotes = True
    ry.width = 120
    return ry


def _ruamel_load(text: str) -> Any:
    """Load YAML text via ruamel.yaml, preserving order and comments."""
    return _ruamel_yaml().load(text)


def _ruamel_dump(data: Any) -> str:
    """Dump data via ruamel.yaml, preserving order and comments."""
    from io import StringIO
    stream = StringIO()
    _ruamel_yaml().dump(data, stream)
    return stream.getvalue()


def _set_dot_path_ruamel(data: Any, path: str, value: Any) -> None:
    """Set a value at a dot-path in a ruamel.yaml CommentedMap.

    Creates intermediate maps as needed, preserving existing structure.
    """
    from ruamel.yaml.comments import CommentedMap

    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = CommentedMap()
        current = current[key]
    current[keys[-1]] = value


def _delete_dot_path_ruamel(data: Any, path: str) -> None:
    """Delete a key at a dot-path in a ruamel.yaml CommentedMap."""
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        current = current[key]
    del current[keys[-1]]


async def _call_reload(sr2: Any, interface_name: str) -> None:
    """Call sr2.reload_interface, handling both sync and async."""
    result = sr2.reload_interface(interface_name)
    if inspect.isawaitable(result):
        await result


# ---------------------------------------------------------------------------
# ReadConfigTool
# ---------------------------------------------------------------------------


class ReadConfigTool:
    """Read YAML configuration files, optionally navigating to a sub-path."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "read_config",
            "description": (
                "Read your own YAML config files. "
                "Call with file='.' to list all available config files. "
                "Use 'path' to read a specific subsection (dot-notation)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": (
                            "Relative path to the YAML file. "
                            "Use '.' to list available files. "
                            "Examples: 'interfaces/user_message.yaml', 'interfaces/proactive.yaml', 'agent.yaml'"
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Optional dot-separated path to a subsection. "
                            "Examples: 'pipeline.summarization', 'pipeline.token_budget', 'runtime.llm'"
                        ),
                    },
                },
                "required": ["file"],
            },
        }

    async def execute(self, file: str, path: str | None = None) -> str:
        # List available files
        if file == ".":
            return self._list_files()
        try:
            resolved = resolve_and_guard(self._config_dir, file)
            data = yaml.safe_load(resolved.read_text())
            if path:
                data = _navigate_dot_path(data, path)
            return yaml.dump(data, default_flow_style=False)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return f"Error: {exc}"

    def _list_files(self) -> str:
        """List all YAML files in the config directory."""
        files = []
        for p in sorted(self._config_dir.rglob("*.yaml")):
            rel = p.relative_to(self._config_dir)
            # Skip backup directory
            if _BACKUP_DIR_NAME in str(rel):
                continue
            files.append(str(rel))
        if not files:
            return "No config files found."
        return "Available config files:\n" + "\n".join(f"- {f}" for f in files)


# ---------------------------------------------------------------------------
# EditConfigTool
# ---------------------------------------------------------------------------


class EditConfigTool:
    """Edit YAML configuration with validation, backup, and hot-reload."""

    BACKUP_DIR_NAME: str = ".config_history"
    MAX_VERSIONS: int = 5
    PROTECTED_PATHS: frozenset[str] = frozenset({"system_prompt", "extends"})

    def __init__(self, config_dir: Path, sr2: Any) -> None:
        self._config_dir = config_dir
        self._sr2 = sr2

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "edit_config",
            "description": (
                "Edit your own interface config files. Changes are validated, backed up, "
                "and hot-reloaded automatically. Use read_config first to see current values. "
                "Protected fields (system_prompt, extends) cannot be modified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": (
                            "Relative path to the interface YAML file. "
                            "Example: 'interfaces/user_message.yaml'. "
                            "Note: agent.yaml cannot be edited in this version."
                        ),
                    },
                    "edits": {
                        "type": "array",
                        "description": (
                            "List of edits to apply. Each edit has a dot-path and a value. "
                            "Paths match the YAML structure. "
                            "Examples: "
                            "{'path': 'pipeline.token_budget', 'value': 16000}, "
                            "{'path': 'pipeline.summarization.enabled', 'value': true}, "
                            "{'path': 'pipeline.retrieval.top_k', 'value': 10}, "
                            "{'path': 'pipeline.compaction.raw_window', 'delete': true}"
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Dot-separated path in the YAML. Must include 'pipeline.' prefix for pipeline settings.",
                                },
                                "value": {"description": "New value to set."},
                                "delete": {"type": "boolean", "description": "Set true to remove the key instead."},
                            },
                            "required": ["path"],
                        },
                    },
                },
                "required": ["file", "edits"],
            },
        }

    async def execute(self, file: str, edits: list[dict]) -> str:
        # Block agent.yaml
        if Path(file).name == "agent.yaml":
            return "Error: Editing agent.yaml is not supported in Phase 1."

        # Check protected paths
        for edit in edits:
            root_key = edit["path"].split(".")[0]
            if root_key in self.PROTECTED_PATHS:
                return f"Error: Path '{root_key}' is protected and cannot be edited."

        try:
            resolved = resolve_and_guard(self._config_dir, file)
        except (FileNotFoundError, ValueError) as exc:
            return f"Error: {exc}"

        original_content = resolved.read_text()

        # Use ruamel.yaml for order-preserving edits
        data = _ruamel_load(original_content)

        # Apply edits (ruamel-aware helpers preserve structure)
        for edit in edits:
            if edit.get("delete"):
                _delete_dot_path_ruamel(data, edit["path"])
            else:
                _set_dot_path_ruamel(data, edit["path"], edit["value"])

        # Validate using plain dicts (Pydantic needs regular dicts)
        new_content = _ruamel_dump(data)
        plain_data = yaml.safe_load(new_content)
        try:
            self._validate(file, plain_data, resolved)
        except Exception as exc:
            return f"Error: Validation failed — {exc}"

        # Create backup of current content
        _create_backup(self._config_dir, file, original_content, self.MAX_VERSIONS)

        # Write new content (preserves ordering, comments, formatting)
        resolved.write_text(new_content)

        # Reload
        interface_name = Path(file).stem
        try:
            await _call_reload(self._sr2, interface_name)
        except Exception as exc:
            # Revert
            resolved.write_text(original_content)
            return f"Error: Reload failed — {exc}"

        return f"OK: Applied {len(edits)} edit(s) to {file}."

    def _validate(self, file: str, data: dict, resolved: Path) -> None:
        """Validate config data, rejecting unknown fields.

        Uses a strict copy of PipelineConfig with extra='forbid' so
        misplaced keys (e.g. pipeline.preserve_recent_turns instead of
        pipeline.summarization.preserve_recent_turns) are caught.
        """
        filename = Path(file).name
        if filename == "agent.yaml":
            AgentYAMLConfig(**data)
        else:
            pipeline_data = data.get("pipeline", {})
            if pipeline_data:
                pipeline_fields = set(PipelineConfig.model_fields.keys())
                merged = {
                    k: v for k, v in data.items()
                    if k in pipeline_fields and k != "extends"
                }
                merged.update(pipeline_data)
                _validate_strict(merged)
            else:
                clean = {k: v for k, v in data.items() if k != "extends"}
                _validate_strict(clean)


# Strict PipelineConfig that rejects unknown fields at the top level.
# We only forbid extras on the root — nested models keep their own
# extra policy so provider-specific pass-through fields still work.
_StrictPipelineConfig = None


def _validate_strict(data: dict) -> None:
    """Validate *data* as PipelineConfig with extra='forbid' at root level."""
    global _StrictPipelineConfig
    if _StrictPipelineConfig is None:
        # Build once, cache. Dynamic subclass with extra='forbid'.
        _StrictPipelineConfig = type(
            "_StrictPipelineConfig",
            (PipelineConfig,),
            {"model_config": {**PipelineConfig.model_config, "extra": "forbid"}},
        )
        _StrictPipelineConfig.model_rebuild()
    _StrictPipelineConfig(**data)


# ---------------------------------------------------------------------------
# InspectSchemaTool
# ---------------------------------------------------------------------------

_SECTION_MODELS: dict[str, type] = {
    "pipeline": PipelineConfig,
    "compaction": CompactionConfig,
    "summarization": SummarizationConfig,
    "retrieval": RetrievalConfig,
}


class InspectSchemaTool:
    """Return the JSON schema for a config section."""

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "inspect_schema",
            "description": "Return the JSON schema for a configuration section.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": list(_SECTION_MODELS.keys()),
                    },
                },
            },
        }

    async def execute(self, section: str = "pipeline") -> str:
        model = _SECTION_MODELS.get(section)
        if model is None:
            available = ", ".join(sorted(_SECTION_MODELS.keys()))
            return f"Error: Unknown section '{section}'. Available: {available}"
        return json.dumps(model.model_json_schema(), indent=2)


# ---------------------------------------------------------------------------
# RollbackConfigTool
# ---------------------------------------------------------------------------


class RollbackConfigTool:
    """Rollback a config file to a previous backup version."""

    def __init__(self, config_dir: Path, sr2: Any) -> None:
        self._config_dir = config_dir
        self._sr2 = sr2

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "rollback_config",
            "description": "Rollback a config file to a previous version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "version": {"type": "integer", "default": 1},
                },
                "required": ["file"],
            },
        }

    async def execute(self, file: str, version: int = 1) -> str:
        try:
            resolved = resolve_and_guard(self._config_dir, file)
        except (FileNotFoundError, ValueError) as exc:
            return f"Error: {exc}"

        filename = Path(file).name
        bdir = _backup_dir_for(self._config_dir, file)

        if not bdir.exists():
            return "Error: No backups found for this file."

        backups = sorted(bdir.glob(f"{filename}.*"))
        if not backups:
            return "Error: No backups found for this file."

        if version < 1 or version > len(backups):
            return f"Error: Invalid version {version}. Available: 1–{len(backups)}."

        # version=1 means most recent (last in sorted list)
        target = backups[-version]
        backup_content = target.read_text()

        # Validate backup content
        try:
            data = yaml.safe_load(backup_content)
            # Use same validation as EditConfigTool
            pipeline_data = data.get("pipeline", {})
            if pipeline_data:
                pipeline_fields = set(PipelineConfig.model_fields.keys())
                merged = {
                    k: v for k, v in data.items()
                    if k in pipeline_fields and k != "extends"
                }
                merged.update(pipeline_data)
                PipelineConfig(**merged)
            else:
                clean = {k: v for k, v in data.items() if k != "extends"}
                PipelineConfig(**clean)
        except Exception as exc:
            return f"Error: Backup validation failed — {exc}"

        # Back up current state before restoring (makes rollback reversible)
        current_content = resolved.read_text()
        _create_backup(
            self._config_dir, file, current_content, EditConfigTool.MAX_VERSIONS
        )

        # Restore
        resolved.write_text(backup_content)

        # Reload
        interface_name = Path(file).stem
        try:
            await _call_reload(self._sr2, interface_name)
        except Exception as exc:
            resolved.write_text(current_content)
            return f"Error: Reload failed — {exc}"

        return f"OK: Rolled back {file} to version {version}."
