"""YAML configuration loader for SR2 v2.

Reads YAML files and validates them against Pydantic models.
Supports config inheritance via `extends` key.

Design principles:
- SRP: Only handles loading and parsing — validation is in models.
- DRY: Single load function handles files, strings, and dicts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sr2.config.models import PipelineConfig


class ConfigLoader:
    """Load and validate SR2 pipeline configuration from YAML."""

    @staticmethod
    def load(source: str | Path | dict[str, Any]) -> PipelineConfig:
        """Load config from a YAML file path, YAML string, or dict.

        Args:
            source: File path, YAML string, or already-parsed dict.

        Returns:
            Validated PipelineConfig.

        Raises:
            ConfigError: If the config is invalid or file not found.
        """
        raw = ConfigLoader._parse(source)
        return ConfigLoader._resolve_extends(raw)

    @staticmethod
    def _parse(source: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Parse source into a dict."""
        if isinstance(source, dict):
            return source

        if isinstance(source, Path) or isinstance(source, str):
            path = Path(source)
            if path.exists():
                text = path.read_text(encoding="utf-8")
            else:
                text = source  # Treat as raw YAML string

            return yaml.safe_load(text) or {}

        raise ValueError(f"Unsupported config source type: {type(source)}")

    @staticmethod
    def _resolve_extends(config: dict[str, Any]) -> PipelineConfig:
        """Resolve `extends` inheritance chain, then validate.

        If the config has an `extends` key, load the base config and
        merge on top. Supports single-level inheritance for now.
        """
        base_path = config.pop("extends", None)
        base = {}

        if base_path:
            base = ConfigLoader._parse(Path(base_path))

        # Merge: base first, then override with derived
        merged = {**base, **config}
        # Deep-merge 'layers' if both define it
        if "layers" in base and "layers" in config:
            merged["layers"] = config["layers"]  # Derived layers fully override

        return PipelineConfig.model_validate(merged)
