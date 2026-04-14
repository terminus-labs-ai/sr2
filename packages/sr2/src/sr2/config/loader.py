import logging
import os
import re
import yaml
from pathlib import Path
from sr2.config.models import PipelineConfig, LLMModelOverride

logger = logging.getLogger(__name__)

# Matches ${VAR} and ${VAR:-default}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+?)(?::-(.*?))?\}")

# Fields from LLMModelOverride that we allow through from sr2_runtime.llm
_LLM_OVERRIDE_FIELDS = set(LLMModelOverride.model_fields.keys())


def _expand_env(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns in a string using os.environ."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        return match.group(0)  # Leave unresolved vars as-is

    return _ENV_VAR_PATTERN.sub(_replace, value)


class ConfigLoader:
    def __init__(self, defaults_path: str | None = None):
        """Initialize with optional path to library defaults YAML."""
        self._defaults_path = defaults_path
        self._loading_chain: set[str] = set()  # For circular detection

    def load(self, config_path: str) -> PipelineConfig:
        """Load a config file, resolve inheritance chain, return validated PipelineConfig."""
        path = Path(config_path).resolve()
        self._loading_chain = set()
        raw = self._load_with_inheritance(path)
        raw.pop("extends", None)

        pipeline_fields = PipelineConfig.model_fields
        # Root-level PipelineConfig fields (e.g. system_prompt at top level)
        root_level = {k: v for k, v in raw.items() if k in pipeline_fields and v is not None}
        # pipeline: dict fields override root-level
        pipeline_raw = raw.get("pipeline", {})
        nested_level = {
            k: v for k, v in pipeline_raw.items() if k in pipeline_fields and v is not None
        }

        # Extract runtime.llm as fallback for PipelineConfig.llm
        runtime_llm = self._extract_llm_override(raw)

        # Merge: root-level first, then pipeline: dict overrides
        merged = {**root_level, **nested_level}

        # Apply runtime.llm as lowest-priority fallback for llm field
        if runtime_llm and "llm" not in nested_level:
            if "llm" in root_level:
                # Root-level llm wins over runtime.llm (deep merge, root on top)
                merged["llm"] = self.merge(runtime_llm, root_level["llm"])
            else:
                merged["llm"] = runtime_llm

        return PipelineConfig(**merged)

    def _extract_llm_override(self, raw: dict) -> dict | None:
        """Extract runtime.llm from raw config dict, filtering to LLMModelOverride fields.

        Returns an LLMConfig-shaped dict (with model/fast_model/embedding keys),
        or None if runtime.llm is absent.
        """
        runtime = raw.get("runtime")
        if not isinstance(runtime, dict):
            return None
        runtime_llm = runtime.get("llm")
        if runtime_llm is not None and not isinstance(runtime_llm, dict):
            logger.warning(
                "runtime.llm config is present but not a dict (got %s), LLM overrides will be ignored",
                type(runtime_llm).__name__,
            )
            return None
        if not isinstance(runtime_llm, dict):
            return None

        result = {}
        for slot in ("model", "fast_model", "embedding"):
            slot_data = runtime_llm.get(slot)
            if isinstance(slot_data, dict):
                filtered = {k: v for k, v in slot_data.items() if k in _LLM_OVERRIDE_FIELDS}
                if filtered:
                    result[slot] = filtered

        return result if result else None

    def _load_with_inheritance(self, path: Path) -> dict:
        """Recursively load and merge configs following extends chain."""
        path_str = str(path)
        if path_str in self._loading_chain:
            raise ValueError(f"Circular config inheritance detected: {path_str}")
        self._loading_chain.add(path_str)

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        config = self.expand_env_vars(config)

        extends = config.get("extends")
        if extends is None:
            return config

        # Resolve the parent path
        parent_path = self._resolve_extends(extends, path)
        parent_config = self._load_with_inheritance(parent_path)

        # Merge: parent is base, current overrides
        merged = self.merge(parent_config, config)
        return merged

    def _resolve_extends(self, extends: str, current_path: Path) -> Path:
        """Resolve an extends value to an absolute path.

        For arbitrary extends paths (not 'defaults' or 'agent' keywords),
        validates that the resolved path does not escape the config directory
        to prevent path traversal attacks via malicious extends values.
        """
        if extends == "defaults":
            if self._defaults_path is None:
                raise FileNotFoundError("No defaults_path configured but config extends 'defaults'")
            return Path(self._defaults_path).resolve()
        elif extends == "agent":
            # Known safe pattern: interface configs extend their parent agent.yaml
            return (current_path.parent.parent / "agent.yaml").resolve()
        else:
            resolved = (current_path.parent / extends).resolve()
            allowed_root = current_path.parent.resolve()
            if not resolved.is_relative_to(allowed_root):
                raise ValueError(
                    f"extends path escapes config directory: '{extends}' "
                    f"resolves to {resolved}, which is outside {allowed_root}"
                )
            return resolved

    def load_from_dict(self, config: dict) -> PipelineConfig:
        """Load from a dict (for testing). No inheritance resolution."""
        return PipelineConfig(**config)

    def expand_env_vars(self, data: object) -> object:
        """Recursively expand ${VAR} and ${VAR:-default} in string values."""
        if isinstance(data, str):
            return _expand_env(data)
        if isinstance(data, dict):
            return {k: self.expand_env_vars(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.expand_env_vars(item) for item in data]
        return data

    def merge(self, base: dict, override: dict) -> dict:
        """Deep merge override into base.

        Rules:
        - Scalar values: override wins
        - Dicts: recursive merge
        - Lists: override replaces entirely
        - None values in override are ignored (don't erase base values)
        """
        result = base.copy()
        for key, value in override.items():
            if value is None:
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(result[key], value)
            else:
                result[key] = value
        return result
