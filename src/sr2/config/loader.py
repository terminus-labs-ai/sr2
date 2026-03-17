import yaml
from pathlib import Path
from sr2.config.models import PipelineConfig


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

        # Merge: root-level first, then pipeline: dict overrides
        merged = {**root_level, **nested_level}
        return PipelineConfig(**merged)

    def _load_with_inheritance(self, path: Path) -> dict:
        """Recursively load and merge configs following extends chain."""
        path_str = str(path)
        if path_str in self._loading_chain:
            raise ValueError(f"Circular config inheritance detected: {path_str}")
        self._loading_chain.add(path_str)

        with open(path) as f:
            config = yaml.safe_load(f) or {}

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
        """Resolve an extends value to an absolute path."""
        if extends == "defaults":
            if self._defaults_path is None:
                raise FileNotFoundError("No defaults_path configured but config extends 'defaults'")
            return Path(self._defaults_path).resolve()
        elif extends == "agent":
            return (current_path.parent.parent / "agent.yaml").resolve()
        else:
            return (current_path.parent / extends).resolve()

    def load_from_dict(self, config: dict) -> PipelineConfig:
        """Load from a dict (for testing). No inheritance resolution."""
        return PipelineConfig(**config)

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
