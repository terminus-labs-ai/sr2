"""Tests for SR2.get_raw_window() accessor (Fix 11: heartbeat context leak).

Verifies that the SR2 facade exposes a get_raw_window() method that resolves
the raw_window turn limit from the appropriate interface's CompactionConfig.

Implementation note for Agent C:
  The plan's pseudocode says `return self._config.compaction.raw_window` for the
  fallback path. This is wrong — `self._config` is SR2Config (a dataclass), which
  has no `compaction` attribute. The correct fallback is `self._conversation.raw_window`,
  which is wired from `agent_config.compaction.raw_window` during __init__. The
  interface path is correct: `self._router.route(interface_name).compaction.raw_window`.
"""

import os
import tempfile

import pytest

from sr2.config.models import CompactionConfig, MemoryConfig, PipelineConfig, RetrievalConfig, SummarizationConfig
from sr2.sr2 import SR2, SR2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_sr2(
    config_dir: str,
    agent_yaml: dict | None = None,
    preloaded_config: PipelineConfig | None = None,
) -> SR2:
    """Build a minimal SR2 instance for testing get_raw_window().

    Uses preloaded_config when provided so no agent.yaml file is required.
    When testing per-interface routing via filesystem, pass a config_dir that
    has an interfaces/ subdirectory with YAML files.
    """
    # When no preloaded_config is given and no agent.yaml exists, provide a
    # minimal PipelineConfig so SR2.__init__ does not attempt file-based loading.
    if preloaded_config is None:
        preloaded_config = PipelineConfig(
            memory=MemoryConfig(extract=False),
            summarization=SummarizationConfig(enabled=False),
            retrieval=RetrievalConfig(enabled=False),
        )

    return SR2(
        SR2Config(
            config_dir=config_dir,
            agent_yaml=agent_yaml or {},
            preloaded_config=preloaded_config,
        )
    )


def _write_yaml(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Tests: method existence and signature
# ---------------------------------------------------------------------------


class TestGetRawWindowExists:
    """get_raw_window() must exist on the SR2 facade."""

    def test_method_exists_on_sr2(self):
        """SR2 class must have a get_raw_window method."""
        assert hasattr(SR2, "get_raw_window"), (
            "SR2 must have a get_raw_window() method — it does not exist"
        )

    def test_method_is_callable(self):
        """get_raw_window must be callable (not a property or class variable)."""
        assert callable(getattr(SR2, "get_raw_window", None))


# ---------------------------------------------------------------------------
# Tests: base-config fallback (no router, no interface)
# ---------------------------------------------------------------------------


class TestGetRawWindowBaseConfigFallback:
    """When no interface-specific config exists, fall back to base config raw_window."""

    def test_returns_default_raw_window_when_no_interface(self):
        """Returns CompactionConfig default (5) when no interface is given."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_raw_window()
            assert result == 5, (
                f"Expected default raw_window=5, got {result}"
            )

    def test_returns_int(self):
        """Return value must be int."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            assert isinstance(sr2.get_raw_window(), int)

    def test_interface_name_none_uses_base_config(self):
        """Passing interface_name=None explicitly falls back to base config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_raw_window(interface_name=None)
            assert result == 5

    def test_unknown_interface_falls_back_to_base_config(self):
        """An unregistered interface_name falls back to base config, not an exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sr2 = _minimal_sr2(tmpdir)
            # "heartbeat" is not registered — should not raise
            result = sr2.get_raw_window(interface_name="heartbeat")
            assert result == 5

    def test_base_config_raw_window_via_preloaded_config(self):
        """Base config raw_window is read from the preloaded PipelineConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preloaded = PipelineConfig(
                compaction=CompactionConfig(raw_window=12),
                memory=MemoryConfig(extract=False),
                summarization=SummarizationConfig(enabled=False),
                retrieval=RetrievalConfig(enabled=False),
            )
            sr2 = _minimal_sr2(tmpdir, preloaded_config=preloaded)
            # No interface name -> base config fallback
            result = sr2.get_raw_window()
            assert result == 12


# ---------------------------------------------------------------------------
# Tests: per-interface config
# ---------------------------------------------------------------------------


class TestGetRawWindowPerInterface:
    """When an interface has its own pipeline config, use its raw_window."""

    def test_returns_interface_raw_window(self):
        """Returns raw_window from the interface's pipeline config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iface_dir = os.path.join(tmpdir, "interfaces")
            _write_yaml(
                os.path.join(iface_dir, "heartbeat.yaml"),
                "compaction:\n  raw_window: 3\n",
            )
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_raw_window(interface_name="heartbeat")
            assert result == 3, (
                f"Expected raw_window=3 from heartbeat interface config, got {result}"
            )

    def test_interface_raw_window_differs_from_default(self):
        """Per-interface raw_window overrides the global default of 5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iface_dir = os.path.join(tmpdir, "interfaces")
            _write_yaml(
                os.path.join(iface_dir, "heartbeat.yaml"),
                "compaction:\n  raw_window: 8\n",
            )
            sr2 = _minimal_sr2(tmpdir)
            hb_window = sr2.get_raw_window(interface_name="heartbeat")
            base_window = sr2.get_raw_window()
            assert hb_window == 8
            assert base_window == 5
            assert hb_window != base_window

    def test_multiple_interfaces_independent(self):
        """Two interfaces can have different raw_window values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iface_dir = os.path.join(tmpdir, "interfaces")
            _write_yaml(
                os.path.join(iface_dir, "heartbeat.yaml"),
                "compaction:\n  raw_window: 3\n",
            )
            _write_yaml(
                os.path.join(iface_dir, "telegram.yaml"),
                "compaction:\n  raw_window: 10\n",
            )
            sr2 = _minimal_sr2(tmpdir)
            assert sr2.get_raw_window(interface_name="heartbeat") == 3
            assert sr2.get_raw_window(interface_name="telegram") == 10

    def test_interface_without_explicit_compaction_uses_default(self):
        """Interface config with no compaction section falls back to default raw_window=5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iface_dir = os.path.join(tmpdir, "interfaces")
            # Pipeline config with no compaction section at all
            _write_yaml(
                os.path.join(iface_dir, "heartbeat.yaml"),
                "token_budget: 8000\n",
            )
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_raw_window(interface_name="heartbeat")
            # CompactionConfig default raw_window is 5
            assert result == 5

    def test_unregistered_interface_falls_back_to_base_window(self):
        """An interface name not registered in the router falls back to base raw_window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preloaded = PipelineConfig(
                compaction=CompactionConfig(raw_window=6),
                memory=MemoryConfig(extract=False),
                summarization=SummarizationConfig(enabled=False),
                retrieval=RetrievalConfig(enabled=False),
            )
            sr2 = _minimal_sr2(tmpdir, preloaded_config=preloaded)
            # "heartbeat" has no interfaces/heartbeat.yaml — not registered
            result = sr2.get_raw_window(interface_name="heartbeat")
            assert result == 6, (
                "Unregistered interface should fall back to base config raw_window=6"
            )


# ---------------------------------------------------------------------------
# Tests: return value is the compaction raw_window, not something else
# ---------------------------------------------------------------------------


class TestGetRawWindowSemantics:
    """get_raw_window() must return CompactionConfig.raw_window specifically."""

    def test_returns_compaction_raw_window_not_summarization(self):
        """Verify the value comes from CompactionConfig, not SummarizationConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iface_dir = os.path.join(tmpdir, "interfaces")
            # raw_window=4 in compaction, preserve_recent_turns=2 in summarization
            _write_yaml(
                os.path.join(iface_dir, "heartbeat.yaml"),
                "compaction:\n  raw_window: 4\nsummarization:\n  preserve_recent_turns: 2\n",
            )
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_raw_window(interface_name="heartbeat")
            # Must be 4 (compaction.raw_window), not 2 (summarization.preserve_recent_turns)
            assert result == 4

    def test_raw_window_zero_is_valid(self):
        """raw_window=0 is a valid config value and must be returned as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            iface_dir = os.path.join(tmpdir, "interfaces")
            _write_yaml(
                os.path.join(iface_dir, "heartbeat.yaml"),
                "compaction:\n  raw_window: 0\n",
            )
            sr2 = _minimal_sr2(tmpdir)
            result = sr2.get_raw_window(interface_name="heartbeat")
            assert result == 0
