"""Tests for the bridge adapter registry."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sr2_bridge.adapters import get_execution_adapter


class TestGetExecutionAdapter:
    """Tests for get_execution_adapter() factory."""

    def test_unknown_adapter_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown execution adapter 'nope'"):
            get_execution_adapter("nope", {})

    def test_claude_code_adapter_instantiation(self):
        """Successful instantiation with default config values."""
        adapter = get_execution_adapter("claude_code", {})
        assert hasattr(adapter, "stream_execute")
        assert hasattr(adapter, "shutdown")

    def test_claude_code_adapter_with_custom_config(self):
        with patch("shutil.which", return_value="/usr/bin/claude"):
            adapter = get_execution_adapter("claude_code", {
                "path": "/usr/bin/claude",
                "timeout_seconds": 600,
                "max_concurrent": 5,
            })
        assert adapter._timeout == 600

    def test_claude_code_adapter_invalid_config(self):
        """Invalid config values are rejected by Pydantic validation."""
        with pytest.raises(Exception):  # ValidationError from Pydantic
            get_execution_adapter("claude_code", {
                "timeout_seconds": 1,  # ge=10 constraint
            })
