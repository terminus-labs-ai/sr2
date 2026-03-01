"""Tests for the generic CLI runner (HOTFIX-04)."""

import os
import tempfile

import yaml

from runtime.cli import parse_args, resolve_name


class TestResolveName:
    """resolve_name() reads agent name from YAML or falls back."""

    def test_reads_from_agent_yaml(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "agent.yaml"), "w") as f:
            yaml.dump({"agent_name": "EDI"}, f)

        assert resolve_name(tmpdir) == "EDI"

    def test_falls_back_to_directory_name(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "agent.yaml"), "w") as f:
            yaml.dump({"token_budget": 8000}, f)  # No agent_name

        expected = os.path.basename(tmpdir)
        assert resolve_name(tmpdir) == expected

    def test_uses_override_when_provided(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "agent.yaml"), "w") as f:
            yaml.dump({"agent_name": "EDI"}, f)

        assert resolve_name(tmpdir, override="CustomName") == "CustomName"

    def test_missing_yaml_uses_directory_name(self):
        tmpdir = tempfile.mkdtemp()
        expected = os.path.basename(tmpdir)
        assert resolve_name(tmpdir) == expected


class TestParseArgs:
    """parse_args() handles CLI arguments correctly."""

    def test_config_dir_only(self):
        args = parse_args(["config/agents/edi"])
        assert args.config_dir == "config/agents/edi"
        assert args.http is False
        assert args.port == 8008
        assert args.name is None

    def test_http_with_port(self):
        args = parse_args(["config/agents/edi", "--http", "--port", "9000"])
        assert args.http is True
        assert args.port == 9000

    def test_name_override(self):
        args = parse_args(["config/agents/edi", "--name", "MyAgent"])
        assert args.name == "MyAgent"

    def test_log_level(self):
        args = parse_args(["config/agents/edi", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"
