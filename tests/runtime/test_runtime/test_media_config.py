"""Tests for MediaConfig and STTProviderConfig."""

from __future__ import annotations

import pytest

from sr2_runtime.config import MediaConfig, RuntimeConfig, STTProviderConfig


class TestSTTProviderConfig:
    def test_defaults(self):
        cfg = STTProviderConfig()
        assert cfg.provider == "openai_compatible"
        assert cfg.api_base is None
        assert cfg.model is None

    def test_custom_provider(self):
        cfg = STTProviderConfig(
            provider="groq",
            api_base="https://api.groq.com/openai/v1",
            model="whisper-large-v3-turbo",
        )
        assert cfg.provider == "groq"
        assert cfg.api_base == "https://api.groq.com/openai/v1"
        assert cfg.model == "whisper-large-v3-turbo"


class TestMediaConfig:
    def test_disabled_by_default(self):
        cfg = MediaConfig()
        assert cfg.enabled is False

    def test_enabled_with_stt(self):
        cfg = MediaConfig(
            enabled=True,
            stt=STTProviderConfig(
                api_base="http://localhost:8787/v1",
                model="Systran/faster-whisper-small",
            ),
        )
        assert cfg.enabled is True
        assert cfg.stt.api_base == "http://localhost:8787/v1"
        assert cfg.stt.model == "Systran/faster-whisper-small"

    def test_from_dict(self):
        """MediaConfig can be constructed from a dict (as from YAML)."""
        data = {
            "enabled": True,
            "stt": {
                "provider": "openai_compatible",
                "api_base": "http://localhost:8787/v1",
                "model": "whisper-1",
            },
        }
        cfg = MediaConfig(**data)
        assert cfg.enabled is True
        assert cfg.stt.model == "whisper-1"


class TestRuntimeConfigIncludesMedia:
    def test_default_media_in_runtime(self):
        cfg = RuntimeConfig()
        assert hasattr(cfg, "media")
        assert cfg.media.enabled is False

    def test_media_round_trip(self):
        cfg = RuntimeConfig(
            media=MediaConfig(
                enabled=True,
                stt=STTProviderConfig(api_base="http://stt:8787/v1"),
            )
        )
        dumped = cfg.model_dump()
        assert dumped["media"]["enabled"] is True
        assert dumped["media"]["stt"]["api_base"] == "http://stt:8787/v1"
