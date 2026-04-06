"""Tests for MediaConfig and STTProviderConfig."""

from __future__ import annotations


from sr2_runtime.config import (
    DocumentMediaConfig,
    MediaConfig,
    PhotoMediaConfig,
    RuntimeConfig,
    STTProviderConfig,
    VoiceMediaConfig,
)


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


class TestVoiceMediaConfig:
    def test_disabled_by_default(self):
        cfg = VoiceMediaConfig()
        assert cfg.enabled is False

    def test_enabled_with_stt(self):
        cfg = VoiceMediaConfig(
            enabled=True,
            stt=STTProviderConfig(
                api_base="http://localhost:8787/v1",
                model="Systran/faster-whisper-small",
            ),
        )
        assert cfg.enabled is True
        assert cfg.stt.api_base == "http://localhost:8787/v1"


class TestPhotoMediaConfig:
    def test_disabled_by_default(self):
        cfg = PhotoMediaConfig()
        assert cfg.enabled is False


class TestDocumentMediaConfig:
    def test_disabled_by_default(self):
        cfg = DocumentMediaConfig()
        assert cfg.enabled is False


class TestMediaConfig:
    def test_all_disabled_by_default(self):
        cfg = MediaConfig()
        assert cfg.voice.enabled is False
        assert cfg.photo.enabled is False
        assert cfg.document.enabled is False

    def test_enable_voice_only(self):
        cfg = MediaConfig(
            voice=VoiceMediaConfig(
                enabled=True,
                stt=STTProviderConfig(
                    api_base="http://localhost:8787/v1",
                    model="Systran/faster-whisper-small",
                ),
            ),
        )
        assert cfg.voice.enabled is True
        assert cfg.photo.enabled is False
        assert cfg.document.enabled is False
        assert cfg.voice.stt.api_base == "http://localhost:8787/v1"

    def test_enable_all(self):
        cfg = MediaConfig(
            voice=VoiceMediaConfig(enabled=True),
            photo=PhotoMediaConfig(enabled=True),
            document=DocumentMediaConfig(enabled=True),
        )
        assert cfg.voice.enabled is True
        assert cfg.photo.enabled is True
        assert cfg.document.enabled is True

    def test_from_dict(self):
        """MediaConfig can be constructed from a dict (as from YAML)."""
        data = {
            "voice": {
                "enabled": True,
                "stt": {
                    "provider": "openai_compatible",
                    "api_base": "http://localhost:8787/v1",
                    "model": "whisper-1",
                },
            },
            "photo": {"enabled": True},
            "document": {"enabled": False},
        }
        cfg = MediaConfig(**data)
        assert cfg.voice.enabled is True
        assert cfg.voice.stt.model == "whisper-1"
        assert cfg.photo.enabled is True
        assert cfg.document.enabled is False


class TestRuntimeConfigIncludesMedia:
    def test_default_media_in_runtime(self):
        cfg = RuntimeConfig()
        assert hasattr(cfg, "media")
        assert cfg.media.voice.enabled is False
        assert cfg.media.photo.enabled is False
        assert cfg.media.document.enabled is False

    def test_media_round_trip(self):
        cfg = RuntimeConfig(
            media=MediaConfig(
                voice=VoiceMediaConfig(
                    enabled=True,
                    stt=STTProviderConfig(api_base="http://stt:8787/v1"),
                ),
            )
        )
        dumped = cfg.model_dump()
        assert dumped["media"]["voice"]["enabled"] is True
        assert dumped["media"]["voice"]["stt"]["api_base"] == "http://stt:8787/v1"
        assert dumped["media"]["photo"]["enabled"] is False
